use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;

use crate::{audio, virtmic};

#[derive(Clone, Debug)]
pub struct LiveConfig {
    pub mic_name: String,
    /// capture node.name; None = auto-pick (usb-preferred hw source)
    pub input: Option<String>,
    pub block: usize,
    pub ctx_left: usize,
    pub crossfade: usize,
    pub lookahead: usize,
    pub pitch: i32,
    pub seed: u64,
    pub passthrough: bool,
    pub tone: bool,
    pub dump_input: Option<String>,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            mic_name: "vc_mic".into(),
            input: None,
            block: 2560,
            ctx_left: 4160,
            crossfade: 640,
            lookahead: 0,
            pitch: 0,
            seed: 1337,
            passthrough: false,
            tone: false,
            dump_input: None,
        }
    }
}

/// Shared live stats/controls. RMS values stored as milli-units, times in us.
#[derive(Default)]
pub struct LiveStats {
    pub in_rms_milli: AtomicU64,
    pub out_rms_milli: AtomicU64,
    pub hop_p50_us: AtomicU64,
    pub hop_p95_us: AtomicU64,
    pub hops: AtomicU64,
    pub misses: AtomicU64,
    pub internal_latency_ms: AtomicU64,
    /// live-tweakable
    pub pitch: AtomicI32,
    pub mute: AtomicBool,
    /// resolved capture node (informational)
    pub capture_node: std::sync::Mutex<String>,
}

pub struct LiveHandle {
    stop: Arc<AtomicBool>,
    pub stats: Arc<LiveStats>,
    pub cfg: LiveConfig,
    worker: Option<std::thread::JoinHandle<()>>,
    pw_tx: Option<pipewire::channel::Sender<()>>,
    pw_thread: Option<std::thread::JoinHandle<()>>,
    _mic: virtmic::VirtMic,
}

/// Find the repo root (server/pretrain + native/) from cwd or exe path.
pub fn find_repo_root() -> Option<std::path::PathBuf> {
    let is_root = |p: &std::path::Path| p.join("server/pretrain/content_vec_500.onnx").exists();
    if let Ok(cwd) = std::env::current_dir() {
        if is_root(&cwd) {
            return Some(cwd);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        while let Some(p) = d {
            if is_root(&p) {
                return Some(p);
            }
            d = p.parent().map(|q| q.to_path_buf());
        }
    }
    None
}

pub fn build_engine(cfg: &LiveConfig) -> Result<Option<vc_core::stream::StreamEngine>> {
    if cfg.passthrough {
        return Ok(None);
    }
    let root = find_repo_root()
        .ok_or_else(|| anyhow::anyhow!("cannot locate repo root (server/pretrain/...) from cwd or exe path"))?;
    let j = |p: &str| root.join(p).to_string_lossy().to_string();
    let ecfg = vc_core::engine::EngineCfg {
        contentvec_onnx: j("server/pretrain/content_vec_500.onnx"),
        fcpe_onnx: j("native/golden/fcpe_fp32.onnx"),
        synth_onnx: j("native/golden/synth_alexjones_fp32.onnx"),
        frontend_npz: j("native/golden/fcpe_frontend.npz"),
        ..Default::default()
    };
    let eng = vc_core::engine::Engine::new(ecfg)?;
    let scfg = vc_core::stream::StreamCfg {
        block: cfg.block,
        ctx_left: cfg.ctx_left,
        crossfade: cfg.crossfade,
        sola_search: 320,
        lookahead: cfg.lookahead,
        pitch_semitones: cfg.pitch,
        seed: cfg.seed,
    };
    Ok(Some(vc_core::stream::StreamEngine::new(eng, scfg)))
}

/// Start the live path. Engine is built BEFORE any audio-graph mutation.
pub fn start(cfg: LiveConfig) -> Result<LiveHandle> {
    let mut engine_stream = build_engine(&cfg)?;

    let mic = virtmic::VirtMic::create(&cfg.mic_name)?;
    let stop = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(LiveStats::default());
    stats.pitch.store(cfg.pitch, Ordering::Relaxed);

    let (cap_tx, mut cap_rx) = rtrb::RingBuffer::<f32>::new(64000);
    let (mut out_tx, out_rx) = rtrb::RingBuffer::<f32>::new(160000);
    let warm = Arc::new(AtomicBool::new(false));

    // worker
    let st = stats.clone();
    let stop_w = stop.clone();
    let wa = warm.clone();
    let block = cfg.block;
    let tone = cfg.tone;
    let passthrough = cfg.passthrough;
    let dump_path = cfg.dump_input.clone();
    let worker = std::thread::spawn(move || {
        let mut dump: Vec<f32> = Vec::new();
        let mut inbuf: Vec<f32> = Vec::with_capacity(block);
        let mut burst_t: Option<std::time::Instant> = None;
        let mut total_in: u64 = 0;
        let mut hop_win: Vec<f64> = Vec::new();
        let budget_ms = block as f64 / 16.0;
        loop {
            if stop_w.load(Ordering::Relaxed) {
                return;
            }
            while inbuf.len() < block {
                if stop_w.load(Ordering::Relaxed) {
                    return;
                }
                match cap_rx.pop() {
                    Ok(v) => inbuf.push(v),
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(2)),
                }
            }
            if tone {
                for (i, v) in inbuf.iter_mut().enumerate() {
                    let abs = total_in + i as u64;
                    let ph = abs % 32000;
                    *v = if ph < 1600 {
                        if ph == 0 {
                            burst_t = Some(std::time::Instant::now());
                        }
                        0.5 * (2.0 * std::f32::consts::PI * 1000.0 * (ph as f32) / 16000.0).sin()
                    } else {
                        0.0
                    };
                }
            }
            total_in += block as u64;
            let in_rms = (inbuf.iter().map(|v| v * v).sum::<f32>() / block as f32).sqrt();
            st.in_rms_milli.store((in_rms * 1000.0) as u64, Ordering::Relaxed);
            if dump_path.is_some() && dump.len() < 16000 * 30 {
                dump.extend_from_slice(&inbuf);
                if dump.len() >= 16000 * 30 {
                    if let Some(dp) = &dump_path {
                        write_wav16(dp, &dump);
                        eprintln!("DUMPED 30s input to {dp}");
                    }
                }
            }

            let t0 = std::time::Instant::now();
            let mut out: Vec<f32> = if let Some(se) = engine_stream.as_mut() {
                se.set_pitch(st.pitch.load(Ordering::Relaxed));
                match se.push(&inbuf) {
                    Ok(Some(b)) => b,
                    Ok(None) => Vec::new(),
                    Err(e) => {
                        eprintln!("engine error: {e}");
                        vec![0.0; block * 5 / 2]
                    }
                }
            } else {
                let n_out = block * 5 / 2;
                (0..n_out)
                    .map(|i| {
                        let pos = i as f32 * (block as f32 - 1.0) / (n_out as f32 - 1.0);
                        let j = pos as usize;
                        let f = pos - j as f32;
                        inbuf[j] * (1.0 - f) + inbuf[(j + 1).min(block - 1)] * f
                    })
                    .collect()
            };
            if st.mute.load(Ordering::Relaxed) {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            st.hops.fetch_add(1, Ordering::Relaxed);
            if ms > budget_ms && !passthrough {
                st.misses.fetch_add(1, Ordering::Relaxed);
            }
            hop_win.push(ms);
            if hop_win.len() > 64 {
                hop_win.remove(0);
            }
            let mut sorted = hop_win.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            st.hop_p50_us.store((sorted[sorted.len() / 2] * 1000.0) as u64, Ordering::Relaxed);
            st.hop_p95_us.store(
                (sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)] * 1000.0) as u64,
                Ordering::Relaxed,
            );

            if let Some(bt) = burst_t {
                let orms = if out.is_empty() {
                    0.0
                } else {
                    (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt()
                };
                if orms > 0.02 {
                    st.internal_latency_ms.store(bt.elapsed().as_millis() as u64, Ordering::Relaxed);
                    burst_t = None;
                }
            }
            if !out.is_empty() {
                let orms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
                st.out_rms_milli.store((orms * 1000.0) as u64, Ordering::Relaxed);
            }
            for v in out {
                let _ = out_tx.push(v);
            }
            if !wa.load(Ordering::Relaxed) && st.hops.load(Ordering::Relaxed) >= 2 {
                for _ in 0..(block * 5 / 4) {
                    let _ = out_tx.push(0.0);
                }
                wa.store(true, Ordering::Relaxed);
            }
            inbuf.clear();
        }
    });

    // pipewire thread w/ quit channel
    let (pw_tx, pw_rx) = pipewire::channel::channel::<()>();
    let cap_target = cfg.input.clone().or_else(virtmic::pick_hw_source);
    *stats.capture_node.lock().unwrap() = cap_target.clone().unwrap_or_else(|| "(default)".into());
    let mic_name = cfg.mic_name.clone();
    let pw_thread = std::thread::spawn(move || {
        let run = || -> Result<()> {
            let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
            let context = pipewire::context::ContextRc::new(&mainloop, None)?;
            let core = context.connect_rc(None)?;
            let mut play_id = None;
            for _ in 0..10 {
                play_id = virtmic::node_id(&mic_name);
                if play_id.is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let play_target = virtmic::node_serial(&mic_name).unwrap_or_else(|| mic_name.clone());
            let _cap = audio::capture_stream(&core, cap_target.as_deref(), cap_tx)?;
            let _play = audio::playback_stream(&core, &play_target, play_id, 40000, out_rx, warm)?;
            audio::link_playback_to_mic(mic_name.clone());
            let ml = mainloop.clone();
            let _rx = pw_rx.attach(mainloop.loop_(), move |_| ml.quit());
            mainloop.run();
            Ok(())
        };
        if let Err(e) = run() {
            eprintln!("pipewire thread error: {e}");
        }
    });

    Ok(LiveHandle {
        stop,
        stats,
        cfg,
        worker: Some(worker),
        pw_tx: Some(pw_tx),
        pw_thread: Some(pw_thread),
        _mic: mic,
    })
}

impl LiveHandle {
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(tx) = self.pw_tx.take() {
            let _ = tx.send(());
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
        if let Some(p) = self.pw_thread.take() {
            let _ = p.join();
        }
        // mic teardown happens in VirtMic::drop
    }
}

impl Drop for LiveHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn write_wav16(path: &str, samples: &[f32]) {
    let mut bytes: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16000u32.to_le_bytes());
    bytes.extend_from_slice(&32000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        bytes.extend_from_slice(&(((s.clamp(-1.0, 1.0)) * 32767.0) as i16).to_le_bytes());
    }
    let _ = std::fs::write(path, bytes);
}
