use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

mod audio;
mod virtmic;

#[derive(Parser)]
#[command(about = "vc-native live voice changer (M2)")]
struct Args {
    /// enumerate pipewire audio nodes and exit
    #[arg(long, default_value_t = false)]
    probe: bool,
    /// create the virtual mic, hold until Ctrl-C (lifecycle test)
    #[arg(long, default_value_t = false)]
    hold_mic: bool,
    /// virtual mic node name
    #[arg(long, default_value = "vc_mic")]
    mic_name: String,
    /// capture target node name/id (default: system default source)
    #[arg(long)]
    input: Option<String>,
    /// passthrough (no conversion) - audio path test
    #[arg(long, default_value_t = false)]
    passthrough: bool,
    /// pitch shift semitones
    #[arg(long, default_value_t = 0)]
    pitch: i32,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// block size, 16k samples (multiple of 320)
    #[arg(long, default_value_t = 2560)]
    block: usize,
    /// left context, 16k samples (multiple of 320)
    #[arg(long, default_value_t = 4160)]
    ctx_left: usize,
    /// crossfade, 16k samples (multiple of 320)
    #[arg(long, default_value_t = 640)]
    crossfade: usize,
    /// extra lookahead, 16k samples (multiple of 320); adds latency
    #[arg(long, default_value_t = 0)]
    lookahead: usize,
    /// replace mic input with a 1 kHz tone burst every 2 s (latency/path test)
    #[arg(long, default_value_t = false)]
    tone: bool,
}

/// Find the repo root (dir containing server/pretrain + native/) from cwd or
/// walking up from the executable path, so vc-live runs from anywhere.
fn find_repo_root() -> Option<std::path::PathBuf> {
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

fn probe() -> Result<()> {
    let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
    let context = pipewire::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry()?;

    let ml = mainloop.clone();
    // stop the loop once the initial registry dump has flushed (sync trick)
    let pending = core.sync(0)?;
    let _sync_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pipewire::core::PW_ID_CORE && seq == pending {
                ml.quit();
            }
        })
        .register();

    let _reg_listener = registry
        .add_listener_local()
        .global(|global| {
            if let Some(props) = &global.props {
                if let Some(class) = props.get("media.class") {
                    if class.starts_with("Audio/") {
                        println!(
                            "{:5} {:24} {}",
                            global.id,
                            class,
                            props.get("node.description").or(props.get("node.name")).unwrap_or("?")
                        );
                    }
                }
            }
        })
        .register();

    mainloop.run();
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    pipewire::init();
    if args.probe {
        probe()?;
        unsafe { pipewire::deinit() };
        return Ok(());
    }

    virtmic::install_guards();
    let _mic = virtmic::VirtMic::create(&args.mic_name)?;

    if args.hold_mic {
        eprintln!("virtual mic '{}' up; Ctrl-C to exit", args.mic_name);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    }

    // build engine FIRST (before touching audio graph) so path errors are
    // clean failures, not a mic-teardown cascade mid-panic
    let engine_stream = if args.passthrough {
        None
    } else {
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
        let cfg = vc_core::stream::StreamCfg {
            block: args.block,
            ctx_left: args.ctx_left,
            crossfade: args.crossfade,
            sola_search: 320,
            lookahead: args.lookahead,
            pitch_semitones: args.pitch,
            seed: args.seed,
        };
        Some(vc_core::stream::StreamEngine::new(eng, cfg))
    };

    // rings: 4 s of 16k input, 4 s of 40k output
    let (cap_tx, mut cap_rx) = rtrb::RingBuffer::<f32>::new(64000);
    let (mut out_tx, out_rx) = rtrb::RingBuffer::<f32>::new(160000);
    let warm = Arc::new(AtomicBool::new(false));

    // worker: engine or passthrough
    let wa = warm.clone();
    let block = args.block;
    let passthrough = args.passthrough;
    let tone = args.tone;
    std::thread::spawn(move || {
        let mut engine_stream = engine_stream;
        eprintln!("worker up ({})", if passthrough { "passthrough" } else { "engine" });

        let mut inbuf: Vec<f32> = Vec::with_capacity(block);
        let mut burst_t: Option<std::time::Instant> = None;
        let mut total_in: u64 = 0;
        let mut in_rms_acc: f32 = 0.0;
        let mut misses = 0u64;
        let mut hops = 0u64;
        let mut hop_ms: Vec<f64> = Vec::new();
        let budget_ms = block as f64 / 16.0;
        let mut last_stats = std::time::Instant::now();
        loop {
            while inbuf.len() < block {
                match cap_rx.pop() {
                    Ok(v) => inbuf.push(v),
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(2)),
                }
            }
            if tone {
                // 100 ms 1 kHz burst at the start of every 2 s period; log burst wall time
                for (i, v) in inbuf.iter_mut().enumerate() {
                    let abs = total_in + i as u64;
                    let phase_in_period = abs % 32000;
                    *v = if phase_in_period < 1600 {
                        if phase_in_period == 0 {
                            burst_t = Some(std::time::Instant::now());
                        }
                        0.5 * (2.0 * std::f32::consts::PI * 1000.0 * (phase_in_period as f32) / 16000.0).sin()
                    } else {
                        0.0
                    };
                }
            }
            total_in += block as u64;
            in_rms_acc += inbuf.iter().map(|v| v * v).sum::<f32>();
            let t0 = std::time::Instant::now();
            let out: Vec<f32> = if let Some(se) = engine_stream.as_mut() {
                match se.push(&inbuf) {
                    Ok(Some(b)) => b,
                    Ok(None) => Vec::new(),
                    Err(e) => {
                        eprintln!("engine error: {e}");
                        vec![0.0; block * 5 / 2]
                    }
                }
            } else {
                // naive linear 16k -> 40k upsample (test path only)
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
            // internal latency: burst injection -> converted energy in output
            if let Some(bt) = burst_t {
                let orms = if out.is_empty() { 0.0 } else { (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt() };
                if orms > 0.02 {
                    eprintln!("INTERNAL_LATENCY_MS {:.0}", bt.elapsed().as_secs_f64() * 1e3);
                    burst_t = None;
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            hops += 1;
            hop_ms.push(ms);
            if ms > budget_ms && !passthrough {
                misses += 1;
            }
            for v in out {
                let _ = out_tx.push(v);
            }
            if !wa.load(Ordering::Relaxed) && hops >= 2 {
                // prime a little silence to absorb scheduling jitter, then go live
                for _ in 0..(block * 5 / 4) {
                    let _ = out_tx.push(0.0);
                }
                wa.store(true, Ordering::Relaxed);
            }
            inbuf.clear();
            if last_stats.elapsed().as_secs() >= 5 {
                let mut sorted = hop_ms.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = sorted[sorted.len() / 2];
                let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)];
                let in_rms = (in_rms_acc / (total_in.max(1) as f32)).sqrt();
                in_rms_acc = 0.0;
                eprintln!(
                    "hops {hops}  p50 {p50:.1}ms p95 {p95:.1}ms budget {budget_ms:.0}ms  misses {misses}  in_rms {in_rms:.4}  cap_xrun {}  out_underrun {}",
                    audio::CAPTURE_XRUNS.load(Ordering::Relaxed),
                    audio::PLAYBACK_UNDERRUNS.load(Ordering::Relaxed),
                );
                last_stats = std::time::Instant::now();
            }
        }
    });

    let mainloop = pipewire::main_loop::MainLoopRc::new(None)?;
    let context = pipewire::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let cap_target = args.input.clone().or_else(virtmic::pick_hw_source);
    eprintln!("capture from: {}", cap_target.as_deref().unwrap_or("(default)"));
    // freshly-created node can lag pw-dump; retry briefly
    let mut play_id = None;
    for _ in 0..10 {
        play_id = virtmic::node_id(&args.mic_name);
        if play_id.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let play_target = virtmic::node_serial(&args.mic_name).unwrap_or_else(|| args.mic_name.clone());
    eprintln!("playback target serial: {play_target} id: {play_id:?}");
    let _cap = audio::capture_stream(&core, cap_target.as_deref(), cap_tx)?;
    let _play = audio::playback_stream(&core, &play_target, play_id, 40000, out_rx, warm)?;
    audio::link_playback_to_mic(args.mic_name.clone());
    eprintln!("live: mic -> '{}' virtual source. Ctrl-C to stop.", args.mic_name);
    mainloop.run();
    unsafe { pipewire::deinit() };
    Ok(())
}
