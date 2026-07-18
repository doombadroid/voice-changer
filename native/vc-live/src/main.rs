use anyhow::Result;
use clap::Parser;
use vc_live::{livecore, virtmic};

#[derive(Parser)]
#[command(about = "vc-native live voice changer (M2) - CLI")]
struct Args {
    /// enumerate pipewire audio nodes and exit
    #[arg(long, default_value_t = false)]
    probe: bool,
    #[arg(long, default_value = "vc_mic")]
    mic_name: String,
    /// capture target node name (default: auto-pick usb-preferred hw mic)
    #[arg(long)]
    input: Option<String>,
    #[arg(long, default_value_t = false)]
    passthrough: bool,
    #[arg(long, default_value_t = 0)]
    pitch: i32,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    #[arg(long, default_value_t = 2560)]
    block: usize,
    #[arg(long, default_value_t = 4160)]
    ctx_left: usize,
    #[arg(long, default_value_t = 640)]
    crossfade: usize,
    #[arg(long, default_value_t = 0)]
    lookahead: usize,
    /// replace mic input with 1 kHz tone bursts (path/latency test)
    #[arg(long, default_value_t = false)]
    tone: bool,
    /// dump 30 s of worker input to this wav (diagnosis)
    #[arg(long)]
    dump_input: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    pipewire::init();
    if args.probe {
        vc_live::probe()?;
        unsafe { pipewire::deinit() };
        return Ok(());
    }

    virtmic::install_guards();
    let cfg = livecore::LiveConfig {
        mic_name: args.mic_name,
        input: args.input,
        block: args.block,
        ctx_left: args.ctx_left,
        crossfade: args.crossfade,
        lookahead: args.lookahead,
        pitch: args.pitch,
        seed: args.seed,
        passthrough: args.passthrough,
        tone: args.tone,
        dump_input: args.dump_input,
    };
    let handle = livecore::start(cfg)?;
    eprintln!(
        "live: {} -> '{}' virtual source. Ctrl-C to stop.",
        handle.stats.capture_node.lock().unwrap(),
        handle.cfg.mic_name
    );
    use std::sync::atomic::Ordering;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let s = &handle.stats;
        eprintln!(
            "hops {}  p50 {:.1}ms p95 {:.1}ms  misses {}  in_rms {:.3}  lat {}ms",
            s.hops.load(Ordering::Relaxed),
            s.hop_p50_us.load(Ordering::Relaxed) as f64 / 1000.0,
            s.hop_p95_us.load(Ordering::Relaxed) as f64 / 1000.0,
            s.misses.load(Ordering::Relaxed),
            s.in_rms_milli.load(Ordering::Relaxed) as f64 / 1000.0,
            s.internal_latency_ms.load(Ordering::Relaxed),
        );
    }
}
