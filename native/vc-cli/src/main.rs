use anyhow::{bail, Context, Result};
use clap::Parser;
use vc_core::dsp::resample::Resampler;
use vc_core::engine::{seeded_rng, Engine, EngineCfg};
use vc_core::stream::{StreamCfg, StreamEngine};

#[derive(Parser)]
#[command(about = "vc-native offline converter (M1)")]
struct Args {
    input: String,
    output: String,
    /// pitch shift in semitones
    #[arg(long, default_value_t = 0)]
    pitch: i32,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// single-pass whole-file conversion (no chunking)
    #[arg(long, default_value_t = false)]
    single_pass: bool,
    /// chunk block size in 16k samples (multiple of 320)
    #[arg(long, default_value_t = 2560)]
    block: usize,
    /// left context in 16k samples (multiple of 320)
    #[arg(long, default_value_t = 4160)]
    ctx_left: usize,
    /// print per-hop stage timings
    #[arg(long, default_value_t = false)]
    bench: bool,
    /// model/asset paths (defaults assume repo root cwd)
    #[arg(long, default_value = "server/pretrain/content_vec_500.onnx")]
    contentvec: String,
    #[arg(long, default_value = "native/golden/fcpe_fp32.onnx")]
    fcpe: String,
    #[arg(long, default_value = "native/golden/synth_alexjones_fp32.onnx")]
    synth: String,
    #[arg(long, default_value = "native/golden/fcpe_frontend.npz")]
    frontend: String,
}

fn read_wav_mono(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {path}"))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / max)).collect::<Result<_, _>>()?
        }
    };
    if ch == 0 {
        bail!("zero channels");
    }
    let mono: Vec<f32> = samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    Ok((mono, spec.sample_rate))
}

fn write_wav(path: &str, samples: &[f32], sr: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (audio, sr) = read_wav_mono(&args.input)?;
    eprintln!("in: {} samples @ {sr} Hz", audio.len());

    let mut a16 = if sr == 16000 {
        audio
    } else {
        let mut down = Resampler::new(sr as usize, 16000)?;
        let mut v = down.process(&audio)?;
        v.extend(down.flush()?);
        v
    };
    // guard against out-of-range floats confusing the models
    for v in a16.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }

    let cfg = EngineCfg {
        contentvec_onnx: args.contentvec.clone(),
        fcpe_onnx: args.fcpe.clone(),
        synth_onnx: args.synth.clone(),
        frontend_npz: args.frontend.clone(),
        ..Default::default()
    };
    let mut eng = Engine::new(cfg)?;
    let mut rng = seeded_rng(args.seed);
    #[allow(unused_assignments)]
    let mut write_out_sr = eng.out_sr();

    let t0 = std::time::Instant::now();
    let out = if args.single_pass {
        let (out, times) = eng.convert(&a16, args.pitch, &mut rng)?;
        eprintln!(
            "stages ms: contentvec {:.1}  pitch {:.1}  synth {:.1}",
            times.contentvec, times.pitch, times.synth
        );
        out
    } else {
        let _ = &mut rng;
        let scfg = StreamCfg {
            block: args.block,
            ctx_left: args.ctx_left,
            pitch_semitones: args.pitch,
            seed: args.seed,
            ..Default::default()
        };
        let block = scfg.block;
        let mut se = StreamEngine::new(eng, scfg);
        let mut out: Vec<f32> = Vec::new();
        let mut hop_times: Vec<f64> = Vec::new();
        let mut chunks = a16.chunks_exact(block);
        for c in chunks.by_ref() {
            let t = std::time::Instant::now();
            if let Some(mut b) = se.push(c)? {
                out.append(&mut b);
            }
            hop_times.push(t.elapsed().as_secs_f64() * 1e3);
            if args.bench {
                let lt = se.last_times;
                eprintln!(
                    "hop {:.1} ms (cv {:.1} pitch {:.1} synth {:.1}) vs block {:.0} ms",
                    hop_times.last().unwrap(),
                    lt.contentvec,
                    lt.pitch,
                    lt.synth,
                    block as f64 / 16.0
                );
            }
        }
        // final partial chunk: zero-pad to a full block so tail audio isn't dropped
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut last = rem.to_vec();
            last.resize(block, 0.0);
            if let Some(mut b) = se.push(&last)? {
                out.append(&mut b);
            }
        }
        hop_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !hop_times.is_empty() {
            let p50 = hop_times[hop_times.len() / 2];
            let p95 = hop_times[((hop_times.len() as f64 * 0.95) as usize).min(hop_times.len() - 1)];
            let budget = block as f64 / 16.0;
            eprintln!(
                "hops: {}  p50 {:.1} ms  p95 {:.1} ms  block-budget {:.0} ms  headroom {:.0}%",
                hop_times.len(),
                p50,
                p95,
                budget,
                (1.0 - p50 / budget) * 100.0
            );
        }
        write_out_sr = se.out_sr();
        out
    };
    let wall = t0.elapsed().as_secs_f64();
    let audio_secs = a16.len() as f64 / 16000.0;
    eprintln!("xRT: {:.2} ({}s audio in {:.2}s)", audio_secs / wall, audio_secs as i64, wall);

    write_wav(&args.output, &out, write_out_sr)?;
    eprintln!("out: {} samples @ {} Hz -> {}", out.len(), write_out_sr, args.output);
    Ok(())
}
