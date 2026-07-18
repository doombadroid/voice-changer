use std::fs::File;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::TensorRef;

#[derive(Parser)]
struct Args {
    /// cpu | webgpu
    #[arg(long, default_value = "cpu")]
    ep: String,
    #[arg(long, default_value_t = 100)]
    iters: usize,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    /// enable ORT graph capture (webgpu only, needs static shapes)
    #[arg(long, default_value_t = false)]
    graph_capture: bool,
    /// paths
    #[arg(long, default_value = "native/golden")]
    golden: String,
    #[arg(long, default_value = "server/pretrain/content_vec_500.onnx")]
    contentvec: String,
    /// override synth model path (e.g. fp16 variant)
    #[arg(long)]
    synth: Option<String>,
    /// override contentvec model path (e.g. fp16 variant)
    #[arg(long)]
    contentvec_override: Option<String>,
}

fn npz_f32(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<f32>> {
    npz.by_name(&format!("{key}.npy"))
        .or_else(|_| npz.by_name(key))
        .with_context(|| format!("npz key {key}"))
}

fn npz_i64(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<i64>> {
    npz.by_name(&format!("{key}.npy"))
        .or_else(|_| npz.by_name(key))
        .with_context(|| format!("npz key {key}"))
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f32::NAN;
    }
    (a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n as f32).sqrt()
}

/// median abs cents error on frames voiced in BOTH
fn cents_err(a: &[f32], b: &[f32]) -> f32 {
    let mut errs: Vec<f32> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| **x > 0.0 && **y > 0.0)
        .map(|(x, y)| 1200.0 * (x / y).log2().abs())
        .collect();
    if errs.is_empty() {
        return f32::NAN;
    }
    errs.sort_by(|p, q| p.partial_cmp(q).unwrap());
    errs[errs.len() / 2]
}

struct Bench {
    iters: usize,
    warmup: usize,
}

impl Bench {
    fn run<F: FnMut() -> Result<Vec<f32>>>(&self, name: &str, mut f: F) -> Result<Vec<f32>> {
        for _ in 0..self.warmup {
            f()?;
        }
        let mut times: Vec<f64> = Vec::with_capacity(self.iters);
        let mut out = Vec::new();
        for _ in 0..self.iters {
            let t = Instant::now();
            out = f()?;
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = times[self.iters / 2];
        let p95 = times[((self.iters as f64 * 0.95) as usize).min(self.iters - 1)];
        println!("{name:12} p50 {p50:8.2} ms   p95 {p95:8.2} ms");
        Ok(out)
    }
}

fn oe(e: ort::Error) -> anyhow::Error {
    anyhow::anyhow!(e.to_string())
}

fn build_session(args: &Args, path: &str) -> Result<Session> {
    let b = Session::builder().map_err(oe)?;
    let mut b = match args.ep.as_str() {
        "webgpu" => {
            let ep = ort::ep::webgpu::WebGPU::default()
                .with_enable_graph_capture(args.graph_capture)
                .with_validation_mode(ort::ep::webgpu::ValidationMode::Disabled);
            b.with_execution_providers([ep.build().error_on_failure()]).map_err(|e| anyhow::anyhow!("{e}"))?
        }
        "cpu" => b,
        other => bail!("unknown ep {other}"),
    };
    b.commit_from_file(path).map_err(oe)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("== spike-ort  ep={}  graph_capture={}  ({}) ==", args.ep, args.graph_capture, ort::info());
    let bench = Bench { iters: args.iters, warmup: args.warmup };

    let mut npz = NpzReader::new(File::open(format!("{}/golden.npz", args.golden))?)?;
    let audio = npz_f32(&mut npz, "audio_16k")?;
    let feats_g = npz_f32(&mut npz, "feats")?;
    let f0_g = npz_f32(&mut npz, "f0_rmvpe")?;
    let pitch = npz_i64(&mut npz, "pitch_coarse")?;
    let pitchf = npz_f32(&mut npz, "pitchf")?;
    let rnd = npz_f32(&mut npz, "rnd")?;
    let nsf_noise = npz_f32(&mut npz, "nsf_noise")?;
    let audio_out_g = npz_f32(&mut npz, "audio_out")?;
    let mel_fcpe = npz_f32(&mut npz, "mel_fcpe")?;
    let latent_fcpe_g = npz_f32(&mut npz, "latent_fcpe")?;

    let mut chain_p50 = 0.0f64;

    // --- ContentVec ---
    let cv_path = args.contentvec_override.clone().unwrap_or_else(|| args.contentvec.clone());
    let mut cv = build_session(&args, &cv_path)?;
    let cv_input = cv.inputs()[0].name().to_string();
    let t0 = Instant::now();
    let out = bench.run("contentvec", || {
        let o = cv.run(ort::inputs![cv_input.as_str() => TensorRef::from_array_view(&audio).map_err(oe)?]).map_err(oe)?;
        let (_, data) = o["unit12"].try_extract_tensor::<f32>().map_err(oe)?;
        Ok(data.to_vec())
    })?;
    let _ = t0;
    println!("{:12} RMSE {:.3e}   (gate 1e-4)", "contentvec", rmse(&out, feats_g.as_slice().unwrap()));

    // --- RMVPE ---
    let mut rm = build_session(&args, &format!("{}/rmvpe_20231006.onnx", args.golden))?;
    let thr = ndarray::arr1(&[0.3f32]).into_dyn();
    let out = bench.run("rmvpe", || {
        let o = rm.run(ort::inputs![
            "waveform" => TensorRef::from_array_view(&audio).map_err(oe)?,
            "threshold" => TensorRef::from_array_view(&thr).map_err(oe)?
        ]).map_err(oe)?;
        let (_, data) = o["pitchf"].try_extract_tensor::<f32>().map_err(oe)?;
        Ok(data.to_vec())
    })?;
    println!("{:12} median cents err {:.2}   (gate 5)", "rmvpe", cents_err(&out, f0_g.as_slice().unwrap()));

    // --- fcpe ---
    let fcpe_path = format!("{}/fcpe_fp32.onnx", args.golden);
    if std::path::Path::new(&fcpe_path).exists() {
        let mut fc = build_session(&args, &fcpe_path)?;
        let out = bench.run("fcpe", || {
            let o = fc.run(ort::inputs!["mel" => TensorRef::from_array_view(&mel_fcpe).map_err(oe)?])?;
            let (_, data) = o["latent"].try_extract_tensor::<f32>().map_err(oe)?;
            Ok(data.to_vec())
        })?;
        println!("{:12} latent RMSE {:.3e}   (gate 1e-4)", "fcpe", rmse(&out, latent_fcpe_g.as_slice().unwrap()));
    }

    // --- Synthesizer ---
    let syn_path = args.synth.clone().unwrap_or_else(|| format!("{}/synth_alexjones_fp32.onnx", args.golden));
    let mut syn = build_session(&args, &syn_path)?;
    let t = feats_g.shape()[1] as i64;
    let p_len = ndarray::arr1(&[t]).into_dyn();
    let sid = ndarray::arr1(&[0i64]).into_dyn();
    let out = bench.run("synth", || {
        let o = syn.run(ort::inputs![
            "feats" => TensorRef::from_array_view(&feats_g).map_err(oe)?,
            "p_len" => TensorRef::from_array_view(&p_len).map_err(oe)?,
            "pitch" => TensorRef::from_array_view(&pitch).map_err(oe)?,
            "pitchf" => TensorRef::from_array_view(&pitchf).map_err(oe)?,
            "sid" => TensorRef::from_array_view(&sid).map_err(oe)?,
            "rnd" => TensorRef::from_array_view(&rnd).map_err(oe)?,
            "nsf_noise" => TensorRef::from_array_view(&nsf_noise).map_err(oe)?
        ]).map_err(oe)?;
        let (_, data) = o["audio"].try_extract_tensor::<f32>().map_err(oe)?;
        Ok(data.to_vec())
    })?;
    println!("{:12} RMSE {:.3e}   (gate 1e-3)", "synth", rmse(&out, audio_out_g.as_slice().unwrap()));

    let _ = &mut chain_p50;
    println!("done");
    Ok(())
}
