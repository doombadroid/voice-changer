extern crate alloc;

use std::fs::File;
use std::time::Instant;

use anyhow::{Context, Result};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;

mod fcpe_model {
    include!(concat!(env!("OUT_DIR"), "/model/fcpe_fp32.rs"));
}
mod synth_model {
    include!(concat!(env!("OUT_DIR"), "/model/synth_alexjones_static49.rs"));
}
mod cv_model {
    include!(concat!(env!("OUT_DIR"), "/model/content_vec_500.rs"));
}

use burn::prelude::*;

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
    (a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n as f32).sqrt()
}

fn bench<T, F: FnMut() -> T>(name: &str, warmup: usize, iters: usize, mut f: F) -> T {
    for _ in 0..warmup {
        f();
    }
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    let mut out = None;
    for _ in 0..iters {
        let t = Instant::now();
        out = Some(f());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[iters / 2];
    let p95 = times[((iters as f64 * 0.95) as usize).min(iters - 1)];
    println!("{name:12} p50 {p50:8.2} ms   p95 {p95:8.2} ms");
    out.unwrap()
}

struct Golden {
    audio: (Vec<f32>, Vec<usize>),
    feats: (Vec<f32>, Vec<usize>),
    pitch: (Vec<i64>, Vec<usize>),
    pitchf: (Vec<f32>, Vec<usize>),
    rnd: (Vec<f32>, Vec<usize>),
    nsf_noise: (Vec<f32>, Vec<usize>),
    audio_out: Vec<f32>,
    mel: (Vec<f32>, Vec<usize>),
    latent: Vec<f32>,
}

fn load_golden(path: &str) -> Result<Golden> {
    let mut npz = NpzReader::new(File::open(path)?)?;
    let g = |a: ArrayD<f32>| -> (Vec<f32>, Vec<usize>) {
        let shape = a.shape().to_vec();
        (a.into_raw_vec_and_offset().0, shape)
    };
    let audio = g(npz_f32(&mut npz, "audio_16k")?);
    let feats = g(npz_f32(&mut npz, "feats")?);
    let pitch_a = npz_i64(&mut npz, "pitch_coarse")?;
    let pitch = (pitch_a.iter().copied().collect::<Vec<i64>>(), pitch_a.shape().to_vec());
    let pitchf = g(npz_f32(&mut npz, "pitchf")?);
    let rnd = g(npz_f32(&mut npz, "rnd")?);
    let nsf_noise = g(npz_f32(&mut npz, "nsf_noise")?);
    let audio_out = g(npz_f32(&mut npz, "audio_out")?).0;
    let mel = g(npz_f32(&mut npz, "mel_fcpe")?);
    let latent = g(npz_f32(&mut npz, "latent_fcpe")?).0;
    Ok(Golden { audio, feats, pitch, pitchf, rnd, nsf_noise, audio_out, mel, latent })
}

fn t3<B: Backend>(d: &(Vec<f32>, Vec<usize>), dev: &B::Device) -> Tensor<B, 3> {
    Tensor::from_data(burn::tensor::TensorData::new(d.0.clone(), [d.1[0], d.1[1], d.1[2]]), dev)
}

fn out_dir_path(name: &str) -> String {
    format!("{}/model/{name}", env!("OUT_DIR"))
}

fn run_backend<B: Backend>(label: &str, dev: &B::Device, g: &Golden, iters: usize) {
    println!("== backend: {label} ==");
    let warmup = 10;

    // fcpe
    let m = fcpe_model::Model::<B>::from_file(out_dir_path("fcpe_fp32.bpk"), dev);
    let mel = t3::<B>(&g.mel, dev);
    let out = bench("fcpe", warmup, iters, || {
        m.forward(mel.clone()).into_data().to_vec::<f32>().unwrap()
    });
    println!("{:12} latent RMSE {:.3e}   (gate 1e-4)", "fcpe", rmse(&out, &g.latent));
    drop(m);

    // synth (static T=49)
    let m = synth_model::Model::<B>::from_file(out_dir_path("synth_alexjones_static49.bpk"), dev);
    let feats = t3::<B>(&g.feats, dev);
    let pitchf: Tensor<B, 2> =
        Tensor::from_data(burn::tensor::TensorData::new(g.pitchf.0.clone(), [g.pitchf.1[0], g.pitchf.1[1]]), dev);
    let pitch: Tensor<B, 2, Int> =
        Tensor::from_data(burn::tensor::TensorData::new(g.pitch.0.clone(), [g.pitch.1[0], g.pitch.1[1]]), dev);
    let p_len: Tensor<B, 1, Int> =
        Tensor::from_data(burn::tensor::TensorData::new(vec![g.feats.1[1] as i64], [1]), dev);
    let sid: Tensor<B, 1, Int> = Tensor::from_data(burn::tensor::TensorData::new(vec![0i64], [1]), dev);
    let rnd = t3::<B>(&g.rnd, dev);
    let nsf_noise = t3::<B>(&g.nsf_noise, dev);
    let out = bench("synth", warmup, iters, || {
        m.forward(feats.clone(), p_len.clone(), pitch.clone(), pitchf.clone(), sid.clone(), rnd.clone(), nsf_noise.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    });
    println!("{:12} RMSE {:.3e}   (gate 1e-3)", "synth", rmse(&out, &g.audio_out));
    drop(m);

    // contentvec
    let m = cv_model::Model::<B>::from_file(out_dir_path("content_vec_500.bpk"), dev);
    let audio: Tensor<B, 2> =
        Tensor::from_data(burn::tensor::TensorData::new(g.audio.0.clone(), [g.audio.1[0], g.audio.1[1]]), dev);
    let out = bench("contentvec", warmup, iters, || {
        let (_units9, unit12, _unit12s) = m.forward(audio.clone());
        unit12.into_data().to_vec::<f32>().unwrap()
    });
    println!("{:12} RMSE {:.3e}   (gate 1e-4)", "contentvec", rmse(&out, &g.feats.0));
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize =
        args.iter().position(|a| a == "--iters").map(|i| args[i + 1].parse().unwrap()).unwrap_or(50);
    let only = args.iter().position(|a| a == "--backend").map(|i| args[i + 1].clone());
    let g = load_golden("native/golden/golden.npz")?;

    if only.is_none() || only.as_deref() == Some("vulkan") {
        let dev = burn::backend::wgpu::WgpuDevice::default();
        run_backend::<burn::backend::Vulkan>("vulkan", &dev, &g, iters);
    }
    if only.as_deref() == Some("wgsl") {
        let dev = burn::backend::wgpu::WgpuDevice::default();
        run_backend::<burn::backend::Wgpu>("wgpu-wgsl", &dev, &g, iters);
    }
    if only.is_none() || only.as_deref() == Some("ndarray") {
        let dev = Default::default();
        run_backend::<burn::backend::NdArray>("ndarray-cpu", &dev, &g, iters);
    }
    Ok(())
}
