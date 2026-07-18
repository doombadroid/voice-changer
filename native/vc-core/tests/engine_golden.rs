use std::fs::File;

use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use vc_core::engine::{Engine, EngineCfg};

fn npz_f32(npz: &mut NpzReader<File>, key: &str) -> ArrayD<f32> {
    npz.by_name(&format!("{key}.npy")).or_else(|_| npz.by_name(key)).unwrap()
}

/// Full single-pass chain vs golden audio_out_v2fcpe (same pitch source, same
/// injected noise) — the M1 correctness anchor.
#[test]
fn engine_matches_golden_chain() {
    if !std::path::Path::new("../golden/golden.npz").exists() {
        eprintln!("goldens missing; skipping");
        return;
    }
    let mut npz = NpzReader::new(File::open("../golden/golden.npz").unwrap()).unwrap();
    let audio = npz_f32(&mut npz, "audio_16k");
    let rnd = npz_f32(&mut npz, "rnd_v2");
    let noise = npz_f32(&mut npz, "nsf_noise_v2");
    let golden = npz_f32(&mut npz, "audio_out_v2fcpe");

    let cfg = EngineCfg {
        contentvec_onnx: "../../server/pretrain/content_vec_500.onnx".into(),
        fcpe_onnx: "../golden/fcpe_fp32.onnx".into(),
        synth_onnx: "../golden/synth_alexjones_fp32.onnx".into(),
        frontend_npz: "../golden/fcpe_frontend.npz".into(),
        ..Default::default()
    };
    let mut eng = Engine::new(cfg).unwrap();
    let (out, times) = eng
        .convert_with_noise(audio.as_slice().unwrap(), 0, rnd, noise)
        .unwrap();

    let g = golden.as_slice().unwrap();
    assert_eq!(out.len(), g.len(), "length {} vs golden {}", out.len(), g.len());
    let rmse =
        (out.iter().zip(g).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / g.len() as f32).sqrt();
    eprintln!("engine chain RMSE {rmse:.3e}  times {times:?}");
    assert!(rmse < 1e-3, "chain RMSE {rmse} >= 1e-3");
}
