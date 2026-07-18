use std::fs::File;

use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use vc_core::dsp::mel::MelFrontend;

fn npz_f32(npz: &mut NpzReader<File>, key: &str) -> ArrayD<f32> {
    npz.by_name(&format!("{key}.npy")).or_else(|_| npz.by_name(key)).unwrap()
}

#[test]
fn mel_matches_golden() {
    let golden_path = "../golden/golden.npz";
    if !std::path::Path::new(golden_path).exists() {
        eprintln!("golden.npz missing - run native/tools/make_golden.py; skipping");
        return;
    }
    let mut npz = NpzReader::new(File::open(golden_path).unwrap()).unwrap();
    let audio = npz_f32(&mut npz, "audio_16k");
    let mel_g = npz_f32(&mut npz, "mel_fcpe");

    let fe = MelFrontend::from_npz("../golden/fcpe_frontend.npz").unwrap();
    let mel = fe.process(audio.as_slice().unwrap()).unwrap();

    assert_eq!(mel.shape()[0], mel_g.shape()[1], "frame count mismatch");
    assert_eq!(mel.shape()[1], mel_g.shape()[2]);
    let g = mel_g.as_slice().unwrap();
    let m = mel.as_slice().unwrap();
    let rmse = (m.iter().zip(g).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / m.len() as f32).sqrt();
    assert!(rmse < 1e-4, "mel RMSE {rmse} >= 1e-4");
    eprintln!("mel RMSE {rmse:.3e} over {} frames", mel.shape()[0]);
}
