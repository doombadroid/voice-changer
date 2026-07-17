"""Generate golden tensors for spike correctness gates. All CPU, all deterministic.

Chain: demo wav 1s@16k -> contentvec onnx (unit12) -> rmvpe onnx f0 -> coarse pitch
-> synth onnx (seeded rnd + nsf_noise) -> audio_out. Plus fcpe mel/latent/f0.

Run: server/.venv/bin/python native/tools/make_golden.py
"""
import json

import librosa
import numpy as np
import onnxruntime as ort_rt
import soundfile as sf
import torch

SEED = 1337


def main() -> None:
    wav, _sr = librosa.load("demo_alexjones.wav", sr=16000, mono=True)
    audio = wav[16000:32000].astype(np.float32)[None, :]  # [1,16000] - 1s..2s, first second is silence

    # --- ContentVec (existing onnx, CPU EP) ---
    cv = ort_rt.InferenceSession("server/pretrain/content_vec_500.onnx", providers=["CPUExecutionProvider"])
    cv_in = cv.get_inputs()[0]
    feed = audio if len(cv_in.shape) == 2 else audio[None, :]
    outs = cv.run(None, {cv_in.name: feed})
    names = [o.name for o in cv.get_outputs()]
    feats = outs[names.index("unit12")]  # [1,T,768]

    # --- RMVPE (onnx, CPU) ---
    rm = ort_rt.InferenceSession("native/golden/rmvpe_20231006.onnx", providers=["CPUExecutionProvider"])
    rm_names = [i.name for i in rm.get_inputs()]
    f0 = rm.run(None, {rm_names[0]: audio, rm_names[1]: np.array([0.3], dtype=np.float32)})[0]
    f0 = np.asarray(f0, dtype=np.float32).ravel()

    # --- coarse pitch (RVC convention) aligned to feats T ---
    T = feats.shape[1]
    f0r = np.interp(np.linspace(0, len(f0) - 1, T), np.arange(len(f0)), f0).astype(np.float32)
    f0_mel = 1127 * np.log(1 + f0r / 700)
    mel_min, mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    coarse = np.clip((f0_mel - mel_min) * 254 / (mel_max - mel_min) + 1, 1, 255)
    coarse = np.rint(np.where(f0r > 0, coarse, 1)).astype(np.int64)[None, :]
    pitchf = f0r[None, :]

    # --- Synthesizer (deterministic) ---
    syn = ort_rt.InferenceSession("native/golden/synth_alexjones_fp32.onnx", providers=["CPUExecutionProvider"])
    meta = json.loads(open("native/golden/synth_meta.json").read()) if False else {"upp": 400, "inter_channels": 192}
    rs = np.random.RandomState(SEED)
    rnd = rs.randn(1, meta["inter_channels"], T).astype(np.float32)
    nsf_noise = rs.randn(1, T * meta["upp"], 1).astype(np.float32)
    audio_out = syn.run(None, {
        "feats": feats, "p_len": np.array([T], dtype=np.int64), "pitch": coarse,
        "pitchf": pitchf, "sid": np.array([0], dtype=np.int64), "rnd": rnd,
        "nsf_noise": nsf_noise,
    })[0]  # 1-D [T*upp]

    # --- fcpe: mel via torchfcpe wav2mel, latent via onnx, f0 via python decoder ---
    from torchfcpe import spawn_bundled_infer_model

    m = spawn_bundled_infer_model(device="cpu")
    with torch.no_grad():
        mel = m.wav2mel(torch.from_numpy(audio).unsqueeze(-1), 16000)  # [1,T',128]
        fc = ort_rt.InferenceSession("native/golden/fcpe_fp32.onnx", providers=["CPUExecutionProvider"])
        latent = fc.run(None, {"mel": mel.numpy().astype(np.float32)})[0]
        cents = m.model.latent2cents_local_decoder(torch.from_numpy(latent), threshold=0.006)
        f0_fcpe = m.model.cent_to_f0(cents).numpy().ravel().astype(np.float32)

    np.savez(
        "native/golden/golden.npz",
        audio_16k=audio, feats=feats, f0_rmvpe=f0, pitch_coarse=coarse, pitchf=pitchf,
        rnd=rnd, nsf_noise=nsf_noise, audio_out=audio_out.astype(np.float32),
        mel_fcpe=mel.numpy().astype(np.float32), latent_fcpe=latent, f0_fcpe=f0_fcpe,
    )
    sf.write("native/golden/golden_out.wav", np.asarray(audio_out).ravel(), 40000)
    voiced = f0[f0 > 0]
    print(json.dumps({
        "feats": list(feats.shape), "f0_frames": len(f0), "f0_voiced": int(len(voiced)),
        "f0_median_hz": float(np.median(voiced)) if len(voiced) else 0.0,
        "audio_out": list(np.asarray(audio_out).shape),
        "mel_fcpe": list(mel.shape), "f0_fcpe_frames": len(f0_fcpe),
        "out_rms": float(np.sqrt(np.mean(np.asarray(audio_out) ** 2))),
    }))


if __name__ == "__main__":
    main()
