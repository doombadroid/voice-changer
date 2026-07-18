"""Dump fcpe frontend assets (mel filterbank, hann window, decoder tables) to npz.

Everything the rust MelFrontend/F0Decoder needs to match torchfcpe exactly:
- mel_basis [128, 513]  (librosa slaney mel, sr16000 nfft1024 fmin0 fmax8000)
- hann_window [1024]
- cent_table [360]      (for latent -> cents decode)
- scalars: clip_val 1e-5, hop 160, win 1024, n_fft 1024, pad_left 432, pad_right 432

Semantics replicated in rust (from torchfcpe/mel_extractor.py + models.py):
- pad reflect (constant only if signal shorter than pad), stft center=False
- mag = sqrt(re^2 + im^2 + 1e-9); mel = mel_basis @ mag; log(clamp(mel, 1e-5))
- frame count forced to T//hop + 1 (pad-repeat or trim last)

Run: server/.venv/bin/python native/tools/dump_fcpe_frontend.py native/golden/fcpe_frontend.npz
"""
import sys

import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model


def main() -> None:
    out = sys.argv[1]
    m = spawn_bundled_infer_model(device="cpu")
    mel_mod = m.wav2mel.mel_extractor
    core = m.model

    mel_basis = mel_mod.mel_basis.numpy().astype(np.float32)
    hann = torch.hann_window(int(mel_mod.win_size)).numpy().astype(np.float32)
    cent_table = core.cent_table.numpy().astype(np.float32).ravel()

    np.savez(
        out,
        mel_basis=mel_basis,
        hann_window=hann,
        cent_table=cent_table,
        scalars=np.array(
            [
                float(mel_mod.n_fft),
                float(mel_mod.win_size),
                float(mel_mod.hop_length),
                float(mel_mod.clip_val),
                float(mel_mod.fmin),
                float(mel_mod.fmax),
            ],
            dtype=np.float32,
        ),
    )
    print(
        {
            "mel_basis": mel_basis.shape,
            "hann": hann.shape,
            "cent_table": cent_table.shape,
            "cent_range": (float(cent_table.min()), float(cent_table.max())),
        }
    )


if __name__ == "__main__":
    main()
