"""Export fcpe core (mel -> f0 latent) to ONNX. Prints mel config JSON.

Graph: input mel[1,T,128] float32 -> output latent[1,T,360] float32 (sigmoid).
Decoding (local_argmax -> cents -> Hz) stays OUTSIDE the graph (plain math,
reimplemented natively; golden f0 uses the python decoder on this latent).
Mel extraction also stays outside (sr16000 n_fft1024 win1024 hop160 fmin0 fmax8000).

Run: server/.venv/bin/python native/tools/export_fcpe_onnx.py native/golden/fcpe_fp32.onnx
"""
import json
import sys

import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model


def main() -> None:
    out_path = sys.argv[1]
    m = spawn_bundled_infer_model(device="cpu")
    core = m.model.eval()
    mel_cfg = m.get_mel_config()

    T = 100
    mel = torch.randn(1, T, int(mel_cfg["num_mels"]))

    class MelToLatent(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, mel):
            return self.core(mel)

    torch.onnx.export(
        MelToLatent(core),
        (mel,),
        out_path,
        opset_version=17,
        dynamo=False,
        input_names=["mel"],
        output_names=["latent"],
        dynamic_axes={"mel": [1], "latent": [1]},
    )

    import onnxruntime as ort_rt

    s = ort_rt.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    a = s.run(None, {"mel": mel.numpy()})[0]
    b = s.run(None, {"mel": mel.numpy()})[0]
    assert np.array_equal(a, b), "NON-DETERMINISTIC fcpe EXPORT"
    with torch.no_grad():
        ref = core(mel).numpy()
    rmse = float(np.sqrt(np.mean((a - ref) ** 2)))
    assert rmse < 1e-4, f"onnx-vs-torch mismatch rmse={rmse}"
    print(json.dumps({"mel_cfg": mel_cfg, "latent_shape": list(a.shape), "rmse_vs_torch": rmse, "out": out_path}))


if __name__ == "__main__":
    main()
