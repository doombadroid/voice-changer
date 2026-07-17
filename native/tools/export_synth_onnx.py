"""Export RVC v2 synthesizer to ONNX with externalized rnd input (deterministic).

Mirrors server/voice_changer/RVC/onnxExporter/export2onnx.py exactly (opset 17,
fp32, no remove_weight_norm, dynamic T axis) except torch.randn_like is replaced
by a graph input `rnd` (scale 0.66666 stays inside the graph, matching this
tree's runtime semantics).

Run: server/.venv/bin/python native/tools/export_synth_onnx.py <model.pth> <out.onnx>
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "server")
import torch.nn.functional as F  # noqa: E402

from voice_changer.RVC.inferencer.rvc_models.infer_pack import models as rvc_models  # noqa: E402
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (  # noqa: E402
    SynthesizerTrnMs768NSFsid_ONNX,
)


def det_sinegen_forward(self, f0, upp):
    """SineGen.forward with both random sites made deterministic:
    - rand_ini -> zeros: identical for harmonic_num=0 (dim=1: rand_ini[:,0]=0
      overwrites the whole tensor anyway)
    - randn_like -> external tensor stashed on the module by the top wrapper
      (traces as a graph edge from the `nsf_noise` input)
    Otherwise a verbatim copy of models.py SineGen.forward.
    """
    with torch.no_grad():
        f0 = f0[:, None].transpose(1, 2)
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        f0_buf[:, :, 0] = f0[:, :, 0]
        for idx in np.arange(self.harmonic_num):
            f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
        rad_values = (f0_buf / self.sampling_rate) % 1
        rand_ini = torch.zeros(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        tmp_over_one = torch.cumsum(rad_values, 1)
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1), scale_factor=upp, mode="linear", align_corners=True
        ).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * self._ext_noise
        sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


class SynthesizerDeterministic(SynthesizerTrnMs768NSFsid_ONNX):
    def forward(self, feats, p_len, pitch, pitchf, sid, rnd, nsf_noise):
        self.dec.m_source.l_sin_gen._ext_noise = nsf_noise
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(feats, pitch, p_len)
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :], pitchf, g=g)
        return torch.clip(o[0, 0], -1.0, 1.0)


def main() -> None:
    pth_path, out_path = sys.argv[1], sys.argv[2]
    rvc_models.SineGen.forward = det_sinegen_forward
    cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    sr = cpt["config"][-1]
    net = SynthesizerDeterministic(*cpt["config"], is_half=False)
    net.eval()
    net.load_state_dict(cpt["weight"], strict=False)
    assert net.dec.m_source.l_sin_gen.dim == 1, "harmonic_num != 0: rand_ini zeroing no longer exact"

    T = 64
    inter_channels = cpt["config"][2]
    upp = int(np.prod(cpt["config"][12]))  # upsample_rates product (e.g. 400 for 40k)
    feats = torch.randn(1, T, 768)
    p_len = torch.LongTensor([T])
    pitch = torch.randint(1, 255, (1, T), dtype=torch.int64)
    pitchf = torch.rand(1, T) * 300 + 100
    sid = torch.LongTensor([0])
    rnd = torch.randn(1, inter_channels, T)
    nsf_noise = torch.randn(1, T * upp, 1)

    torch.onnx.export(
        net,
        (feats, p_len, pitch, pitchf, sid, rnd, nsf_noise),
        out_path,
        dynamic_axes={"feats": [1], "pitch": [1], "pitchf": [1], "rnd": [2], "nsf_noise": [1]},
        do_constant_folding=False,
        opset_version=17,
        dynamo=False,
        input_names=["feats", "p_len", "pitch", "pitchf", "sid", "rnd", "nsf_noise"],
        output_names=["audio"],
    )

    import onnxruntime as ort_rt

    s = ort_rt.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    ins = {
        "feats": feats.numpy(),
        "p_len": p_len.numpy(),
        "pitch": pitch.numpy(),
        "pitchf": pitchf.numpy(),
        "sid": sid.numpy(),
        "rnd": rnd.numpy(),
        "nsf_noise": nsf_noise.numpy(),
    }
    a = s.run(None, ins)[0]
    b = s.run(None, ins)[0]
    assert np.array_equal(a, b), "NON-DETERMINISTIC EXPORT - randn still in graph"
    print(json.dumps({"sr": int(sr), "inter_channels": int(inter_channels), "upp": upp, "out": out_path, "audio_shape": list(a.shape)}))


if __name__ == "__main__":
    main()
