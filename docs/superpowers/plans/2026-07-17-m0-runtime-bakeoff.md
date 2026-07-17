# M0 Runtime Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide the GPU inference runtime (ORT-WebGPU vs burn) for vc-native by benchmarking the real RVC pipeline on gfx1151/RADV, with golden-reference correctness gates against the working python install.

**Architecture:** Export deterministic ONNX graphs of all four pipeline nets (ContentVec, RMVPE, fcpe, RVC synthesizer) from the python oracle at `~/voice-changer/server`, dump golden input/output tensors, then run identical graphs through two Rust spike crates (`ort` with WebGPU EP; `burn-onnx` with Vulkan backend), measuring p50/p95 latency and RMSE vs golden. Produce a decision doc.

**Tech Stack:** Rust (rustup stable), `ort` 2.0.0-rc.12 (feature `webgpu`), burn 0.21 + burn-onnx 0.21 (feature `vulkan`), python venv at `server/.venv` (torch 2.9.1+rocm, onnx 1.22, onnxruntime 1.27 CPU) for exports/goldens.

## Global Constraints

- Repo: `~/voice-changer`, branch `native`. Push to `origin` (doombadroid fork) only — https remote.
- All new native code under `native/`. Python helper scripts under `native/tools/` (they run with `server/.venv/bin/python`).
- Do NOT modify the python server beyond what's already dirty (3 files) — oracle must stay working. New scripts only.
- Do NOT commit model weights or golden tensors. `native/golden/` and `*.onnx` are gitignored; scripts must regenerate them.
- Latency measurement discipline: 20 warmup runs, 100 timed runs, report p50/p95, single process, performance governor not required (report if throttled).
- Correctness gates (fp32): ContentVec feats RMSE < 1e-4; f0 voiced-frame error < 5 cents median; synthesizer audio RMSE < 1e-3 (identical `rnd` input). fp16: log-mel-spectrogram L1 < 0.05 vs fp32 output.
- Chunk shape for all benches: 1.0 s @ 16 kHz (16000 samples) primary; 0.5 s secondary. Static shapes (pad) so ORT graph capture works.
- The venv may lack a `pip` entrypoint — install packages with `uv pip install --python /home/timb/voice-changer/server/.venv/bin/python <pkg>`.
- Commit after every task. Commit messages: conventional commits, terse subject, joke defunct-AI Co-Authored-By trailer (rotation rule), no Anthropic attribution.

---

### Task 1: Rust toolchain + `native/` workspace scaffold

**Files:**
- Create: `native/Cargo.toml` (workspace), `native/spike-ort/Cargo.toml`, `native/spike-ort/src/main.rs`, `native/.gitignore`
- Modify: none

**Interfaces:**
- Produces: cargo workspace `native/` that later tasks add crates to; `spike-ort` binary with subcommands added in Tasks 6–7.

- [ ] **Step 1: Verify/install rust toolchain**

Run: `command -v cargo && cargo --version || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && . "$HOME/.cargo/env" && cargo --version`
Expected: `cargo 1.8x.x` (any 2026 stable)

- [ ] **Step 2: Create workspace**

`native/Cargo.toml`:
```toml
[workspace]
members = ["spike-ort"]
resolver = "2"

[workspace.package]
edition = "2021"
license = "MIT"
```

`native/.gitignore`:
```
target/
golden/
*.onnx
*.npz
```

`native/spike-ort/Cargo.toml`:
```toml
[package]
name = "spike-ort"
version = "0.1.0"
edition.workspace = true

[dependencies]
ort = { version = "2.0.0-rc.12", features = ["webgpu", "fetch-models"] }
ndarray = "0.16"
ndarray-npy = "0.9"
anyhow = "1"
clap = { version = "4", features = ["derive"] }
```

`native/spike-ort/src/main.rs`:
```rust
use anyhow::Result;

fn main() -> Result<()> {
    // Probe: prints ORT version + available EPs. Subcommands land in later tasks.
    println!("ort version: {}", ort::MINOR_VERSION);
    Ok(())
}
```

- [ ] **Step 3: Build (downloads prebuilt ORT with WebGPU EP)**

Run: `cd /home/timb/voice-changer/native && cargo build 2>&1 | tail -5`
Expected: `Finished` line. If `ort` API names differ from this plan (rc-series churn), fix against docs.rs/ort — record actual names in the task commit message.

- [ ] **Step 4: Commit**

```bash
cd /home/timb/voice-changer
git add native/
git commit -m "feat(native): scaffold rust workspace + ort spike crate"
```

---

### Task 2: Deterministic synthesizer ONNX export (`rnd` externalized)

The stock w-okada exporter keeps `torch.randn_like` in-graph → non-deterministic → golden tests impossible, and ORT-WebGPU lacks RandomNormalLike anyway. Patch: export with `rnd` as a graph INPUT (upstream RVC-Project convention).

**Files:**
- Create: `native/tools/export_synth_onnx.py`
- Test: determinism check inside the script (run graph twice, assert identical)

**Interfaces:**
- Consumes: `server/model_dir/0/alexjones.pth` (RVC v2, 40kHz, f0), classes from `server/voice_changer/RVC/inferencer/rvc_models/infer_pack/models.py`
- Produces: `native/golden/synth_alexjones_fp32.onnx` — inputs `feats[1,T,768] f32`, `p_len[1] i64`, `pitch[1,T] i64`, `pitchf[1,T] f32`, `sid[1] i64`, `rnd[1,192,T] f32`; output `audio[1,S] f32`

- [ ] **Step 1: Write export script**

`native/tools/export_synth_onnx.py`:
```python
"""Export RVC v2 synthesizer to ONNX with externalized rnd input (deterministic).
Run: server/.venv/bin/python native/tools/export_synth_onnx.py <model.pth> <out.onnx>
"""
import sys, json, torch
sys.path.insert(0, "server")
sys.path.insert(0, "server/voice_changer/RVC/inferencer/rvc_models")
from infer_pack.models import SynthesizerTrnMs768NSFsid

pth_path, out_path = sys.argv[1], sys.argv[2]
cpt = torch.load(pth_path, map_location="cpu", weights_only=False)
sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk from weights
net = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
net.load_state_dict(cpt["weight"], strict=False)
net.eval().remove_weight_norm()

class Wrapped(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, feats, p_len, pitch, pitchf, sid, rnd):
        # mirror SynthesizerTrnMs768NSFsid.infer but with rnd injected
        g = self.m.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.m.enc_p(feats, pitch, p_len)
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = self.m.flow(z_p, x_mask, g=g, reverse=True)
        o = self.m.dec((z * x_mask)[:, :, :], pitchf, g=g)
        return o.squeeze(1)

w = Wrapped(net)
T = 100
feats = torch.randn(1, T, 768)
p_len = torch.tensor([T], dtype=torch.int64)
pitch = torch.randint(1, 255, (1, T), dtype=torch.int64)
pitchf = torch.rand(1, T) * 300 + 100
sid = torch.tensor([0], dtype=torch.int64)
rnd = torch.randn(1, 192, T)
torch.onnx.export(
    w, (feats, p_len, pitch, pitchf, sid, rnd), out_path, opset_version=17,
    input_names=["feats", "p_len", "pitch", "pitchf", "sid", "rnd"],
    output_names=["audio"],
    dynamic_axes={"feats": {1: "t"}, "pitch": {1: "t"}, "pitchf": {1: "t"}, "rnd": {2: "t"}},
)
# determinism check with onnxruntime CPU
import onnxruntime as ort_rt, numpy as np
s = ort_rt.InferenceSession(out_path, providers=["CPUExecutionProvider"])
ins = {"feats": feats.numpy(), "p_len": p_len.numpy(), "pitch": pitch.numpy(),
       "pitchf": pitchf.numpy(), "sid": sid.numpy(), "rnd": rnd.numpy()}
a, b = s.run(None, ins)[0], s.run(None, ins)[0]
assert np.array_equal(a, b), "NON-DETERMINISTIC EXPORT — randn still in graph"
print(json.dumps({"sr": sr, "out": out_path, "audio_shape": list(a.shape)}))
```

Note: if `enc_p`/`flow`/`dec` attribute names or `infer` signature differ in this tree's `models.py`, read that file first and mirror its actual `infer()` body — the ONLY change is `rnd` replacing `randn_like`.

- [ ] **Step 2: Run export**

Run: `cd /home/timb/voice-changer && mkdir -p native/golden && server/.venv/bin/python native/tools/export_synth_onnx.py server/model_dir/0/alexjones.pth native/golden/synth_alexjones_fp32.onnx`
Expected: JSON line with `"sr": 40000`, no assertion error.

- [ ] **Step 3: Commit script**

```bash
git add native/tools/export_synth_onnx.py
git commit -m "feat(native): deterministic rnd-external synth onnx export"
```

---

### Task 3: fcpe ONNX export (mel-in, decode-out)

Keep STFT/mel OUTSIDE the graph (ORT-WebGPU has no STFT kernel; burn's kokoro divergence was STFT-rooted). Export mel→cent-logits; decode (local-argmax → Hz) happens in Rust later; for golden purposes decode in python.

**Files:**
- Create: `native/tools/export_fcpe_onnx.py`

**Interfaces:**
- Consumes: `torchfcpe` pip package (installs `fcpe_c_v001.pt`)
- Produces: `native/golden/fcpe_fp32.onnx` — input `mel[1,T,128] f32`; output `logits[1,T,360] f32`; plus recorded mel params (n_fft/hop/sr/mel-fmin/fmax) printed as JSON for the Rust mel impl.

- [ ] **Step 1: Install torchfcpe into venv**

Run: `uv pip install --python /home/timb/voice-changer/server/.venv/bin/python torchfcpe`
Expected: installed OK (torch already satisfied).

- [ ] **Step 2: Write + run export script**

`native/tools/export_fcpe_onnx.py`:
```python
"""Export fcpe encoder (mel -> cent logits) to ONNX. Prints mel config JSON.
Run: server/.venv/bin/python native/tools/export_fcpe_onnx.py native/golden/fcpe_fp32.onnx
"""
import sys, json, inspect, torch
from torchfcpe import spawn_bundled_infer_model
m = spawn_bundled_infer_model(device="cpu")
# Locate the mel->logits core module and the mel extractor config.
core = m.model if hasattr(m, "model") else m  # inspect actual attr names at runtime
print("model attrs:", [a for a in dir(m) if not a.startswith("_")][:40], file=sys.stderr)
mel_cfg = {}
for name in ("mel_extractor", "wav2mel", "mel"):
    if hasattr(m, name):
        me = getattr(m, name)
        for k in ("n_fft", "hop_length", "hop_size", "sampling_rate", "sr", "num_mels", "n_mels", "fmin", "fmax", "mel_fmin", "mel_fmax"):
            if hasattr(me, k):
                mel_cfg[k] = getattr(me, k)
T = 100
n_mels = int(mel_cfg.get("num_mels", mel_cfg.get("n_mels", 128)))
mel = torch.randn(1, T, n_mels)
out_path = sys.argv[1]

class MelToLogits(torch.nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, mel):
        return self.core(mel)  # adjust after inspecting core.forward signature

torch.onnx.export(MelToLogits(core), (mel,), out_path, opset_version=17,
                  input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {1: "t"}, "logits": {1: "t"}})
print(json.dumps({"mel_cfg": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mel_cfg.items()}, "out": out_path}))
```

This script REQUIRES runtime inspection — torchfcpe's internal attr names must be read from the installed package first (`server/.venv/bin/python -c "import torchfcpe, inspect; print(inspect.getsource(torchfcpe.spawn_bundled_infer_model))"`), then the script adjusted so `core` is the actual mel→logits module and the decode path (`latent2cents_decoder` or similar) is identified. If the bundled checkpoint uses linear attention that won't trace, retry with `conv_only` config; if that fails too, record fcpe as EXPORT-BLOCKED in the decision doc and continue (RMVPE covers pitch).

Run: `cd /home/timb/voice-changer && server/.venv/bin/python native/tools/export_fcpe_onnx.py native/golden/fcpe_fp32.onnx`
Expected: JSON with mel_cfg containing hop/sr values; file created.

- [ ] **Step 3: Commit**

```bash
git add native/tools/export_fcpe_onnx.py
git commit -m "feat(native): fcpe mel->logits onnx export"
```

---

### Task 4: Fetch real RMVPE ONNX (replaces broken stub)

**Files:**
- Create: `native/tools/fetch_pretrained.sh`

**Interfaces:**
- Produces: `native/golden/rmvpe_20231006.onnx` (inputs `waveform[1,T] f32` @16k, `threshold[1] f32`=0.3; output `pitchf[1,T'] f32` Hz) and repaired `server/pretrain/rmvpe.onnx`.

- [ ] **Step 1: Write + run fetch script**

`native/tools/fetch_pretrained.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
URL="https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"
OUT="native/golden/rmvpe_20231006.onnx"
[ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 100000000 ] || curl -L --fail -o "$OUT" "$URL"
stat -c '%n %s' "$OUT"
sha256sum "$OUT"
# repair the 110-byte AccessDenied stub the python install has
if [ "$(stat -c%s server/pretrain/rmvpe.onnx 2>/dev/null || echo 0)" -lt 1000 ]; then
  cp "$OUT" server/pretrain/rmvpe.onnx && echo "repaired server/pretrain/rmvpe.onnx"
fi
```

Run: `chmod +x native/tools/fetch_pretrained.sh && native/tools/fetch_pretrained.sh`
Expected: ~362MB file, sha256 printed, stub repaired.

- [ ] **Step 2: Commit**

```bash
git add native/tools/fetch_pretrained.sh
git commit -m "feat(native): fetch rmvpe onnx + repair stub"
```

---

### Task 5: Golden reference tensors

**Files:**
- Create: `native/tools/make_golden.py`

**Interfaces:**
- Consumes: `server/pretrain/content_vec_500.onnx` (input `audio[1,T]` f32 16k; outputs `units9/unit12/unit12s`), Task 2/3/4 artifacts, `server/model_dir/0/alexjones.pth` metadata (40k, index_rate 0.3 ignored — index OFF for goldens)
- Produces: `native/golden/golden.npz` with keys: `audio_16k[1,16000]`, `feats[1,T,768]` (unit12), `f0_rmvpe[frames]`, `pitch_coarse[1,T]`, `pitchf[1,T]`, `rnd[1,192,T]` (seeded), `audio_out[1,S]`; and `native/golden/golden_out.wav`. (fcpe goldens added in Task 7 era only if Task 3 succeeded — separate `fcpe_golden.npz`, same script pattern.)

- [ ] **Step 1: Write golden script**

`native/tools/make_golden.py`:
```python
"""Generate golden tensors for spike correctness gates. All CPU, all deterministic.
Run: server/.venv/bin/python native/tools/make_golden.py
"""
import json, numpy as np, onnxruntime as ort_rt, torch, soundfile as sf
import librosa

SEED = 1337
np.random.seed(SEED); torch.manual_seed(SEED)

# 1s test clip @16k from the repo's known-good demo wav
wav, sr = librosa.load("demo_alexjones.wav", sr=16000, mono=True)
audio = wav[: 16000].astype(np.float32)[None, :]  # [1,16000]

# ContentVec (existing onnx, CPU EP = deterministic)
cv = ort_rt.InferenceSession("server/pretrain/content_vec_500.onnx", providers=["CPUExecutionProvider"])
cv_outs = cv.run(None, {cv.get_inputs()[0].name: audio})
names = [o.name for o in cv.get_outputs()]
feats = cv_outs[names.index("unit12")]  # [1,T',768] v2 path

# RMVPE onnx (CPU)
rm = ort_rt.InferenceSession("native/golden/rmvpe_20231006.onnx", providers=["CPUExecutionProvider"])
f0 = rm.run(None, {"waveform": audio, "threshold": np.array([0.3], dtype=np.float32)})[0]

# pitch -> coarse (RVC convention: 1-255 mel-scale buckets) aligned to feats T
T = feats.shape[1]
f0r = np.interp(np.linspace(0, len(f0.ravel()) - 1, T), np.arange(len(f0.ravel())), f0.ravel())
f0_mel = 1127 * np.log(1 + f0r / 700)
f0_mel_min, f0_mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
coarse = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255)
coarse = np.rint(np.where(f0r > 0, coarse, 1)).astype(np.int64)[None, :]
pitchf = f0r.astype(np.float32)[None, :]

# Synthesizer (deterministic rnd)
rnd = np.random.RandomState(SEED).randn(1, 192, T).astype(np.float32)
syn = ort_rt.InferenceSession("native/golden/synth_alexjones_fp32.onnx", providers=["CPUExecutionProvider"])
audio_out = syn.run(None, {"feats": feats, "p_len": np.array([T], dtype=np.int64),
                           "pitch": coarse, "pitchf": pitchf,
                           "sid": np.array([0], dtype=np.int64), "rnd": rnd})[0]

np.savez("native/golden/golden.npz", audio_16k=audio, feats=feats, f0_rmvpe=f0,
         pitch_coarse=coarse, pitchf=pitchf, rnd=rnd, audio_out=audio_out)
sf.write("native/golden/golden_out.wav", audio_out.ravel(), 40000)
print(json.dumps({"feats": list(feats.shape), "f0": list(f0.shape), "audio_out": list(audio_out.shape)}))
```

- [ ] **Step 2: Run + listen**

Run: `server/.venv/bin/python native/tools/make_golden.py && aplay native/golden/golden_out.wav` (or `pw-play`)
Expected: JSON shapes; wav sounds like Alex Jones saying the demo line. If garbage → coarse-pitch mapping or feats output pick is wrong; check against `server/voice_changer/RVC/pipeline/Pipeline.py` conventions before proceeding.

- [ ] **Step 3: Commit**

```bash
git add native/tools/make_golden.py
git commit -m "feat(native): golden reference tensor generator"
```

---

### Task 6: spike-ort — CPU EP baseline (correctness + timing)

**Files:**
- Modify: `native/spike-ort/src/main.rs` (replace probe with real CLI)

**Interfaces:**
- Consumes: `native/golden/*.onnx`, `native/golden/golden.npz`
- Produces: `spike-ort --ep cpu` → prints per-net RMSE vs golden + p50/p95 ms lines (`name: p50 X ms p95 Y ms` + `name RMSE: Z`). Task 7 reuses the same binary with `--ep webgpu`.

- [ ] **Step 1: Implement runner**

`native/spike-ort/src/main.rs` — full replacement:
```rust
use anyhow::{Context, Result};
use clap::Parser;
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;
use std::{fs::File, time::Instant};

#[derive(Parser)]
struct Args {
    /// cpu | webgpu
    #[arg(long, default_value = "cpu")]
    ep: String,
    #[arg(long, default_value = "native/golden")]
    golden: String,
    #[arg(long, default_value_t = 100)]
    iters: usize,
}

fn npz_f32(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<f32>> {
    Ok(npz.by_name::<ndarray::OwnedRepr<f32>, IxDyn>(key)?)
}
fn npz_i64(npz: &mut NpzReader<File>, key: &str) -> Result<ArrayD<i64>> {
    Ok(npz.by_name::<ndarray::OwnedRepr<i64>, IxDyn>(key)?)
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    (a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / n as f32).sqrt()
}

fn bench<F: FnMut() -> Result<Vec<f32>>>(name: &str, iters: usize, mut f: F) -> Result<(Vec<f32>, f64, f64)> {
    for _ in 0..20 { f()?; } // warmup
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    let mut out = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        out = f()?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[iters / 2];
    let p95 = times[(iters as f64 * 0.95) as usize];
    println!("{name}: p50 {p50:.2} ms  p95 {p95:.2} ms");
    Ok((out, p50, p95))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut npz = NpzReader::new(File::open(format!("{}/golden.npz", args.golden))?)?;
    let audio = npz_f32(&mut npz, "audio_16k")?;
    let feats_g = npz_f32(&mut npz, "feats")?;
    let f0_g = npz_f32(&mut npz, "f0_rmvpe")?;
    let pitch = npz_i64(&mut npz, "pitch_coarse")?;
    let pitchf = npz_f32(&mut npz, "pitchf")?;
    let rnd = npz_f32(&mut npz, "rnd")?;
    let audio_out_g = npz_f32(&mut npz, "audio_out")?;

    let build = |path: &str| -> Result<ort::session::Session> {
        let b = ort::session::Session::builder()?;
        // EP wiring: for webgpu use ort::ep::WebGPU with graph capture; exact builder
        // API is rc-series — consult docs.rs/ort 2.0.0-rc.12 and adjust here.
        let b = match args.ep.as_str() {
            "webgpu" => b.with_execution_providers([ort::ep::webgpu::WebGPU::default().build()])?,
            _ => b,
        };
        Ok(b.commit_from_file(path)?)
    };

    // --- ContentVec ---
    let cv = build(&format!("{}/../../server/pretrain/content_vec_500.onnx", args.golden))
        .or_else(|_| build("server/pretrain/content_vec_500.onnx"))?;
    let (out, _, _) = bench("contentvec", args.iters, || {
        let o = cv.run(ort::inputs!["audio" => audio.view()]?)?;
        Ok(o["unit12"].try_extract_tensor::<f32>()?.1.to_vec())
    })?;
    println!("contentvec RMSE: {:.2e}", rmse(&out, feats_g.as_slice().unwrap()));

    // --- RMVPE ---
    let rm = build(&format!("{}/rmvpe_20231006.onnx", args.golden))?;
    let thr = ndarray::arr1(&[0.3f32]).into_dyn();
    let (out, _, _) = bench("rmvpe", args.iters, || {
        let o = rm.run(ort::inputs!["waveform" => audio.view(), "threshold" => thr.view()]?)?;
        Ok(o["pitchf"].try_extract_tensor::<f32>()?.1.to_vec())
    })?;
    println!("rmvpe RMSE(Hz): {:.3}", rmse(&out, f0_g.as_slice().unwrap()));

    // --- Synthesizer ---
    let t = feats_g.shape()[1] as i64;
    let p_len = ndarray::arr1(&[t]).into_dyn();
    let sid = ndarray::arr1(&[0i64]).into_dyn();
    let syn = build(&format!("{}/synth_alexjones_fp32.onnx", args.golden))?;
    let (out, _, _) = bench("synth", args.iters, || {
        let o = syn.run(ort::inputs![
            "feats" => feats_g.view(), "p_len" => p_len.view(), "pitch" => pitch.view(),
            "pitchf" => pitchf.view(), "sid" => sid.view(), "rnd" => rnd.view()]?)?;
        Ok(o["audio"].try_extract_tensor::<f32>()?.1.to_vec())
    })?;
    println!("synth RMSE: {:.2e}", rmse(&out, audio_out_g.as_slice().unwrap()));

    // fcpe optional — same pattern if native/golden/fcpe_fp32.onnx exists
    Ok(())
}
```

API-churn warning is part of the plan: `ort` rc APIs move; the implementer MUST reconcile against the crate docs for rc-12 and keep the structure (bench closure, RMSE gate, EP switch).

- [ ] **Step 2: Run CPU baseline**

Run: `cd /home/timb/voice-changer/native && cargo run --release -p spike-ort -- --ep cpu`
Expected: three RMSE lines all within gates (contentvec < 1e-4, synth < 1e-3, rmvpe f0 near-zero — same graph+runtime as golden), p50/p95 table. Record numbers.

- [ ] **Step 3: Commit**

```bash
git add native/spike-ort/
git commit -m "feat(native): ort cpu baseline bench + correctness gates"
```

---

### Task 7: spike-ort — WebGPU EP on RADV

**Files:**
- Modify: `native/spike-ort/src/main.rs` (EP options: graph capture, fp16 variant)
- Create: `native/tools/fp16_convert.py`

**Interfaces:**
- Produces: same bench table on `--ep webgpu`, plus node-placement report (which ops fell back to CPU), plus fp16 numbers for contentvec+synth.

- [ ] **Step 1: Run WebGPU fp32**

Run: `cd /home/timb/voice-changer/native && RUST_LOG=ort=debug cargo run --release -p spike-ort -- --ep webgpu 2>ep.log`
Expected: RMSE gates hold (fp32 cross-EP: contentvec < 1e-4, synth < 1e-3; rmvpe f0 median cents < 5). `grep -ci "fallback\|CPUExecutionProvider" ep.log` — record which nodes are CPU (RMVPE's GRU expected; anything else = investigate).

- [ ] **Step 2: Enable graph capture + rebench**

Add to EP options: `enableGraphCapture: "1"` (static shapes — inputs already fixed 1s). If capture rejects dynamic-axis models, re-export synth with fixed T (add `--static-t 50` flag to Task 2 script) and rebench. Record delta.

- [ ] **Step 3: fp16 variants**

`native/tools/fp16_convert.py`:
```python
"""fp16-convert an onnx model. Run: server/.venv/bin/python native/tools/fp16_convert.py in.onnx out.onnx"""
import sys
from onnxconverter_common import float16
import onnx
m = onnx.load(sys.argv[1])
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx.save(m16, sys.argv[2])
print("ok")
```

Run: `uv pip install --python /home/timb/voice-changer/server/.venv/bin/python onnxconverter-common` then convert contentvec + synth, rebench with `--ep webgpu`. Gate: log-mel L1 of synth output vs fp32 golden < 0.05 (add a tiny mel check in python or accept RMSE < 5e-2 + listen test via writing out.wav).

- [ ] **Step 4: Commit**

```bash
git add native/spike-ort/ native/tools/fp16_convert.py
git commit -m "feat(native): webgpu ep bench - graph capture + fp16"
```

---

### Task 8: spike-burn — burn-onnx codegen on Vulkan

**Files:**
- Create: `native/spike-burn/Cargo.toml`, `native/spike-burn/build.rs`, `native/spike-burn/src/main.rs`
- Modify: `native/Cargo.toml` (add member)

**Interfaces:**
- Consumes: same ONNX files + golden.npz
- Produces: same bench table on burn Vulkan backend + CPU (ndarray) backend.

- [ ] **Step 1: Crate setup**

`native/spike-burn/Cargo.toml`:
```toml
[package]
name = "spike-burn"
version = "0.1.0"
edition.workspace = true

[dependencies]
burn = { version = "0.21", features = ["vulkan", "ndarray"] }
ndarray = "0.16"
ndarray-npy = "0.9"
anyhow = "1"

[build-dependencies]
burn-onnx = "0.21"
```

`native/spike-burn/build.rs`:
```rust
fn main() {
    // Codegen order: smallest first — if contentvec (378MB) chokes codegen,
    // we still get synth+rmvpe numbers. Comment out failures, record them.
    burn_onnx::ModelGen::new()
        .input("../golden/synth_alexjones_fp32.onnx")
        .input("../golden/rmvpe_20231006.onnx")
        .input("../../server/pretrain/content_vec_500.onnx")
        .out_dir("model/")
        .run_from_script();
}
```

`native/spike-burn/src/main.rs`: same structure as spike-ort main — load golden.npz, run each generated model on `burn::backend::Vulkan` then `burn::backend::NdArray`, bench closure (20 warmup/100 timed p50/p95), RMSE vs golden. Generated model APIs: `include!(concat!(env!("OUT_DIR"), "/model/synth_alexjones_fp32.rs"))` etc., each exposing `Model::<B>::default()` + `forward(...)` — reconcile exact generated signatures after first codegen (`cargo build -p spike-burn` writes them into OUT_DIR; read the generated .rs to wire inputs in graph input order).

- [ ] **Step 2: Build (codegen) — timebox 2h**

Run: `cd /home/timb/voice-changer/native && cargo build --release -p spike-burn 2>&1 | tail -20`
Expected: long compile (kokoro precedent: ~7 min codegen). Failures per-model are DATA not blockers: comment the failing `.input(...)` out, record error verbatim for decision doc, continue with survivors.

- [ ] **Step 3: Run both backends**

Run: `cargo run --release -p spike-burn` (runs Vulkan then NdArray in one process, prints both tables)
Expected: RMSE gates as Task 7. Record wgpu adapter line (should say RADV / gfx1151, coopmat if logged).

- [ ] **Step 4: Commit**

```bash
git add native/Cargo.toml native/spike-burn/
git commit -m "feat(native): burn-onnx vulkan spike bench"
```

---

### Task 9: Decision doc + spec update

**Files:**
- Create: `docs/superpowers/specs/2026-07-17-m0-decision.md`
- Modify: `docs/superpowers/specs/2026-07-17-vc-native-standalone-design.md` (GPU runtime section: replace 4-way spike table with the decision + link)

**Interfaces:**
- Produces: locked runtime + language for M1 planning.

- [ ] **Step 1: Write decision doc**

Structure (fill with measured numbers, no adjectives without data):
```markdown
# M0 Decision: GPU Runtime

## Measured (gfx1151, RADV, Mesa 26.1.4, 1s chunk)
| net | ort-cpu p50 | ort-webgpu p50 | ort-webgpu fp16 p50 | burn-vulkan p50 | burn-cpu p50 | gates |
|-----|-----|-----|-----|-----|-----|-----|
| contentvec | | | | | | pass/fail + RMSE |
| rmvpe      | | | | | | (note CPU-partitioned nodes) |
| fcpe       | | | | | | (or EXPORT-BLOCKED) |
| synth      | | | | | | |
| **chain** (cv+pitch+synth sum) | | | | | | vs 64ms block budget |

## CPU-fallback audit (ort-webgpu)
[node list from ep.log]

## Paper-eliminated candidates
- ncnn: Deconvolution1D/GRU/ConvolutionDepthWise1D have no Vulkan impl (verified src/layer/vulkan listing 2026-07) — NSF-HiFiGAN upsampler + HuBERT pos-conv partition to CPU. Custom-layer cost exceeds ORT/burn adoption cost. Rust bindings stale (2023).
- ggml: no GRU op; conv-transpose-1d Vulkan present but COL2IM_1D path missing; hand-port = weeks (audio.cpp/TTS.cpp precedent: both vendor patched ggml). Kept as M6+ endgame option if chosen runtime underperforms.

## Decision
Runtime: [ort-webgpu | burn | hybrid]. Language: Rust.
Rationale: [latency numbers vs 64ms budget, fallback surface, binary size, API stability]
Pitch default: [fcpe | rmvpe] based on measured cost + export success.

## Consequences for M1
[chunking sizes achievable, fp16 default?, graph-capture constraints -> static shapes]
```

Decision rule (pre-committed, so the numbers decide, not vibes): score = chain-p50 on gfx1151 (lower wins) unless the faster one fails a correctness gate or has >2 CPU-fallback node classes in core nets (contentvec/synth); ties (<20% apart) break toward burn (no C++ dep, pure-Rust single binary, owns kernels) — else ort wins on maturity.

- [ ] **Step 2: Update spec + commit + push**

```bash
git add docs/superpowers/
git commit -m "docs: m0 runtime decision"
git push origin native
```

- [ ] **Step 3: Report to user**

Summarize table + decision + what M1 plan will cover. Update memory (loonybin-style entry for vc-native project state).
