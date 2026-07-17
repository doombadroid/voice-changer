# vc-native — Standalone Native Voice Changer (Design)

**Date:** 2026-07-17
**Status:** Approved (brainstorm complete)
**Repo:** fork of w-okada/voice-changer → github.com/doombadroid/voice-changer, work lives under `native/`
**Working name:** `vc-native` (rename is cheap; decide before M5)

## Goal

Replace the python-server + browser-client architecture with a single native Linux
binary (AppImage) that does real-time RVC voice conversion on any GPU via Vulkan,
with the lowest achievable latency on AMD hardware as the top priority.

## Scope

- **Platform:** Linux only (v1). GPU via Vulkan — vendor-neutral, one binary for
  AMD/NVIDIA/Intel. Primary bench/tuning target: AMD Strix Halo (Radeon 8060S,
  gfx1151, RADV).
- **Engine:** RVC only (v1 + v2 model formats, 32/40/48k). All other upstream
  engines (MMVC, so-vits, DDSP, DiffusionSVC, Beatrice, EasyVC, LLVC) dropped.
- **Priorities (order):** 1) latency on AMD, 2) setup/routing UX, 3) model
  management, 4) UI overhaul.

### Non-goals (v1)

- No training — inference only.
- No Windows/macOS builds.
- No network features, no telemetry, no auto-update.
- No faiss index requirement — index support is a stretch goal (M4+); models run
  index-off by default and the quality delta is acceptable for good models.
- The existing python install is not shipped or packaged — it survives only as
  the golden-reference test oracle (it carries uncommitted local ROCm mods; do
  not clean it).

## Approach (decided)

**B — native rewrite.** Native single binary; RVC graph reimplemented on a
Vulkan-capable inference runtime. Python/torch is not distributable across GPU
vendors (no Vulkan backend, multi-GB per-vendor wheels), so the core is rewritten
and validated stage-by-stage against the python oracle.

**Language:** follows the runtime spike (M0). Rust is the default (realtime-audio
thread safety, cargo static builds, `ort`/`cpal`/pipewire-rs/egui ecosystem);
C++ acceptable iff ggml hand-port wins the spike.

## Architecture

One process, four thread groups:

```
pipewire capture (RT thread)
    → SPSC lock-free ring
        → inference worker (chunker → RVC pipeline → SOLA crossfade)
            → SPSC ring
                → pipewire playback (RT) → virtual source "VC Mic" → apps
UI thread (egui) ⇄ worker: control channels (model swap, pitch, params)
```

Rules:

- RT threads: no allocation, no locks, no GPU calls — ring push/pop only.
- Inference worker owns the GPU context and model exclusively.
- Model swap: build new pipeline in background thread, atomic pointer swap, no
  audio gap.
- Settings: single TOML/JSON file, atomic write (tmp+rename).

## RVC pipeline (fixed graph, we own it)

48k mic → resample 16k mono → **ContentVec/HuBERT** (768-d features, 50 fps) →
optional index blend (index_rate) → **f0 estimation** (RMVPE = quality mode,
fcpe = fast mode; spike decides the default) → 2× feature upsample → **VITS
synthesizer** (enc + flow + NSF-HiFiGAN decoder, f0-conditioned, 32/40/48k per
model) → **SOLA** align + crossfade → resample to device rate → out.

Chunked streaming: block size tunable 40–192 ms with context padding both
sides, lookahead trimmed; target operating point 64 ms.

**Latency target: <150 ms mouth-to-ear verified on gfx1151 (M2 gate); stretch
<100 ms.** Block floor is bounded by HuBERT's 20 ms frame stride. Gate:
inference time < block time at the 64 ms operating point on gfx1151.

## GPU runtime — spike decides (M0, timeboxed ~1 week)

Bench all four candidates on gfx1151 + CPU with the same exported graphs (or
closest proxy):

| Candidate | Key risk to verify |
|---|---|
| ncnn (Vulkan) | pnnx conversion of attention + BiGRU (RMVPE landmine op) |
| ONNX Runtime WebGPU EP (Dawn→Vulkan) | EP maturity; ops silently falling back to CPU |
| burn (wgpu→Vulkan, pure Rust) | op coverage (GRU/conv-transpose), kernel perf |
| ggml Vulkan (hand-port) | effort — weeks not days; bench attention+conv proxies only |

Decision matrix, in priority order: **(1) full-pipeline latency on RADV**,
(2) op coverage with no silent CPU fallback, (3) binary size, (4) effort.
If RMVPE's BiGRU blocks an otherwise-winning runtime, default pitch estimator
becomes fcpe (transformer, no GRU) and RMVPE ships later or CPU-side.

## Latency engineering (AMD-first)

- **Overlap:** capture block N+1 while inferring block N (double-buffer);
  added latency = block + inference, not 2×block.
- **GPU tactics:** fp16 storage+arithmetic (`VK_KHR_shader_float16`, native on
  RDNA3.5); RDNA3 WMMA/cooperative-matrix where the runtime supports it;
  weights uploaded once; zero per-chunk allocations; pre-recorded command
  buffers; pipelined staging — no mid-pipeline GPU↔CPU sync stalls.
- **Context trim:** tune HuBERT left-context and SOLA search window down until
  golden tests show degradation — padding is pure latency.
- **CPU side:** pinned worker thread, alloc-free loop, SIMD resampler.
- **Audio side:** pipewire small quantum (64–256 samples), RT priority via
  rtkit, single clock domain full-duplex (kills resample drift).

## Audio & routing (pipewire native)

- Direct pipewire client — no ALSA/Pulse shim.
- App auto-creates virtual source **"VC Mic"** + optional loopback monitor.
  Discord/OBS simply see a microphone.
- Hotplug: node watch, auto-relink, device picker in UI.
- Live latency ledger in UI: capture buf + block + inference + output buf =
  mouth-to-ear ms — measured (startup loopback ping, optional), not estimated.
- On exit/crash: restore routing — never leave apps holding a dead node.

## Model management

- Import via dialog or drag-drop: `.pth` (+ optional `.index`) → one-time
  native conversion → `~/.local/share/vc-native/models/<name>/` (runtime
  weights + `preset.json`: pitch, index_rate, block size, f0 mode).
- **No bundled python.** The graph is fixed, so the converter parses the .pth
  tensor dict directly (zip+pickle tensor extraction, native code). Detect
  v1/v2 and sample rate from key shapes; reject unknown archs with a clear,
  actionable error. Never half-import.
- faiss `.index`: parse IndexIVFFlat blob into our own IVF search (stretch,
  M4+). If the format fights back, v1 ships index-off.
- Model cards in UI: name, sr/version, f0 default, A/B audition button (canned
  phrase through model).

## UI (egui, GPU-rendered, lives in the binary)

Single window, dark, dense-but-clean:

- **Top strip:** in/out VU meters, live latency ledger, xrun counter, GPU badge
  (device name, fp16/coopmat active).
- **Left:** model cards, import button, A/B audition.
- **Center:** pitch knob (semitones), index_rate, block size, f0 mode
  (fast/quality), monitor toggle, panic-mute hotkey.
- **Bottom:** device pickers, autostart toggle, log drawer.
- Per-model presets. Tray minimize; close-to-tray optional.
- Also `--cli in.wav out.wav` offline mode (doubles as test harness) and
  `--bench` (per-stage ms table).

## Error handling

- GPU init fail → CPU fallback + visible banner. Never silent-slow.
- Silent-CPU-fallback detection: startup micro-bench per stage; any stage 10×
  slower than spike baseline → warn.
- Model conversion: typed errors with fix hints; atomic import.
- Inference stall watchdog: missed chunk deadline → emit crossfaded silence,
  count it, banner after N misses. Audio never locks up.
- Xruns counted and visible. Panic handler saves state and restores routing.

## Testing

- **Golden harness:** python install is the oracle. Fixed input wavs →
  stage-by-stage RMSE tolerances (HuBERT feats, f0 curve, final audio). Runs in
  CI on CPU.
- Unit: resampler, SOLA, rings, .pth parser (fuzzed — untrusted input), faiss
  parser (fuzzed).
- Latency bench CLI regression-tracked per commit.
- Manual per-milestone: real Discord call checklist.

## Milestones

| M | Deliverable | Gate |
|---|---|---|
| M0 | Runtime bake-off on gfx1151; decision matrix; language locked | latency criterion #1 |
| M1 | Offline CLI wav→wav | matches python oracle within tolerance |
| M2 | Realtime pipewire path + virtual mic | <150 ms verified on gfx1151 |
| M3 | egui UI complete | all controls functional |
| M4 | Model import UX + presets (+index if parser cooperated) | clean import of arbitrary community .pth |
| M5 | AppImage, CI builds, README, latency tuning pass #2 | runs on clean non-dev Linux box |
