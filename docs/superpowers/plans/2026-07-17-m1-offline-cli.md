# M1 Offline CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vc-cli in.wav out.wav` — full RVC conversion (any-sr wav → Alex Jones 40k) through the REAL chunked streaming engine (the M2 realtime core, driven offline), matching the python-oracle-derived golden chain within tolerance.

**Architecture:** `native/vc-core` lib (dsp: resample/mel/f0-decode/SOLA; engine: 3 ort sessions + chunked pipeline with context windows) + `native/vc-cli` bin (wav io, args, bench). Single-pass mode = correctness anchor vs extended goldens; chunked mode = production path, gated on spectral distance vs single-pass + zero boundary clicks.

**Tech Stack:** Rust, `ort` 2.0.0-rc.12 (CPU EP default), `rubato` (sinc resampler), `realfft` (mel STFT), `hound` (wav), `clap`. Python venv only for golden/asset dumps.

## Global Constraints

- Repo `~/voice-changer`, branch `native`, push origin only. All commands from repo root.
- Runtime per M0 decision: ort CPU EP default; fcpe pitch; fp32; models = `server/pretrain/content_vec_500.onnx`, `native/golden/fcpe_fp32.onnx`, `native/golden/synth_alexjones_fp32.onnx` (dynamic-axis exports — M0 showed they're fastest on ORT). Paths via config struct, hardcoded defaults OK for M1 (import UX = M4).
- Determinism: all noise (rnd, nsf_noise) from seeded RNG (default seed 1337, `--seed`); same cmd twice → identical output file (byte-compare gate).
- Oracle stays untouched. Goldens regenerated only via `native/tools/` scripts.
- Gates: mel RMSE < 1e-4; f0 median cents err < 5 (voiced); single-pass audio vs golden chain RMSE < 1e-3; chunked vs single-pass log-mel L1 < 0.1 + zero clicks (max abs sample-to-sample jump at boundaries ≤ 3× max jump inside chunks); byte-identical rerun.
- Bench discipline: report xRT (audio-seconds / wall-seconds) and per-stage p50 in `--bench` output.
- Commit per task, conventional commits, joke Co-Author trailer, no Anthropic attribution.

---

### Task 1: vc-core scaffold + wav io + resampler

**Files:**
- Create: `native/vc-core/Cargo.toml`, `native/vc-core/src/lib.rs`, `native/vc-core/src/dsp/mod.rs`, `native/vc-core/src/dsp/resample.rs`, `native/vc-cli/Cargo.toml`, `native/vc-cli/src/main.rs`
- Modify: `native/Cargo.toml` (members += vc-core, vc-cli)

**Interfaces:**
- Produces: `vc_core::dsp::resample::Resampler::new(from_hz, to_hz) -> Self`, `.process(&[f32]) -> Vec<f32>` (streaming-safe, keeps filter state); `vc_cli` reads wav (hound, any sr/channels → mono f32) writes 16-bit wav.

- [ ] **Step 1: crates + deps** — vc-core deps: `ort` (webgpu+ndarray features), `rubato = "0.16"`, `realfft = "3"`, `ndarray = "0.17"`, `ndarray-npy = "0.10"`, `anyhow`, `rand = "0.9"`, `rand_pcg = "0.9"` (seedable, stable). vc-cli deps: vc-core, `hound = "3"`, `clap` derive, `anyhow`.
- [ ] **Step 2: resampler** — wrap rubato `SincFixedIn::<f32>` (256-tap, cutoff 0.95, oversampling 256, WindowFunction::BlackmanHarris2) chunk-feeding API; unit test: 1 kHz sine 48k→16k→ RMS preserved ±1%, spectral peak still 1 kHz (goertzel check); round-trip 16k→40k→16k RMSE < 1e-2.
- [ ] **Step 3: cli skeleton** — `vc-cli in.wav out.wav` just resamples to 16k and back out for now (plumbing proof). Run on `demo_alexjones.wav`, listen-sane.
- [ ] **Step 4: commit** — `feat(native): vc-core scaffold, streaming resampler, cli plumbing`

### Task 2: fcpe mel frontend (exact-match via dumped assets)

**Files:**
- Create: `native/tools/dump_fcpe_frontend.py`, `native/vc-core/src/dsp/mel.rs`
- Test: gate vs `mel_fcpe` in golden.npz

**Interfaces:**
- Consumes: torchfcpe's `Wav2MelModule` internals (filterbank matrix, window, log/clamp params) — dumped, not reimplemented.
- Produces: `MelFrontend::from_npz(path) -> Self`, `.process(&[f32] /*16k*/) -> Array2<f32> /*[T,128]*/` (streaming: hop 160, win 1024, center-pad semantics matching torch — read `torchfcpe` source for center/reflect mode and REPLICATE it; record found values in commit message).

- [ ] **Step 1: dump assets** — script prints/saves npz: mel filterbank [128,513], window [1024], log formula params (clamp min, base), center/pad mode (read from installed torchfcpe source — `inspect.getsource`), keys documented. Save to `native/golden/fcpe_frontend.npz`.
- [ ] **Step 2: rust mel** — realfft 1024 + dumped window/filterbank + exact log/clamp. Gate: run on golden `audio_16k`, RMSE vs `mel_fcpe` < 1e-4. If center-padding makes frame counts differ (101 vs computed), fix padding until shapes match exactly — no truncating to fit.
- [ ] **Step 3: commit**

### Task 3: fcpe decoder (latent → f0)

**Files:**
- Create: `native/vc-core/src/dsp/f0decode.rs`; extend `dump_fcpe_frontend.py` (cent_table [360], decoder constants)

**Interfaces:**
- Produces: `F0Decoder::from_npz(...)`, `.decode(latent: &Array3<f32>, threshold: f32 /*0.006*/) -> Vec<f32>` (Hz, 0.0 = unvoiced), porting `latent2cents_local_decoder` + `cent_to_f0` semantics exactly (read source: local window size, weighted average, mask rule; record in commit).

- [ ] **Step 1: dump cent_table + read decoder source**, port to rust.
- [ ] **Step 2: gate** — decode golden `latent_fcpe` → compare `f0_fcpe`: median cents err < 5 on voiced, voiced/unvoiced agreement > 95% of frames.
- [ ] **Step 3: commit**

### Task 4: full single-pass chain + extended goldens

**Files:**
- Create: `native/vc-core/src/engine.rs`, `native/vc-core/src/pitch.rs`
- Modify: `native/tools/make_golden.py` (chain v2: full-fidelity), `native/vc-cli/src/main.rs`

**Interfaces:**
- Consumes: oracle conventions from `server/voice_changer/RVC/pipeline/Pipeline.py` — READ IT and mirror: feats 2× time-upsample (F.interpolate semantics), pitch alignment at 100fps, coarse mel-scale 1–255 quantization (already in make_golden), pitch-shift math (`f0 * 2^(semitones/12)` before coarse), protect/index paths SKIPPED (index off in M1).
- Produces: `Engine::new(cfg) -> Result<Engine>` (loads 3 ort sessions); `Engine::convert(&mut self, audio16k: &[f32], pitch_semitones: i32, seed: u64) -> Result<Vec<f32>>` (single-pass, returns 40k audio); golden.npz gains `feats_up [1,98,768]`, `audio_out_v2` (chain with 2× upsample — the actual RVC-fidelity output; M0 chain skipped upsample, noted in decision doc).

- [ ] **Step 1: regen goldens v2** — make_golden adds feats-2×-upsample before synth (mirror Pipeline.py interpolate mode), pitch/pitchf at 98 frames, nsf_noise sized 98×400; output ≈0.98 s. Listen-check wav again.
- [ ] **Step 2: engine single-pass** — contentvec → upsample2× (linear interp identical to F.interpolate `nearest`? READ Pipeline.py; replicate found mode bit-exact) → mel→fcpe→decode→shift→coarse → synth(seeded rnd+noise) → out. Gate: RMSE < 1e-3 vs `audio_out_v2` when run on golden audio with same seed.
- [ ] **Step 3: cli wire** — `vc-cli in.wav out.wav --pitch 0 --seed 1337 --single-pass`; run on demo wav full 7.85 s, output 40k wav; determinism gate (run twice, `cmp` byte-identical); listen.
- [ ] **Step 4: commit**

### Task 5: chunked streaming engine + SOLA

**Files:**
- Create: `native/vc-core/src/dsp/sola.rs`, `native/vc-core/src/stream.rs`
- Test: unit (synthetic sine SOLA), integration (chunked vs single-pass gates)

**Interfaces:**
- Consumes: w-okada SOLA reference — READ `server/voice_changer/utils/` (or VoiceChangerV2.py; locate sola search/crossfade impl by grep `sola`) and mirror window sizes/search range semantics.
- Produces: `StreamEngine::new(cfg: StreamCfg {block_ms, ctx_left_ms, ctx_right_ms, sola_search_ms, xfade_ms, pitch, seed})`; `.push(block: &[f32] /*16k mono, block-sized*/) -> Result<Option<Vec<f32>>>` (40k out per hop once warm); internally: rolling 16k buffer, per-hop window = ctx_left+block+ctx_right, full pipeline per window, output tail extraction, SOLA align + equal-power crossfade vs previous tail.

- [ ] **Step 1: SOLA unit** — port algorithm; test: two overlapping windows of same sine with phase drift → post-SOLA join has no discontinuity (max sample jump < 2× intra-chunk max).
- [ ] **Step 2: stream engine** — defaults block 128 ms, ctx_left 512 ms, ctx_right 64 ms (tune later); wire chunk loop; `vc-cli` default path becomes chunked (single-pass behind `--single-pass`).
- [ ] **Step 3: gates** — chunked vs single-pass on demo wav: log-mel L1 < 0.1, click detector zero boundary clicks, determinism byte-gate. If L1 fails: dump per-hop RMSE to find drifting stage before touching params (systematic-debugging).
- [ ] **Step 4: commit**

### Task 6: bench + report + push

**Files:**
- Modify: `native/vc-cli/src/main.rs` (`--bench`), decision-doc appendix or new `docs/superpowers/specs/2026-07-17-m1-results.md`

- [ ] **Step 1: `--bench`** — per-stage p50 (resample, cv, mel+fcpe+decode, synth, sola) + xRT + per-hop compute ms vs block ms (headroom %) printed as table; run at block 64/128/192 ms.
- [ ] **Step 2: results doc** — measured table, gate results, chunk-parameter defaults chosen, open issues for M2.
- [ ] **Step 3: push + report** — `git push origin native`; update memory file M1 status; user summary with listen-artifact paths.
