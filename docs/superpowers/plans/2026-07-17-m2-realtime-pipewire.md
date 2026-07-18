# M2 Realtime Pipewire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vc-live` daemon — mic in, converted voice out through a virtual "VC Mic" source that Discord/OBS pick like a normal microphone; measured mouth-to-ear latency, optimized toward the spec gate.

**Architecture:** pipewire-rs client (direct, per spec): capture stream (default mic, negotiated 16k mono f32) → rtrb SPSC ring → worker thread running vc-core StreamEngine (realtime params) → SPSC ring → playback stream (40k mono) into a null-sink virtual source node. RT callbacks only push/pop rings. Watchdog: missed deadline → crossfaded silence + counter.

**Tech Stack:** `pipewire` crate (0.8), `rtrb` ring buffers, vc-core. Virtual node via `pactl load-module module-null-sink media.class=Audio/Source/Virtual` (pipewire-pulse), torn down on exit/panic.

## Global Constraints

- Repo `~/voice-changer`, branch `native`. Build FROM `native/` cwd (`.cargo/config.toml` rpath lives there — manifest-path builds silently drop it).
- Realtime StreamCfg start point: block 2560, ctx_left 4160, crossfade 640 (40 ms — halved from offline), sola_search 320, lookahead 0. All flags overridable.
- **Latency honesty:** spec gate <150 ms is NOT reachable at current per-hop cost (M1: hop ~147 ms at 160 ms block). Plan ships functional live path first, then optimization passes with measured numbers; user A/B vs python client decides acceptance. Divergence documented in results doc.
- Never leave a dead VC-Mic node: teardown on clean exit, SIGINT/SIGTERM, and panic hook.
- Commit per task, joke Co-Author trailer.

---

### Task 1: vc-live scaffold + pipewire probe

**Files:** Create `native/vc-live/Cargo.toml`, `native/vc-live/src/main.rs`; modify `native/Cargo.toml` (member).

- [ ] deps: `vc-core = { path = "../vc-core" }`, `pipewire = "0.8"`, `rtrb = "0.3"`, `clap` derive, `anyhow`, `ctrlc = "3"`.
- [ ] `vc-live --probe`: connect, enumerate audio nodes (name, media.class), print. Verifies pipewire-rs links + runtime present.
- [ ] Run: `cargo run --release -p vc-live -- --probe` → node list includes the Antlion mic. Commit.

### Task 2: virtual mic lifecycle

**Files:** `native/vc-live/src/virtmic.rs`

- [ ] Create via `pactl load-module module-null-sink sink_name=vc_mic sink_properties=device.description=VC-Mic media.class=Audio/Source/Virtual audio.position=[MONO]` (capture module index from stdout). Fallback error if pactl absent.
- [ ] Teardown: `pactl unload-module <idx>` on Drop + ctrlc handler + panic hook.
- [ ] Verify: `wpctl status | grep -i vc` shows source while running, gone after exit (incl. kill -INT). Commit.

### Task 3: passthrough audio path (no engine)

**Files:** `native/vc-live/src/audio.rs`, main wiring

- [ ] Capture stream: default source, format f32 mono 16000 Hz (pipewire resamples); `process` callback pushes into `rtrb::Producer<f32>` (ring 64k samples). No alloc/lock in callback.
- [ ] Playback stream: f32 mono 40000 Hz, `target.object` = vc_mic node; callback pops output ring, zero-fills on underrun (count xruns).
- [ ] `--passthrough` mode: worker copies input ring → naive 16k→40k linear upsample → output ring. Verify: `pw-record --target vc_mic out.wav` while speaking; audio intelligible; log xruns. Commit.

### Task 4: engine integration

**Files:** `native/vc-live/src/main.rs` (worker)

- [ ] Worker thread: accumulate ring input to `block` samples → `StreamEngine::push` → push result to output ring. Deadline miss (hop > block-time): push crossfaded silence, increment miss counter, log every 10th.
- [ ] Pre-roll: prime output ring with 2×block of silence to absorb jitter.
- [ ] Live smoke: `vc-live` running, `pw-record --target vc_mic` 10 s speech → converted Alex Jones audio, no dropouts at defaults. Report hop p50/p95 + xruns + misses on exit. Commit.

### Task 5: latency measurement harness

**Files:** `native/tools/measure_latency.sh`

- [ ] Method: `--tone` flag makes vc-live inject a 1 kHz burst every 2 s INTO its own input path (marker at capture time, logged timestamp); simultaneously `pw-record --target vc_mic` captures output; script correlates burst positions vs log timestamps → mouth-to-ear ms distribution. (Avoids speaker/mic acoustic loop.)
- [ ] Record baseline at defaults: expected ≈ block 160 + hop ~147 + lookback 60 + buffers. Document real number. Commit.

### Task 6: optimization pass 1

- [ ] ORT intra-op threads sweep (1/2/4/8) on the three sessions independently — synth is the hog; measure hop p50.
- [ ] Stage parallelism: run contentvec and (mel+fcpe+decode) on two threads inside Engine::convert (they share only the input window) — rejoin before synth. Measure.
- [ ] Block/ctx sweep at new hop cost: find lowest block with ≥10% headroom; re-measure mouth-to-ear.
- [ ] If < 250 ms reached: try crossfade 320 (20 ms) + listen check. Commit with numbers table.

### Task 7: live A/B + results

- [ ] User checklist: Discord voice settings shows "VC-Mic"; call test; A/B vs python client (`./run_gpu.sh` path) for quality + latency feel.
- [ ] `docs/superpowers/specs/2026-07-17-m2-results.md`: measured latency table, xrun/miss stats, optimization deltas, spec-gate status (<150 tracked), open items for M3.
- [ ] Push, memory update, user report.
