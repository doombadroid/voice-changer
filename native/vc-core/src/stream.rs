use anyhow::Result;

use crate::dsp::sola::Sola;
use crate::engine::{Engine, StageTimes};

/// All sizes in 16 kHz samples, multiples of 320 (one contentvec frame) so
/// frame alignment stays exact through the 2x-upsample + 400x synth chain.
#[derive(Clone, Debug)]
pub struct StreamCfg {
    pub block: usize,      // emitted per hop (e.g. 1920 = 120 ms)
    pub ctx_left: usize,   // context before the block (e.g. 8320 = 520 ms)
    pub crossfade: usize,  // e.g. 1280 = 80 ms
    pub sola_search: usize, // e.g. 320 = 20 ms
    /// extra right context beyond crossfade+search (offline quality lever;
    /// adds its full duration to realtime latency)
    pub lookahead: usize,
    pub pitch_semitones: i32,
    pub seed: u64,
}

impl Default for StreamCfg {
    fn default() -> Self {
        Self { block: 2560, ctx_left: 4160, crossfade: 1280, sola_search: 320, lookahead: 0, pitch_semitones: 0, seed: 1337 }
    }
}

/// Chunked realtime core: push 16 kHz blocks, receive 40 kHz stitched output.
/// Per hop the engine converts [ctx_left | block | crossfade | search] and the
/// emitted block lags the newest input by crossfade + search (SOLA lookback).
pub struct StreamEngine {
    engine: Engine,
    cfg: StreamCfg,
    in_buf: Vec<f32>,
    sola: Sola,
    /// absolute count of 16k samples pushed so far
    stream_pos: u64,
    pub last_times: StageTimes,
    /// output samples per input sample (40k/16k = 2.5 for 40k models)
    ratio_num: usize,
    ratio_den: usize,
}

impl StreamEngine {
    pub fn new(engine: Engine, cfg: StreamCfg) -> Self {
        for (name, v) in [("block", cfg.block), ("ctx_left", cfg.ctx_left), ("crossfade", cfg.crossfade), ("sola_search", cfg.sola_search), ("lookahead", cfg.lookahead)] {
            assert!(v % 320 == 0, "{name} must be a multiple of 320 (one feature frame), got {v}");
        }
        let out_sr = engine.out_sr() as usize;
        let g = gcd(out_sr, 16000);
        let window = cfg.ctx_left + cfg.block + cfg.crossfade + cfg.sola_search + cfg.lookahead;
        let sola = Sola::new(cfg.crossfade * out_sr / 16000, cfg.sola_search * out_sr / 16000);
        Self {
            engine,
            stream_pos: 0,
            in_buf: vec![0.0; window],
            sola,
            cfg,
            last_times: StageTimes::default(),
            ratio_num: out_sr / g,
            ratio_den: 16000 / g,
        }
    }

    pub fn cfg(&self) -> &StreamCfg {
        &self.cfg
    }

    /// Live pitch update (applies from the next hop).
    pub fn set_pitch(&mut self, semitones: i32) {
        self.cfg.pitch_semitones = semitones;
    }

    /// Push exactly cfg.block new 16 kHz samples. Returns a stitched 40 kHz
    /// block (len = block * 2.5) once warm; None on the first (warmup) hop.
    pub fn push(&mut self, block: &[f32]) -> Result<Option<Vec<f32>>> {
        assert_eq!(block.len(), self.cfg.block, "push() wants exactly cfg.block samples");
        // roll window
        self.in_buf.drain(..self.cfg.block);
        self.in_buf.extend_from_slice(block);
        self.stream_pos += self.cfg.block as u64;

        // window start in absolute 16k samples (in_buf is window-sized; the
        // virtual pre-stream zeros before the first pushes keep this exact)
        let window = self.in_buf.len() as u64;
        let win_start_16k = self.stream_pos.saturating_sub(window);
        debug_assert_eq!(win_start_16k % 320, 0);
        let feat_frame_offset = win_start_16k / 320 * 2;
        let out_sample_offset = feat_frame_offset * (self.engine_upp() as u64);
        let (out, times) = self.engine.convert_positioned(
            &self.in_buf,
            self.cfg.pitch_semitones,
            self.cfg.seed,
            feat_frame_offset,
            out_sample_offset,
        )?;
        self.last_times = times;

        let scale = |n: usize| n * self.ratio_num / self.ratio_den;
        let skip_end = scale(self.cfg.lookahead);
        let tail_len = scale(self.cfg.block + self.cfg.crossfade + self.cfg.sola_search);
        if out.len() < tail_len + skip_end {
            anyhow::bail!("converted window too short: {} < {}", out.len(), tail_len + skip_end);
        }
        let end = out.len() - skip_end;
        let tail = &out[end - tail_len..end];
        Ok(self.sola.process(tail))
    }

    pub fn out_sr(&self) -> u32 {
        self.engine.out_sr()
    }

    fn engine_upp(&self) -> usize {
        // out samples per feature frame (400 for 40k models)
        self.ratio_num * 160 / self.ratio_den
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
