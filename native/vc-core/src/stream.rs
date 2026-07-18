use anyhow::Result;
use rand_pcg::Pcg64;

use crate::dsp::sola::Sola;
use crate::engine::{seeded_rng, Engine, StageTimes};

/// All sizes in 16 kHz samples, multiples of 320 (one contentvec frame) so
/// frame alignment stays exact through the 2x-upsample + 400x synth chain.
#[derive(Clone, Debug)]
pub struct StreamCfg {
    pub block: usize,      // emitted per hop (e.g. 1920 = 120 ms)
    pub ctx_left: usize,   // context before the block (e.g. 8320 = 520 ms)
    pub crossfade: usize,  // e.g. 1280 = 80 ms
    pub sola_search: usize, // e.g. 320 = 20 ms
    pub pitch_semitones: i32,
    pub seed: u64,
}

impl Default for StreamCfg {
    fn default() -> Self {
        Self { block: 2560, ctx_left: 4160, crossfade: 1280, sola_search: 320, pitch_semitones: 0, seed: 1337 }
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
    rng: Pcg64,
    pub last_times: StageTimes,
    /// output samples per input sample (40k/16k = 2.5 for 40k models)
    ratio_num: usize,
    ratio_den: usize,
}

impl StreamEngine {
    pub fn new(engine: Engine, cfg: StreamCfg) -> Self {
        for (name, v) in [("block", cfg.block), ("ctx_left", cfg.ctx_left), ("crossfade", cfg.crossfade), ("sola_search", cfg.sola_search)] {
            assert!(v % 320 == 0, "{name} must be a multiple of 320 (one feature frame), got {v}");
        }
        let out_sr = engine.out_sr() as usize;
        let g = gcd(out_sr, 16000);
        let window = cfg.ctx_left + cfg.block + cfg.crossfade + cfg.sola_search;
        let sola = Sola::new(cfg.crossfade * out_sr / 16000, cfg.sola_search * out_sr / 16000);
        Self {
            engine,
            rng: seeded_rng(cfg.seed),
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

    /// Push exactly cfg.block new 16 kHz samples. Returns a stitched 40 kHz
    /// block (len = block * 2.5) once warm; None on the first (warmup) hop.
    pub fn push(&mut self, block: &[f32]) -> Result<Option<Vec<f32>>> {
        assert_eq!(block.len(), self.cfg.block, "push() wants exactly cfg.block samples");
        // roll window
        self.in_buf.drain(..self.cfg.block);
        self.in_buf.extend_from_slice(block);

        let (out, times) = self.engine.convert(&self.in_buf, self.cfg.pitch_semitones, &mut self.rng)?;
        self.last_times = times;

        let scale = |n: usize| n * self.ratio_num / self.ratio_den;
        let tail_len = scale(self.cfg.block + self.cfg.crossfade + self.cfg.sola_search);
        if out.len() < tail_len {
            anyhow::bail!("converted window too short: {} < {}", out.len(), tail_len);
        }
        let tail = &out[out.len() - tail_len..];
        Ok(self.sola.process(tail))
    }

    pub fn out_sr(&self) -> u32 {
        self.engine.out_sr()
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
