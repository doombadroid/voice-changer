/// SOLA (synchronous overlap-add) chunk stitching, mirroring w-okada
/// VoiceChangerV2: search the first `search` samples of the new tail for the
/// offset best correlated with the previous chunk's crossfade buffer, then
/// equal-power crossfade.
pub struct Sola {
    crossfade: usize,
    search: usize,
    /// previous tail * prev_strength (cos^2 ramp-out), length = crossfade
    buf: Option<Vec<f32>>,
    cur_strength: Vec<f32>,  // sin^2 ramp-in
    prev_strength: Vec<f32>, // cos^2 ramp-out
}

impl Sola {
    pub fn new(crossfade: usize, search: usize) -> Self {
        let cur = (0..crossfade)
            .map(|i| {
                let t = (i as f32 + 0.5) / crossfade as f32 * std::f32::consts::FRAC_PI_2;
                t.sin().powi(2)
            })
            .collect();
        let prev = (0..crossfade)
            .map(|i| {
                let t = (i as f32 + 0.5) / crossfade as f32 * std::f32::consts::FRAC_PI_2;
                t.cos().powi(2)
            })
            .collect();
        Self { crossfade, search, buf: None, cur_strength: cur, prev_strength: prev }
    }

    /// tail: newest converted audio laid out as [block | crossfade | search].
    /// Returns the stitched block (len = tail.len() - crossfade - search), or
    /// None on the warmup call (buffer primed, nothing emitted).
    pub fn process(&mut self, tail: &[f32]) -> Option<Vec<f32>> {
        let block = tail.len() - self.crossfade - self.search;
        match &self.buf {
            None => {
                self.buf = Some(
                    tail[tail.len() - self.crossfade..]
                        .iter()
                        .zip(&self.prev_strength)
                        .map(|(a, s)| a * s)
                        .collect(),
                );
                None
            }
            Some(prev) => {
                // normalized cross-correlation over [0, search)
                let head = &tail[..self.crossfade + self.search];
                let mut best_off = 0usize;
                let mut best = f32::MIN;
                let mut total_energy = 0.0f32;
                for off in 0..=self.search {
                    // w-okada: correlate against the strength-weighted buffer,
                    // normalize by windowed energy of the candidate region
                    let mut nom = 0.0f32;
                    let mut den = 0.0f32;
                    for i in 0..self.crossfade {
                        nom += head[off + i] * prev[i];
                        den += head[off + i] * head[off + i];
                    }
                    total_energy += den;
                    let score = nom / (den.sqrt() + 1e-8);
                    if score > best {
                        best = score;
                        best_off = off;
                    }
                }
                // In silence the correlation is noise and the offset random-walks,
                // time-warping the stream. Anchor to the nominal center instead.
                if total_energy / (self.search as f32 + 1.0) < 1e-4 {
                    best_off = self.search / 2;
                }

                let mut out = tail[best_off..best_off + block].to_vec();
                let prev_buf = self.buf.take().unwrap();
                for i in 0..self.crossfade.min(out.len()) {
                    out[i] = out[i] * self.cur_strength[i] + prev_buf[i];
                }
                // prime next buffer from the region following the emitted block
                let start = best_off + block;
                self.buf = Some(
                    tail[start..start + self.crossfade]
                        .iter()
                        .zip(&self.prev_strength)
                        .map(|(a, s)| a * s)
                        .collect(),
                );
                Some(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_chunks_stitch_without_clicks() {
        let sr = 40000.0f32;
        let hz = 220.0f32;
        let block = 4000usize;
        let cf = 1600usize;
        let search = 400usize;
        let mut sola = Sola::new(cf, search);
        let mut out = Vec::new();
        // simulate chunk pipeline with small phase jitter per chunk
        for c in 0..6 {
            let jitter = if c % 2 == 0 { 0 } else { 37 }; // samples of drift
            let start = c * block;
            let tail: Vec<f32> = (0..block + cf + search)
                .map(|i| (2.0 * std::f32::consts::PI * hz * (start + i + jitter) as f32 / sr).sin())
                .collect();
            if let Some(mut b) = sola.process(&tail) {
                out.append(&mut b);
            }
        }
        // click detector: max jump between consecutive samples
        let max_jump = out.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        // reference: max jump of a clean sine at this frequency
        let clean_jump = 2.0 * std::f32::consts::PI * hz / sr; // ~0.0346
        assert!(
            max_jump < 3.0 * clean_jump,
            "click detected: max jump {max_jump} vs clean {clean_jump}"
        );
    }
}
