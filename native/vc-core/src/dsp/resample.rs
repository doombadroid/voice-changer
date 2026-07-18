use anyhow::Result;
use rubato::{Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Streaming mono resampler. Accumulates input, emits whenever a full rubato
/// chunk is available; `flush` drains the tail (offline use) by zero-padding.
pub struct Resampler {
    inner: SincFixedIn<f32>,
    chunk: usize,
    pending: Vec<f32>,
}

impl Resampler {
    pub fn new(from_hz: usize, to_hz: usize) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let chunk = 1024;
        let inner = SincFixedIn::<f32>::new(to_hz as f64 / from_hz as f64, 2.0, params, chunk, 1)?;
        Ok(Self { inner, chunk, pending: Vec::new() })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.pending.extend_from_slice(input);
        let mut out = Vec::new();
        while self.pending.len() >= self.chunk {
            let take: Vec<f32> = self.pending.drain(..self.chunk).collect();
            let mut res = self.inner.process(&[take], None)?;
            out.append(&mut res.remove(0));
        }
        Ok(out)
    }

    /// Drain remaining samples (pads with zeros). Call once at end of stream.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let mut tail = std::mem::take(&mut self.pending);
        tail.resize(self.chunk, 0.0);
        let mut res = self.inner.process(&[tail], None)?;
        Ok(res.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// Goertzel power at freq.
    fn goertzel(x: &[f32], hz: f32, sr: f32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * hz / sr;
        let coeff = 2.0 * w.cos();
        let (mut s0, mut s1, mut s2) = (0.0f32, 0.0f32, 0.0f32);
        for &v in x {
            s0 = v + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (x.len() as f32).powi(2)
    }

    #[test]
    fn downsample_preserves_tone() {
        let mut r = Resampler::new(48000, 16000).unwrap();
        let x = sine(1000.0, 48000.0, 48000);
        let mut y = r.process(&x).unwrap();
        y.extend(r.flush().unwrap());
        // skip filter transient at both ends
        let core = &y[2000..y.len() - 2000];
        let rin = rms(&x);
        let rout = rms(core);
        assert!((rout / rin - 1.0).abs() < 0.01, "rms ratio {}", rout / rin);
        let p1k = goertzel(core, 1000.0, 16000.0);
        let p3k = goertzel(core, 3000.0, 16000.0);
        assert!(p1k > 100.0 * p3k, "tone not dominant: {p1k} vs {p3k}");
    }

    #[test]
    fn round_trip_16k_40k() {
        let mut up = Resampler::new(16000, 40000).unwrap();
        let mut down = Resampler::new(40000, 16000).unwrap();
        let x = sine(440.0, 16000.0, 32000);
        let mut mid = up.process(&x).unwrap();
        mid.extend(up.flush().unwrap());
        let mut y = down.process(&mid).unwrap();
        y.extend(down.flush().unwrap());
        // align: find lag by peak cross-correlation over first 2000 samples
        let n = 8000;
        let (mut best_lag, mut best) = (0usize, f32::MIN);
        for lag in 0..2000 {
            let c: f32 = (0..n).map(|i| x[i] * y[i + lag]).sum();
            if c > best {
                best = c;
                best_lag = lag;
            }
        }
        let err: f32 = (0..n).map(|i| (x[i] - y[i + best_lag]).powi(2)).sum::<f32>() / n as f32;
        assert!(err.sqrt() < 1e-2, "round trip rmse {}", err.sqrt());
    }
}
