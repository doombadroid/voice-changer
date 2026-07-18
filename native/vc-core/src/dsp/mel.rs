use std::fs::File;
use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayD};
use ndarray_npy::NpzReader;
use realfft::{RealFftPlanner, RealToComplex};

/// fcpe mel frontend, exact-match to torchfcpe MelModule (see dump_fcpe_frontend.py
/// for the replicated semantics). Offline-style: processes a whole 16k window.
pub struct MelFrontend {
    mel_basis: Array2<f32>, // [128, 513]
    window: Vec<f32>,       // [1024]
    n_fft: usize,           // 1024
    win: usize,             // 1024
    hop: usize,             // 160
    clip_val: f32,          // 1e-5
    fft: Arc<dyn RealToComplex<f32>>,
}

impl MelFrontend {
    pub fn from_npz(path: &str) -> Result<Self> {
        let mut npz = NpzReader::new(File::open(path).with_context(|| path.to_string())?)?;
        let mel_basis: ArrayD<f32> = npz.by_name("mel_basis.npy").or_else(|_| npz.by_name("mel_basis"))?;
        let window: ArrayD<f32> = npz.by_name("hann_window.npy").or_else(|_| npz.by_name("hann_window"))?;
        let scalars: ArrayD<f32> = npz.by_name("scalars.npy").or_else(|_| npz.by_name("scalars"))?;
        let s: Vec<f32> = scalars.iter().copied().collect();
        let (n_fft, win, hop, clip_val) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3]);
        let mel_basis = mel_basis.into_dimensionality::<ndarray::Ix2>()?;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        Ok(Self {
            mel_basis,
            window: window.iter().copied().collect(),
            n_fft,
            win,
            hop,
            clip_val,
            fft,
        })
    }

    /// audio: 16 kHz mono. Returns [T, 128] log-mel with T = len/hop + 1.
    pub fn process(&self, audio: &[f32]) -> Result<Array2<f32>> {
        let pad_left = (self.win - self.hop) / 2; // 432
        let pad_right_min = (self.win - self.hop + 1) / 2; // 432
        let pad_right = pad_right_min.max(self.win.saturating_sub(audio.len() + pad_left));

        // reflect padding (constant-zero fallback only for ultra-short signals)
        let n = audio.len();
        let mut padded = Vec::with_capacity(n + pad_left + pad_right);
        if pad_right < n {
            for i in 0..pad_left {
                padded.push(audio[pad_left - i]); // reflect, excluding edge sample
            }
            padded.extend_from_slice(audio);
            for i in 0..pad_right {
                padded.push(audio[n - 2 - i]);
            }
        } else {
            padded.resize(pad_left, 0.0);
            padded.extend_from_slice(audio);
            padded.resize(n + pad_left + pad_right, 0.0);
        }

        // center=False stft: frames start at 0, step hop, length win
        let frames = (padded.len() - self.win) / self.hop + 1;
        let n_bins = self.n_fft / 2 + 1;
        let mut spec = Array2::<f32>::zeros((frames, n_bins)); // magnitude
        let mut buf = vec![0.0f32; self.n_fft];
        let mut out = self.fft.make_output_vec();
        for f in 0..frames {
            let start = f * self.hop;
            for i in 0..self.win {
                buf[i] = padded[start + i] * self.window[i];
            }
            // win == n_fft here; if win < n_fft torch zero-pads centered - not our case
            self.fft.process(&mut buf, &mut out).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (b, c) in out.iter().enumerate() {
                spec[[f, b]] = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
            }
        }

        // mel projection + log-clamp
        let mut mel = Array2::<f32>::zeros((frames, self.mel_basis.shape()[0]));
        for f in 0..frames {
            for m in 0..self.mel_basis.shape()[0] {
                let mut acc = 0.0f32;
                for b in 0..n_bins {
                    acc += self.mel_basis[[m, b]] * spec[[f, b]];
                }
                mel[[f, m]] = acc.max(self.clip_val).ln();
            }
        }

        // force frame count to len/hop + 1 (repeat-last or trim), like Wav2MelModule
        let target = n / self.hop + 1;
        let cur = mel.shape()[0];
        if target > cur {
            let last = mel.row(cur - 1).to_owned();
            let mut grown = Array2::<f32>::zeros((target, mel.shape()[1]));
            grown.slice_mut(ndarray::s![..cur, ..]).assign(&mel);
            for t in cur..target {
                grown.row_mut(t).assign(&last);
            }
            mel = grown;
        } else if target < cur {
            mel = mel.slice(ndarray::s![..target, ..]).to_owned();
        }
        Ok(mel)
    }

    pub fn hop(&self) -> usize {
        self.hop
    }
}

#[allow(dead_code)]
fn unused(_: Array1<f32>) {}
