use std::fs::File;

use anyhow::{Context, Result};
use ndarray::{ArrayD, ArrayView2};
use ndarray_npy::NpzReader;

/// fcpe latent -> f0 decoder (torchfcpe latent2cents_local_decoder + cent_to_f0).
pub struct F0Decoder {
    cent_table: Vec<f32>, // [360]
}

impl F0Decoder {
    pub fn from_npz(path: &str) -> Result<Self> {
        let mut npz = NpzReader::new(File::open(path).with_context(|| path.to_string())?)?;
        let t: ArrayD<f32> = npz.by_name("cent_table.npy").or_else(|_| npz.by_name("cent_table"))?;
        Ok(Self { cent_table: t.iter().copied().collect() })
    }

    /// latent: [T, 360] sigmoid outputs. Returns f0 Hz per frame, 0.0 = unvoiced.
    pub fn decode(&self, latent: ArrayView2<f32>, threshold: f32) -> Vec<f32> {
        let dims = self.cent_table.len();
        latent
            .rows()
            .into_iter()
            .map(|row| {
                let (mut max_i, mut max_v) = (0usize, f32::MIN);
                for (i, &v) in row.iter().enumerate() {
                    if v > max_v {
                        max_v = v;
                        max_i = i;
                    }
                }
                let (mut num, mut den) = (0.0f32, 0.0f32);
                for k in 0..9 {
                    let idx = (max_i as isize - 4 + k as isize).clamp(0, dims as isize - 1) as usize;
                    num += self.cent_table[idx] * row[idx];
                    den += row[idx];
                }
                let cents = num / den;
                if max_v <= threshold {
                    0.0
                } else {
                    10.0 * (cents / 1200.0).exp2()
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_latent_decodes_to_golden_f0() {
        let golden_path = "../golden/golden.npz";
        if !std::path::Path::new(golden_path).exists() {
            eprintln!("golden.npz missing; skipping");
            return;
        }
        let mut npz = NpzReader::new(File::open(golden_path).unwrap()).unwrap();
        let latent: ArrayD<f32> =
            npz.by_name("latent_fcpe.npy").or_else(|_| npz.by_name("latent_fcpe")).unwrap();
        let f0_g: ArrayD<f32> = npz.by_name("f0_fcpe.npy").or_else(|_| npz.by_name("f0_fcpe")).unwrap();

        let d = F0Decoder::from_npz("../golden/fcpe_frontend.npz").unwrap();
        let l2 = latent.into_dimensionality::<ndarray::Ix3>().unwrap();
        let f0 = d.decode(l2.index_axis(ndarray::Axis(0), 0), 0.006);

        let g: Vec<f32> = f0_g.iter().copied().collect();
        assert_eq!(f0.len(), g.len());
        let mut agree = 0usize;
        let mut cents_errs: Vec<f32> = Vec::new();
        for (a, b) in f0.iter().zip(&g) {
            let av = *a > 0.0;
            let bv = *b > 0.0;
            if av == bv {
                agree += 1;
            }
            if av && bv {
                cents_errs.push(1200.0 * (a / b).log2().abs());
            }
        }
        cents_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = cents_errs.get(cents_errs.len() / 2).copied().unwrap_or(f32::NAN);
        let agree_frac = agree as f32 / f0.len() as f32;
        eprintln!("voiced agreement {agree_frac:.3}, median cents err {med:.3}");
        assert!(agree_frac > 0.95, "voiced/unvoiced agreement {agree_frac}");
        assert!(med < 5.0, "median cents err {med}");
    }
}
