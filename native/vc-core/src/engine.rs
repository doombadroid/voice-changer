use anyhow::{anyhow, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayD};
use ort::session::Session;
use ort::value::TensorRef;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::dsp::f0decode::F0Decoder;
use crate::dsp::mel::MelFrontend;
use crate::pitch;

fn oe(e: ort::Error) -> anyhow::Error {
    anyhow!(e.to_string())
}

#[derive(Clone, Debug)]
pub struct EngineCfg {
    pub contentvec_onnx: String,
    pub fcpe_onnx: String,
    pub synth_onnx: String,
    pub frontend_npz: String,
    /// synthesizer upsample product (400 for 40k models: hop 160 @16k -> x2 feats -> 400/frame)
    pub upp: usize,
    pub out_sr: u32,
    pub f0_threshold: f32,
}

impl Default for EngineCfg {
    fn default() -> Self {
        Self {
            contentvec_onnx: "server/pretrain/content_vec_500.onnx".into(),
            fcpe_onnx: "native/golden/fcpe_fp32.onnx".into(),
            synth_onnx: "native/golden/synth_alexjones_fp32.onnx".into(),
            frontend_npz: "native/golden/fcpe_frontend.npz".into(),
            upp: 400,
            out_sr: 40000,
            f0_threshold: 0.006,
        }
    }
}

enum Noise<'a> {
    Rng(&'a mut Pcg64),
    Given { rnd: ArrayD<f32>, nsf_noise: ArrayD<f32> },
}

pub struct Engine {
    cv: Session,
    fcpe: Session,
    synth: Session,
    cv_input: String,
    mel_fe: MelFrontend,
    f0_dec: F0Decoder,
    cfg: EngineCfg,
}

/// Timings of the last convert call, ms.
#[derive(Default, Clone, Copy, Debug)]
pub struct StageTimes {
    pub contentvec: f64,
    pub pitch: f64,
    pub synth: f64,
}

impl Engine {
    pub fn new(cfg: EngineCfg) -> Result<Self> {
        let mk = |p: &str| -> Result<Session> {
            Session::builder().map_err(oe)?.commit_from_file(p).map_err(oe).with_context(|| p.to_string())
        };
        let cv = mk(&cfg.contentvec_onnx)?;
        let cv_input = cv.inputs()[0].name().to_string();
        Ok(Self {
            fcpe: mk(&cfg.fcpe_onnx)?,
            synth: mk(&cfg.synth_onnx)?,
            mel_fe: MelFrontend::from_npz(&cfg.frontend_npz)?,
            f0_dec: F0Decoder::from_npz(&cfg.frontend_npz)?,
            cv,
            cv_input,
            cfg,
        })
    }

    /// Single-pass conversion of a 16 kHz window. Returns audio at cfg.out_sr
    /// plus per-stage times.
    pub fn convert(
        &mut self,
        audio16k: &[f32],
        pitch_semitones: i32,
        rng: &mut Pcg64,
    ) -> Result<(Vec<f32>, StageTimes)> {
        self.convert_impl(audio16k, pitch_semitones, Noise::Rng(rng))
    }

    /// Test hook: run with externally supplied rnd/nsf_noise (golden comparison).
    pub fn convert_with_noise(
        &mut self,
        audio16k: &[f32],
        pitch_semitones: i32,
        rnd: ArrayD<f32>,
        nsf_noise: ArrayD<f32>,
    ) -> Result<(Vec<f32>, StageTimes)> {
        self.convert_impl(audio16k, pitch_semitones, Noise::Given { rnd, nsf_noise })
    }

    fn convert_impl(
        &mut self,
        audio16k: &[f32],
        pitch_semitones: i32,
        noise_src: Noise<'_>,
    ) -> Result<(Vec<f32>, StageTimes)> {
        let mut times = StageTimes::default();
        let n = audio16k.len();

        // --- contentvec ---
        let t0 = std::time::Instant::now();
        let audio = Array2::from_shape_vec((1, n), audio16k.to_vec())?;
        let o = self
            .cv
            .run(ort::inputs![self.cv_input.as_str() => TensorRef::from_array_view(&audio.clone().into_dyn()).map_err(oe)?])
            .map_err(oe)?;
        let (shape, data) = o["unit12"].try_extract_tensor::<f32>().map_err(oe)?;
        let t_feats = shape[1] as usize;
        let feats = Array3::from_shape_vec((1, t_feats, 768), data.to_vec())?;
        drop(o);
        times.contentvec = t0.elapsed().as_secs_f64() * 1e3;

        // feats 2x nearest upsample (Pipeline.py F.interpolate default mode)
        let tu = t_feats * 2;
        let mut feats_up = Array3::<f32>::zeros((1, tu, 768));
        for t in 0..t_feats {
            for c in 0..768 {
                let v = feats[[0, t, c]];
                feats_up[[0, 2 * t, c]] = v;
                feats_up[[0, 2 * t + 1, c]] = v;
            }
        }

        // --- pitch: mel -> fcpe -> decode -> shift -> align tail -> coarse ---
        let t0 = std::time::Instant::now();
        let mel = self.mel_fe.process(audio16k)?; // [Tm, 128]
        let tm = mel.shape()[0];
        let mel3 = Array3::from_shape_vec((1, tm, 128), mel.into_raw_vec_and_offset().0)?;
        let o = self
            .fcpe
            .run(ort::inputs!["mel" => TensorRef::from_array_view(&mel3.clone().into_dyn()).map_err(oe)?])
            .map_err(oe)?;
        let (lshape, ldata) = o["latent"].try_extract_tensor::<f32>().map_err(oe)?;
        let latent = Array2::from_shape_vec((lshape[1] as usize, lshape[2] as usize), ldata.to_vec())?;
        drop(o);
        let mut f0 = self.f0_dec.decode(latent.view(), self.cfg.f0_threshold);
        pitch::shift_semitones(&mut f0, pitch_semitones);
        // tail-align to feats length (Pipeline.py pitch[:, -feats_len:])
        let f0_al: Vec<f32> = if f0.len() >= tu {
            f0[f0.len() - tu..].to_vec()
        } else {
            let mut v = vec![0.0f32; tu - f0.len()];
            v.extend_from_slice(&f0);
            v
        };
        let coarse = pitch::to_coarse(&f0_al);
        times.pitch = t0.elapsed().as_secs_f64() * 1e3;

        // --- synth ---
        let t0 = std::time::Instant::now();
        let p_len = Array1::from_vec(vec![tu as i64]).into_dyn();
        let sid = Array1::from_vec(vec![0i64]).into_dyn();
        let pitch_a = Array2::from_shape_vec((1, tu), coarse)?.into_dyn();
        let pitchf_a = Array2::from_shape_vec((1, tu), f0_al)?.into_dyn();
        let (rnd, noise): (ArrayD<f32>, ArrayD<f32>) = match noise_src {
            Noise::Rng(rng) => (
                Array3::from_shape_fn((1, 192, tu), |_| standard_normal(rng)).into_dyn(),
                Array3::from_shape_fn((1, tu * self.cfg.upp, 1), |_| standard_normal(rng)).into_dyn(),
            ),
            Noise::Given { rnd, nsf_noise } => (rnd, nsf_noise),
        };
        let feats_dyn = feats_up.into_dyn();
        let o = self
            .synth
            .run(ort::inputs![
                "feats" => TensorRef::from_array_view(&feats_dyn).map_err(oe)?,
                "p_len" => TensorRef::from_array_view(&p_len).map_err(oe)?,
                "pitch" => TensorRef::from_array_view(&pitch_a).map_err(oe)?,
                "pitchf" => TensorRef::from_array_view(&pitchf_a).map_err(oe)?,
                "sid" => TensorRef::from_array_view(&sid).map_err(oe)?,
                "rnd" => TensorRef::from_array_view(&rnd).map_err(oe)?,
                "nsf_noise" => TensorRef::from_array_view(&noise).map_err(oe)?
            ])
            .map_err(oe)?;
        let (_, out) = o["audio"].try_extract_tensor::<f32>().map_err(oe)?;
        let out = out.to_vec();
        times.synth = t0.elapsed().as_secs_f64() * 1e3;
        Ok((out, times))
    }

    pub fn out_sr(&self) -> u32 {
        self.cfg.out_sr
    }
}

/// Box-Muller standard normal from a seeded PCG — matches nothing external,
/// but is deterministic across platforms (that's all we need; the oracle's
/// noise is externalized precisely so any gaussian source works).
fn standard_normal(rng: &mut Pcg64) -> f32 {
    loop {
        let u1: f64 = rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        if u1 > 1e-12 {
            return ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        }
    }
}

pub fn seeded_rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}
