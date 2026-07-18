use anyhow::{bail, Context, Result};
use clap::Parser;
use vc_core::dsp::resample::Resampler;

#[derive(Parser)]
#[command(about = "vc-native offline converter (M1)")]
struct Args {
    input: String,
    output: String,
}

fn read_wav_mono(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("open {path}"))?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / max)).collect::<Result<_, _>>()?
        }
    };
    if ch == 0 {
        bail!("zero channels");
    }
    let mono: Vec<f32> = samples.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();
    Ok((mono, spec.sample_rate))
}

fn write_wav(path: &str, samples: &[f32], sr: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (audio, sr) = read_wav_mono(&args.input)?;
    eprintln!("in: {} samples @ {sr} Hz", audio.len());

    // M1-T1 plumbing proof: resample to 16k and write back out.
    let mut down = Resampler::new(sr as usize, 16000)?;
    let mut a16 = down.process(&audio)?;
    a16.extend(down.flush()?);
    write_wav(&args.output, &a16, 16000)?;
    eprintln!("out: {} samples @ 16000 Hz -> {}", a16.len(), args.output);
    Ok(())
}
