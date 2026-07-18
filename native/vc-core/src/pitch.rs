/// RVC pitch conventions (mirrors oracle pipeline / RVC-Project).
///
/// coarse: mel-scale quantization of f0 Hz into 1..=255 (0 Hz -> 1).
pub fn shift_semitones(f0: &mut [f32], semitones: i32) {
    if semitones == 0 {
        return;
    }
    let k = (semitones as f32 / 12.0).exp2();
    for v in f0.iter_mut() {
        if *v > 0.0 {
            *v *= k;
        }
    }
}

pub fn to_coarse(f0: &[f32]) -> Vec<i64> {
    let mel_min = 1127.0f32 * (1.0f32 + 50.0 / 700.0).ln();
    let mel_max = 1127.0f32 * (1.0f32 + 1100.0 / 700.0).ln();
    f0.iter()
        .map(|&hz| {
            if hz <= 0.0 {
                return 1i64;
            }
            let mel = 1127.0 * (1.0 + hz / 700.0).ln();
            let c = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
            c.round().clamp(1.0, 255.0) as i64
        })
        .collect()
}
