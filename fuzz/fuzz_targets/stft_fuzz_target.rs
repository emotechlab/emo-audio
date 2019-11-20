#![no_main]
use libfuzzer_sys::fuzz_target;

use hound::WavReader;
use emo_audio::{FrequencyComponents, stft::*};

fuzz_target!(|data: &[u8]| {
    if let Ok(r) = WavReader::new(data) {
        let norm: f32 = 2.0f32.powi(r.spec().bits_per_sample as i32);
        let audio = r.into_samples::<i32>()
            .map(|x| {
                let x = x.unwrap_or(0);
                (x as f32) / norm
            })
            .collect::<Vec<f32>>();
        
        if let Some(res) = ShortTimeFourierTransform::default().run(&audio) {
            let _ = res.mag();
        }
    };
});
