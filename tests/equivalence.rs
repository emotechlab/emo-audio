use float_cmp::approx_eq;
use lazy_static::lazy_static;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::{read_dir, File};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use emo_audio::prelude::*;
use ndarray::prelude::*;

fn check_data_folder() -> PathBuf {
    lazy_static! {
        static ref DIR_MUTEX: Mutex<()> = Mutex::new(());
    }
    let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    env::set_current_dir(&test_dir).unwrap();
    let data_dir = test_dir.join("data");

    let _m = DIR_MUTEX.lock().expect("Directory mutex poisoned");
    if !data_dir.exists() {
        Command::new("python")
            .arg("init_data_dir.py")
            .arg(format!("-f={}", data_dir.display()))
            .arg("-s 10")
            .output()
            .expect("Failed to create test data");
    }
    data_dir
}

fn stft_from_params(params: ArrayView1<f32>) -> ShortTimeFourierTransform {
    StftBuilder::new()
        .set_fft_num(params[0] as usize)
        .set_window_len(params[1] as usize)
        .set_hop_len(params[2] as usize)
        .set_centred(true)
        .build()
}

#[test]
fn stft_equivalence() {
    let data_dir = check_data_folder();
    for entry in read_dir(&data_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_dir() {
            continue;
        }

        let mut npz = NpzReader::new(File::open(entry.path()).unwrap()).unwrap();
        let samples: Array1<f64> = npz.by_name("audio.npy").unwrap();
        let result: Array2<f32> = npz.by_name("stft.npy").unwrap();
        let params: Array1<f32> = npz.by_name("params.npy").unwrap();

        if !result.is_empty() {
            let stft = stft_from_params(params.view());
            // Parseval's theorem says sum of squares of time domain == frequency domain
            // use this to choose an epsilon
            let max_possible_bin_value = samples.iter().fold(0.0, |acc, x| acc + x.powi(2));
            let eps = (max_possible_bin_value * 2e-4) as f32;
            let stft_res = stft.run(samples.as_slice().unwrap()).unwrap().mag();
            assert_eq!(stft_res.dim(), result.dim());
            for (a, e) in stft_res.iter().zip(result.iter()) {
                assert!(
                    approx_eq!(f32, *a, *e, epsilon = eps),
                    "{} ~= {} (+/- {})",
                    a,
                    e,
                    eps
                );
            }
        }
    }
}

#[test]
fn spectrogram_equivalence() {
    let data_dir = check_data_folder();
    for entry in read_dir(&data_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_dir() {
            continue;
        }

        let mut npz = NpzReader::new(File::open(entry.path()).unwrap()).unwrap();
        let samples: Array1<f64> = npz.by_name("audio.npy").unwrap();
        let result: Array2<f32> = npz.by_name("magnitude.npy").unwrap();
        let params: Array1<f32> = npz.by_name("params.npy").unwrap();

        if !result.is_empty() {
            let max_possible_bin_value = samples.iter().fold(0.0, |acc, x| acc + x.powi(2));
            let eps = (max_possible_bin_value.powf(params[3] as f64) * 2e-4) as f32;

            let stft = stft_from_params(params.view());
            let spectra = samples.spectrum(stft, Some(params[3])).unwrap();
            assert_eq!(spectra.dim(), result.dim());
            for (a, e) in spectra.iter().zip(result.iter()) {
                assert!(
                    approx_eq!(f32, *a, *e, epsilon = eps),
                    "{} ~= {} (+/- {})",
                    a,
                    e,
                    eps
                );
            }
        }
    }
}
