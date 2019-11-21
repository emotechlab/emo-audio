use float_cmp::approx_eq;
use ndarray_npy::NpzReader;
use std::env;
use std::fs::{read_dir, File};
use std::path::PathBuf;
use std::process::Command;

use emo_audio::prelude::*;
use ndarray::prelude::*;

fn check_data_folder() -> PathBuf {
    let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    env::set_current_dir(&test_dir).unwrap();
    let data_dir = test_dir.join("data");
    if !data_dir.exists() {
        Command::new("python")
            .arg("init_data_dir.py")
            .arg("-f=data")
            .arg("-s 10")
            .output()
            .expect("Failed to run equivalence tests");
    }
    data_dir
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
            let stft = StftBuilder::new()
                .set_fft_num(params[0] as usize)
                .set_window_len(params[1] as usize)
                .set_hop_len(params[2] as usize)
                .set_centred(true)
                .build();

            let spectra = samples.spectrum(stft, Some(params[3] as f32)).unwrap();
            assert_eq!(spectra.dim(), result.dim());
            for (a, e) in spectra.iter().zip(result.iter()) {
                println!("Comparing {} and {}", a, e);
                assert!(approx_eq!(f32, *a, *e, epsilon = 1e-4));
            }
        }
    }
}
