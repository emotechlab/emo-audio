/// Emo-audio is designed to contain a set of audio processing routines built
/// around ndarray.
use ndarray::Array2;
use rustfft::num_complex::Complex;
use std::f32::consts::{NAN, PI};

/// Module containing an implementation of a short time fourier transform
pub mod stft;

/// Get the different frequency components from a type containing frequency data
pub trait FrequencyComponents {
    /// Gets the magnitude data as a 2D plot
    fn mag(&self) -> Array2<f32>;
    /// Gets the phase data as a 2D plot
    fn phase(&self) -> Array2<f32>;
}

impl FrequencyComponents for Array2<Complex<f32>> {
    /// Convert the complex array into it's magnitude spectra
    fn mag(&self) -> Array2<f32> {
        self.mapv(|x| (x.re.powi(2) + x.im.powi(2)).sqrt())
    }

    /// Convert every complex number into it's phase. Result may contain NAN
    /// values for instances where `Im(x) == 0 && Re(x) == 0`
    fn phase(&self) -> Array2<f32> {
        self.mapv(|x| {
            if x.re > 0.0 || x.im.abs() > std::f32::EPSILON {
                let mag = (x.re.powi(2) + x.im.powi(2)).sqrt();
                2.0 * (x.im / (mag + x.re)).atan()
            } else if x.re < 0.0 && x.im.abs() <= std::f32::EPSILON {
                PI
            } else {
                NAN
            }
        })
    }
}
