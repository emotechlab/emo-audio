/// Emo-audio is designed to contain a set of audio processing routines built
/// around ndarray.
use ndarray::Array2;
use rustfft::num_complex::Complex;

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
    fn mag(&self) -> Array2<f32> {
        self.mapv(|x| (x.re.powi(2) + x.im.powi(2)).sqrt())
    }

    fn phase(&self) -> Array2<f32> {
        unimplemented!()
    }
}
