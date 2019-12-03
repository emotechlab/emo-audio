use crate::stft::ShortTimeFourierTransform;
use ndarray::{prelude::*, Array2, Data};
use num_traits::{Bounded, Num, NumCast};
use rustfft::num_complex::Complex;
use std::f32::{consts::PI, NAN};

/// Get the different frequency components from a type containing frequency data
pub trait FrequencyComponents {
    /// Gets the magnitude data as a 2D plot
    fn mag(&self) -> Array2<f32>;
    /// Gets the phase data as a 2D plot
    fn phase(&self) -> Array2<f32>;
}

impl<T> FrequencyComponents for ArrayBase<T, Ix2>
where
    T: Data<Elem = Complex<f32>>,
{
    /// Convert the complex array into it's magnitude spectra
    fn mag(&self) -> Array2<f32> {
        self.mapv(|x| x.norm())
    }

    /// Convert every complex number into it's phase. Result may contain NAN
    /// values for instances where `Im(x) == 0 && Re(x) == 0`
    fn phase(&self) -> Array2<f32> {
        self.mapv(|x| {
            if x.re > 0.0 || x.im.abs() > std::f32::EPSILON {
                2.0 * (x.im / (x.norm() + x.re)).atan()
            } else if x.re < 0.0 && x.im.abs() <= std::f32::EPSILON {
                PI
            } else {
                NAN
            }
        })
    }
}

/// Gets a spectrogram from an audio signal
pub trait SpectrumExt {
    type Output;

    /// Given an stft object create a spectrogram with the given power and
    /// stft parameters. If no power is provided 1.0 is used as default
    fn spectrum(&self, stft: ShortTimeFourierTransform, power: Option<f32>) -> Self::Output;
}

impl<T, U> SpectrumExt for ArrayBase<T, Ix1>
where
    T: Data<Elem = U>,
    U: Num + Bounded + NumCast,
{
    type Output = Option<Array<f32, Ix2>>;

    fn spectrum(&self, stft: ShortTimeFourierTransform, power: Option<f32>) -> Self::Output {
        let power = power.unwrap_or(1.0);
        if let Some(data) = self.as_slice() {
            stft.run(data).map(|r| {
                let mut mag = r.mag();
                mag.mapv_inplace(|x| x.powf(power));
                mag
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {}
