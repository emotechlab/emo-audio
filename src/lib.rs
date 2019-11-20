/// Preemphasis extension trait for 1D arrays
pub mod preemphasis;
/// Takes an audio signal and returns a spectrogram
pub mod spectrum;
/// Emo-audio is designed to contain a set of audio processing routines built
/// around ndarray.

/// Module containing an implementation of a short time fourier transform
pub mod stft;

/// Common imports
pub mod prelude {
    pub use crate::preemphasis::*;
    pub use crate::spectrum::*;
    pub use crate::stft::*;
}
