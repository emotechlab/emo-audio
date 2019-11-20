use ndarray::{prelude::*, s};
use num_traits::{Bounded, Num, NumCast};
use rustfft::{num_complex::Complex, FFTplanner};
use std::f32::consts::PI;

/// Padding mode for the audio signal
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PadMode {
    /// Don't apply any padding
    NoPad,
    /// Reflect the signal
    Reflect,
}

/// Windowing algorithm to be applied to the signal. While this only supports
/// Hann windowing at the current time it can be extended to support any
/// windowing functions provided by scipy
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WindowingAlgorithm {
    /// Implements the Hann function on a fourier signal
    Hann(usize),
}

impl WindowingAlgorithm {
    /// Run the windowing algorithm on a given element
    pub fn run(self, n: Complex<f32>) -> Complex<f32> {
        let mut copy = n;
        self.run_inplace(&mut copy);
        copy
    }

    /// Run the windowing algorithm inplace on the given element
    pub fn run_inplace(self, n: &mut Complex<f32>) {
        match self {
            Self::Hann(len) => {
                if len == 0 {
                    panic!("Window length cannot be zero");
                } else if len == 1 {
                    n.re = 1.0;
                    n.im = 0.0;
                } else {
                    let m = if len % 2 == 0 {
                        (len + 1) as f32
                    } else {
                        len as f32
                    };
                    n.re = 0.5 - 0.5 * (2.0 * PI * n.re / m).cos();
                    n.im = 0.5 - 0.5 * (2.0 * PI * n.im / m).cos();
                }
            }
        }
    }
}

///Builds a ShortTimeFourierTransform instance using the supplied parameters
///and setting sensible defaults for unset parameters
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StftBuilder {
    n_fft: usize,
    win_length: Option<usize>,
    hop_length: Option<usize>,
    win_alg: Option<WindowingAlgorithm>,
    centred: Option<bool>,
    pad_mode: Option<PadMode>,
}

impl StftBuilder {
    /// Create a new StftBuilder
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            win_length: None,
            hop_length: None,
            win_alg: None,
            centred: None,
            pad_mode: None,
        }
    }

    /// Set the number of FFT bins
    pub fn set_fft_num(mut self, n: usize) -> Self {
        self.n_fft = n;
        self
    }

    /// Set the length of the signal to be windowed
    pub fn set_window_len(mut self, n: usize) -> Self {
        self.win_length = Some(n);
        self
    }

    /// Set the hop length (or stride) to increment along the signal
    pub fn set_hop_len(mut self, n: usize) -> Self {
        self.hop_length = Some(n);
        self
    }

    /// Set the windowing algorithm
    pub fn set_windowing_algorithm(mut self, alg: WindowingAlgorithm) -> Self {
        self.win_alg = Some(alg);
        self
    }

    /// Sets whether the signal should be centred
    pub fn set_centred(mut self, centred: bool) -> Self {
        self.centred = Some(centred);
        self
    }

    /// Set the type of padding to be applied to the signal
    pub fn set_padding_mode(mut self, padding: PadMode) -> Self {
        self.pad_mode = Some(padding);
        self
    }

    /// Build the ShortTimeFourierTransform instance. This uses the following
    /// derivation for the defaults as defined by librosa
    ///
    /// N_FFT default 2048
    /// window_length default same as N_FFT value
    /// hop_length default window_length/4
    /// centred default True
    /// pad_mode default Reflect
    /// window_algorithm: Hann
    pub fn build(self) -> ShortTimeFourierTransform {
        let n_fft = if self.n_fft > 0 { self.n_fft } else { 2048 };
        let win_length = self.win_length.unwrap_or(n_fft);
        let hop_length = self.hop_length.unwrap_or(win_length / 4);
        let centred = self.centred.unwrap_or(true);
        let pad_mode = self.pad_mode.unwrap_or(PadMode::Reflect);
        let win_alg = self
            .win_alg
            .unwrap_or_else(|| WindowingAlgorithm::Hann(win_length));
        ShortTimeFourierTransform {
            n_fft,
            hop_length,
            win_length,
            centred,
            pad_mode,
            win_alg,
        }
    }
}

/// Object to execute short time fourier transforms with the provided parameters.
/// Building using the StftBuilder
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShortTimeFourierTransform {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    win_alg: WindowingAlgorithm,
    centred: bool,
    pad_mode: PadMode,
}

impl Default for ShortTimeFourierTransform {
    fn default() -> Self {
        let win_length = 16000 / 25;
        Self {
            n_fft: 1024,
            hop_length: 16000 / 100,
            win_length,
            win_alg: WindowingAlgorithm::Hann(win_length),
            centred: true,
            pad_mode: PadMode::Reflect,
        }
    }
}

impl ShortTimeFourierTransform {
    /// Creates a new default ShortTimeFourierTransform
    pub fn new() -> Self {
        Self::default()
    }

    /// Run on a set of input samples
    pub fn run<T: Num + Bounded + NumCast>(&self, samples: &[T]) -> Option<Array2<Complex<f32>>> {
        if samples.len() < 2 {
            return None;
        }
        let mut input: Vec<Complex<f32>> = samples
            .iter()
            .map(|x| Complex::new(x.to_f32().unwrap_or_default(), 0.0))
            .collect();

        let rows = 1 + self.n_fft / 2;

        self.apply_padding(&mut input);

        let window_mat = self.get_window_matrix();
        // Not implementing memory limiting initially
        let frame_len = (input.len() - self.n_fft + 1) as f32 / self.hop_length as f32;
        let frame_len = frame_len.ceil() as usize;

        let mut result = unsafe { Array2::uninitialized((rows, frame_len)) };
        let mut scratchpad = unsafe { Array2::uninitialized((self.n_fft, frame_len)) };
        let fft = FFTplanner::new(false).plan_fft(self.n_fft);

        for i in 0..self.n_fft {
            let win_coef = window_mat[i];
            let data = input
                .iter()
                .skip(i)
                .step_by(self.hop_length)
                .take(frame_len)
                .map(|x| *x * win_coef)
                .collect::<Vec<_>>();

            scratchpad.row_mut(i).assign(&Array1::from(data));
        }

        let mut temp = Vec::with_capacity(self.n_fft);
        for (idx, mut col) in result.lanes_mut(Axis(0)).into_iter().enumerate() {
            let mut column = scratchpad.column(idx).into_owned();
            temp.clear();
            temp.resize(self.n_fft, Complex::new(0.0, 0.0));
            if let Some(c) = column.view_mut().into_slice() {
                fft.process(c, temp.as_mut_slice());
            }
            // Get rid of the mirrored values
            temp.resize(rows, Complex::new(0.0, 0.0));
            col.assign(&Array1::from(temp.clone()));
        }
        Some(result)
    }

    /// Get the signal matrix post windowing
    fn get_window_matrix(&self) -> Array1<Complex<f32>> {
        let result: Array1<f32> =
            Array1::linspace(0.0, self.win_length as f32 - 1.0, self.win_length);
        let mut result: Array1<Complex<f32>> = result.mapv(|x| Complex::new(x, 0.0));
        result.mapv_inplace(|x| self.win_alg.run(x));
        if self.n_fft != self.win_length {
            let mut padded_result = Array1::from_elem(self.n_fft, Complex::new(0.0, 0.0));
            let win_start = (self.n_fft - self.win_length) / 2;
            padded_result
                .slice_mut(s![win_start..win_start + self.win_length])
                .assign(&result);
            padded_result
        } else {
            result
        }
    }

    /// Apply padding to the input signal
    fn apply_padding(&self, arr: &mut Vec<Complex<f32>>) {
        if self.centred {
            // Apply padding
            let pad_width = self.n_fft / 2;
            if self.pad_mode == PadMode::Reflect {
                let mut pos = (arr.len() - 2) as i64;
                let mut delta = -1;
                for _ in 0..pad_width {
                    arr.push(arr[pos as usize]);

                    if pos == 0 {
                        delta = 1;
                    } else if pos == arr.len() as i64 - 1 {
                        delta = -1;
                    }
                    pos += delta;
                }
                let mut pos = 1;
                // We've padded to the end so won't overflow
                for i in 0..pad_width {
                    arr.insert(0, arr[pos + i]);
                    pos += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn representative_usage() {
        // Test that stft can handle a long signal. Just going to use a very
        // long ramp. This is because the original implementation chunked
        // the data into smaller batches
        let data = Array::range(0.0, 1.0, 1e-6);

        // Use a more representative STFT config
        let stft = StftBuilder::new()
            .set_fft_num(512)
            .set_centred(true)
            .build();

        assert!(stft.run(data.as_slice().unwrap()).is_some())
    }

    #[test]
    fn signal_padding() {
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
        ];

        let small = StftBuilder::new().set_fft_num(5).set_centred(true).build();

        small.apply_padding(&mut data);

        let expected = vec![
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(3.0, 0.0),
        ];

        assert_eq!(data, expected);
    }

    #[test]
    fn long_signal_padding() {
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
        ];

        let small = StftBuilder::new().set_fft_num(10).set_centred(true).build();

        small.apply_padding(&mut data);

        let expected = vec![
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
        ];

        assert_eq!(data, expected);
    }

    #[test]
    fn stft_window() {
        let stft = StftBuilder::new()
            .set_fft_num(5)
            .set_hop_len(1)
            .set_centred(true)
            .set_windowing_algorithm(WindowingAlgorithm::Hann(5))
            .build();

        let win = stft.get_window_matrix();
        let expected = vec![0.0, 0.3454915, 0.9045085, 0.9045085, 0.3454915];
        let expected = Array::from(expected);

        for (a, e) in win.iter().zip(expected.iter()) {
            assert!(approx_eq!(f32, a.re, *e));
            assert!(approx_eq!(f32, a.im, 0.0));
        }
    }

    #[test]
    fn hann_window() {
        let hann = WindowingAlgorithm::Hann(5);
        let expected = vec![0.0, 0.3454915, 0.9045085, 0.9045085, 0.3454915];

        for i in 0..5 {
            let res = hann.run(Complex::new(i as f32, 0.0));
            assert!(approx_eq!(f32, res.re, expected[i]));
            assert!(approx_eq!(f32, res.im, 0.0));
        }
    }

    #[test]
    fn empty_signal() {
        let small = StftBuilder::new()
            .set_fft_num(5)
            .set_hop_len(1)
            .set_centred(true)
            .set_windowing_algorithm(WindowingAlgorithm::Hann(5))
            .build();

        let empty: Vec<f32> = vec![];
        let result = small.run(empty.as_slice());
        assert_eq!(result, None);
    }

    #[test]
    fn short_sample() {
        let small = StftBuilder::new()
            .set_fft_num(5)
            .set_hop_len(1)
            .set_centred(true)
            .set_windowing_algorithm(WindowingAlgorithm::Hann(5))
            .build();

        let result = small.run(&(1..20).collect::<Vec<_>>()).unwrap();
        assert_eq!(result.shape(), &[3, 19]);
        // Generated via
        // librosa.stft(np.arange(1.0, 20.0), n_fft=5, hop_length=1, win_length=5, window='hann')
        let expected = vec![
            Complex::new(4.4409828, 0.0),
            Complex::new(6.25, 0.0),
            Complex::new(8.75, 0.0),
            Complex::new(11.25, 0.0),
            Complex::new(13.75, 0.0),
            Complex::new(16.25, 0.0),
            Complex::new(18.75, 0.0),
            Complex::new(21.25, 0.0),
            Complex::new(23.75, 0.0),
            Complex::new(26.25, 0.0),
            Complex::new(28.75, 0.0),
            Complex::new(31.25, 0.0),
            Complex::new(33.75, 0.0),
            Complex::new(36.25, 0.0),
            Complex::new(38.75, 0.0),
            Complex::new(41.25, 0.0),
            Complex::new(43.75, 0.0),
            Complex::new(45.559017, 0.0),
            Complex::new(45.559017, 0.0),
            Complex::new(-1.6614756, 0.8602387),
            Complex::new(-3.125, 1.5174026),
            Complex::new(-4.375, 1.5174026),
            Complex::new(-5.625, 1.5174026),
            Complex::new(-6.875, 1.5174026),
            Complex::new(-8.125, 1.5174026),
            Complex::new(-9.375, 1.5174026),
            Complex::new(-10.625, 1.5174026),
            Complex::new(-11.875, 1.5174026),
            Complex::new(-13.125, 1.5174026),
            Complex::new(-14.375, 1.5174026),
            Complex::new(-15.625, 1.5174026),
            Complex::new(-16.875, 1.5174026),
            Complex::new(-18.125, 1.5174026),
            Complex::new(-19.375, 1.5174026),
            Complex::new(-20.625, 1.5174026),
            Complex::new(-21.875, 1.5174026),
            Complex::new(-23.338526, 0.8602387),
            Complex::new(-23.338526, -0.8602387),
            Complex::new(-0.559017, -0.6571639),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(-4.44089e-16, -0.25101426),
            Complex::new(4.4408921e-16, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(-8.8817842e-16, -0.25101426),
            Complex::new(-8.8817842e-16, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(0.0, -0.25101426),
            Complex::new(-1.7763568e-17, -0.25101426),
            Complex::new(-1.7763568e-15, -0.25101426),
            Complex::new(5.59017e-1, -0.6571639),
            Complex::new(5.59017e-1, 0.6571639),
        ];

        let expected = Array2::from_shape_vec((3, 19), expected).unwrap();
        assert_eq!(expected.shape(), result.shape());

        for entry in result.iter().zip(expected.iter()) {
            let res = entry.0;
            let exp = entry.1;
            assert!(approx_eq!(f32, res.re, exp.re, epsilon = 0.000005));
            assert!(approx_eq!(f32, res.im, exp.im, epsilon = 0.000005));
        }
    }
}
