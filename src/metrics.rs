use num_traits::{Num, NumCast};

/// Calculate the power of a signal
pub fn power<T: Num + NumCast>(samples: &[T]) -> f64 {
    samples
        .iter()
        .map(|x| x.to_f64().unwrap_or_default().powi(2))
        .fold(0.0, |acc, x| x + acc)
        / (samples.len() as f64)
}

/// Calculate the root mean square of the signal
pub fn rms<T: Num + NumCast>(samples: &[T]) -> f64 {
    power(samples).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn simple_power() {
        let data = vec![0, 2, 5, 3, 4];
        let power_act = power(data.as_slice());
        assert!(
            approx_eq!(f64, power_act, 10.8, ulps = 3),
            "{} ~= {}",
            power_act,
            10.8
        );
        let data = vec![0, 1, 2, 3, 4, 5];
        let power_act = power(data.as_slice());
        let expected = 9.166666666666666666;
        assert!(
            approx_eq!(f64, power_act, expected, ulps = 3),
            "{} ~= {}",
            power_act,
            expected
        );
    }

    #[test]
    fn simple_rms() {
        let data = vec![0, 1, 2, 3, 4, 5];
        let power_root = power(data.as_slice()).sqrt();
        let rms_act = rms(data.as_slice());
        assert!(
            approx_eq!(f64, power_root, rms_act, ulps = 3),
            "{} ~= {}",
            power_root,
            rms_act
        );
    }
}
