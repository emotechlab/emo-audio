use ndarray::{prelude::*, Data, ScalarOperand};
use num_traits::{real::Real, NumAssignOps};

/// Implements a preemphasis extension trait on data. This transforms a sample
/// `T_n` in the following way `T'_n = T_n - coefficient * T_{n-1}`
///
/// Pre-emphasis increases the magnitude of a band of signals (usually high
/// frequency components). This is to mitigate the effects of greater amplitude
/// noise in those frequency components by boosting that range
pub trait PreemphasisExt<T>
where
    T: Real,
{
    /// Output type
    type Output;
    /// Run preemphasis on a signal return a new altered version
    fn preemphasis(&self, coefficient: T) -> Self::Output;
}

impl<T, U> PreemphasisExt<U> for ArrayBase<T, Ix1>
where
    T: Data<Elem = U>,
    U: Real + NumAssignOps + ScalarOperand,
{
    type Output = Array<U, Ix1>;

    fn preemphasis(&self, coefficient: U) -> Self::Output {
        let mut shift = Array1::zeros(self.len());
        shift.slice_mut(s![1..]).assign(&self.slice(s![..-1]));
        shift *= coefficient;

        self - &shift
    }
}

#[cfg(test)]
mod tests {
    use super::PreemphasisExt;
    use float_cmp::approx_eq;
    use ndarray::Array1;

    #[test]
    fn preemphasis_small_signal() {
        let data: Array1<f64> = vec![1.0f64, 2.0f64, 3.0f64, 4.0f64].into();
        let preemp = data.preemphasis(0.9f64);
        let expected: Vec<f64> = vec![1.0, 1.1, 1.2, 1.3];
        for (a, e) in preemp.iter().zip(expected.iter()) {
            assert!(approx_eq!(f64, *a, *e));
        }

        let preemp_view = data.view().preemphasis(0.9f64);
        for (a, e) in preemp_view.iter().zip(expected.iter()) {
            assert!(approx_eq!(f64, *a, *e));
        }
    }
}
