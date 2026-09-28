#![cfg(feature = "hp")]
use xc_operator::LinearOperator;
use xc_variational::maynard::{MkSymmetricIMetricF64, MkSymmetricReference};

#[test]
fn positive_definite_metric_must_reject_total_entry_underflow() {
    // Degree-zero simplex Gram matrix is the 1x1 matrix [1/k!]. Exact
    // positivity follows without an eigensolver. At k=178 it underflows f64.
    let reference = MkSymmetricReference::new(178, 0).unwrap();
    assert!(reference.i_entry(0, 0).unwrap() > 0);
    if let Ok(metric) = MkSymmetricIMetricF64::new(&reference) {
        let mut output = [0.0];
        let result = metric.apply(&[1.0], &mut output);
        eprintln!(
            "I(1)={:?}; action={:?}; metadata={:?}",
            output,
            result,
            metric.metadata()
        );
        assert!(
            result.is_err() || output[0] > 0.0,
            "a PositiveDefiniteMetric adapter returned the zero action successfully"
        );
    }
}

#[test]
fn dense_and_streamed_routes_reject_lost_entries_and_preserve_representable_ones() {
    use xc_variational::maynard::MkMonomialReference;
    let lost = MkSymmetricReference::new(178, 0).unwrap();
    assert!(lost.dense_i_f64().is_err());
    let mut output = [0.0];
    assert!(lost.apply_i_f64(&[1.0], &mut output).is_err());
    let monomials = MkMonomialReference::new(178, 0).unwrap();
    assert!(monomials.dense_i_f64().is_err());
    assert!(monomials.apply_i_f64(&[1.0], &mut output).is_err());
    let finite = MkSymmetricReference::new(170, 0).unwrap();
    let metric = MkSymmetricIMetricF64::new(&finite).unwrap();
    metric.apply(&[1.0], &mut output).unwrap();
    assert!(output[0].is_finite() && output[0] > 0.0);
    metric.apply(&[0.0], &mut output).unwrap();
    assert_eq!(output, [0.0]);
    assert!(metric.apply(&[f64::MIN_POSITIVE], &mut output).is_err());
}
