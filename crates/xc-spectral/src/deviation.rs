// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Decomposition of a profile deviation against a runtime-supplied auxiliary profile.
//!
//! For supplied samples D=f-tau and a reference g, this module measures
//! a=<D,g>/<g,g> and the residual D-a*g in either of two discrete metrics.
//! This does not establish that a particular reference explains a family of
//! deviations, or that an observed component is signal rather than noise.
//!
//! Trapezoidal node weights approximate integrals on the profile domain
//! `[1,lambda]`. FactorWeighted uses 1/u and IntegrandWeighted uses 1/sqrt(u).
//! Each result must retain its metric. No continuum quadrature error is enclosed.
//! A zero overlap is valid; the relative residual is then near one for a nonzero
//! deviation. Trends across cutoffs and model interpretation are separate work.

/// Which inner product the projection is taken in.
///
/// Both integrate over the profile domain `[1, λ]`; they differ only in how
/// the distance functional's `u^{−1/2}` is distributed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviationMetric {
    /// `⟨a,b⟩ = ∫ a(u) b(u) u^{−1} du`, the weight applied to each factor.
    FactorWeighted,
    /// `⟨a,b⟩ = ∫ a(u) b(u) u^{−1/2} du`, the weight of `d(N,λ)` applied once.
    IntegrandWeighted,
}

impl DeviationMetric {
    /// Stable identifier to record alongside any number this metric produced.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FactorWeighted => "factor_weighted_u_inverse",
            Self::IntegrandWeighted => "integrand_weighted_u_inverse_sqrt",
        }
    }
}

#[cfg(feature = "hp")]
#[path = "deviation/projection.rs"]
pub mod hp;

#[cfg(all(test, feature = "hp"))]
mod tests {
    use super::hp::project;
    use super::DeviationMetric;
    use rug::Float;

    const PREC: u32 = 192;

    /// A profile-like grid: uniform in `u` over `[1, λ]`, as the eigenfunction
    /// profile artifact samples it.
    fn grid(lambda: f64, steps: usize) -> Vec<Float> {
        (0..=steps)
            .map(|k| {
                let t = (k as f64) / (steps as f64);
                Float::with_val(PREC, 1.0 + t * (lambda - 1.0))
            })
            .collect()
    }

    fn reference_samples(us: &[Float]) -> Vec<Float> {
        us.iter().map(|u| Float::with_val(PREC, u - 1u32)).collect()
    }

    fn scaled(values: &[Float], factor: f64) -> Vec<Float> {
        values
            .iter()
            .map(|v| Float::with_val(PREC, v) * Float::with_val(PREC, factor))
            .collect()
    }

    /// An exact multiple of the reference must return that multiple and leave
    /// nothing behind, in either metric.
    #[test]
    fn an_exact_multiple_of_the_reference_is_recovered_with_no_residual() {
        let us = grid(4.0, 400);
        let reference = reference_samples(&us);
        for metric in [
            DeviationMetric::FactorWeighted,
            DeviationMetric::IntegrandWeighted,
        ] {
            for factor in [1.0_f64, -2.5, 1e-6] {
                let deviation = scaled(&reference, factor);
                let got = project(&us, &deviation, &reference, metric, PREC).unwrap();
                let error = Float::with_val(PREC, &got.amplitude - Float::with_val(PREC, factor))
                    .abs()
                    / factor.abs();
                assert!(
                    error < Float::with_val(PREC, 1e-40),
                    "{metric:?} factor {factor}: amplitude {:?}",
                    got.amplitude
                );
                assert!(
                    got.relative_residual < Float::with_val(PREC, 1e-40),
                    "{metric:?} factor {factor}: residual {:?}",
                    got.relative_residual
                );
            }
        }
    }

    /// The residual is what the reference cannot explain, so it must be orthogonal
    /// to the reference in the metric the projection was taken in.
    #[test]
    fn the_residual_is_orthogonal_to_the_reference() {
        let us = grid(4.0, 400);
        let reference = reference_samples(&us);
        // A deviation the reference cannot fully explain.
        let deviation: Vec<Float> = us
            .iter()
            .zip(&reference)
            .map(|(u, r)| {
                let mut value = Float::with_val(PREC, r);
                value *= 3u32;
                value += Float::with_val(PREC, u).recip();
                value
            })
            .collect();
        let metric = DeviationMetric::FactorWeighted;
        let got = project(&us, &deviation, &reference, metric, PREC).unwrap();

        let residual: Vec<Float> = deviation
            .iter()
            .zip(&reference)
            .map(|(d, r)| {
                let mut scaled = Float::with_val(PREC, r);
                scaled *= &got.amplitude;
                Float::with_val(PREC, d) - scaled
            })
            .collect();
        let reprojected = project(&us, &residual, &reference, metric, PREC).unwrap();
        assert!(
            reprojected.amplitude.clone().abs() < Float::with_val(PREC, 1e-35),
            "residual retains reference content: {:?}",
            reprojected.amplitude
        );
        assert!(got.relative_residual > Float::with_val(PREC, 0u32));
    }

    /// The two metrics are genuinely different readings of the same weight, so
    /// an amplitude is only meaningful with its metric attached.
    #[test]
    fn the_two_metrics_disagree_on_a_generic_deviation() {
        let us = grid(4.0, 400);
        let reference = reference_samples(&us);
        let deviation: Vec<Float> = us
            .iter()
            .map(|u| Float::with_val(PREC, u).recip())
            .collect();
        let a = project(
            &us,
            &deviation,
            &reference,
            DeviationMetric::FactorWeighted,
            PREC,
        )
        .unwrap();
        let b = project(
            &us,
            &deviation,
            &reference,
            DeviationMetric::IntegrandWeighted,
            PREC,
        )
        .unwrap();
        let spread = Float::with_val(PREC, &a.amplitude - &b.amplitude).abs()
            / Float::with_val(PREC, &a.amplitude).abs();
        assert!(
            spread > Float::with_val(PREC, 1e-3),
            "metrics agree to {spread:?}; the distinction would be cosmetic"
        );
    }

    /// A configuration sitting on the crossing has `a₁ ≈ 0`. It must be
    /// recorded, not rejected: those configurations locate the crossing.
    #[test]
    fn a_vanishing_amplitude_is_recorded_rather_than_rejected() {
        let us = grid(4.0, 400);
        let reference = reference_samples(&us);
        // Orthogonal-by-construction deviation: the residual is everything.
        let deviation: Vec<Float> = us
            .iter()
            .map(|u| Float::with_val(PREC, u).recip())
            .collect();
        let metric = DeviationMetric::FactorWeighted;
        let first = project(&us, &deviation, &reference, metric, PREC).unwrap();
        let purged: Vec<Float> = deviation
            .iter()
            .zip(&reference)
            .map(|(d, r)| {
                let mut scaled = Float::with_val(PREC, r);
                scaled *= &first.amplitude;
                Float::with_val(PREC, d) - scaled
            })
            .collect();
        let got = project(&us, &purged, &reference, metric, PREC).unwrap();
        assert!(got.amplitude.clone().abs() < Float::with_val(PREC, 1e-35));
        let one = Float::with_val(PREC, 1u32);
        assert!(
            Float::with_val(PREC, &got.relative_residual - &one).abs()
                < Float::with_val(PREC, 1e-30),
            "relative residual {:?} should approach one",
            got.relative_residual
        );
    }

    #[test]
    fn malformed_input_is_rejected() {
        let us = grid(4.0, 8);
        let reference = reference_samples(&us);
        assert!(project(
            &us,
            &reference[..4],
            &reference,
            DeviationMetric::FactorWeighted,
            PREC
        )
        .is_err());
        assert!(project(
            &us[..1],
            &reference[..1],
            &reference[..1],
            DeviationMetric::FactorWeighted,
            PREC
        )
        .is_err());

        let mut descending = us.clone();
        descending.swap(2, 3);
        assert!(project(
            &descending,
            &reference,
            &reference,
            DeviationMetric::FactorWeighted,
            PREC
        )
        .is_err());

        let zero = vec![Float::with_val(PREC, 0u32); us.len()];
        assert!(project(
            &us,
            &reference,
            &zero,
            DeviationMetric::FactorWeighted,
            PREC
        )
        .is_err());
    }
}
