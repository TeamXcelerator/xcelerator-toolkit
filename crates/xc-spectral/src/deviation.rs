// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Decomposition of a profile deviation against a runtime-supplied auxiliary profile.
//!
//! The normalized even CCM eigenfunction departs from the target `τ` by a
//! shape, not by noise. That shape is the reference deviation profile of
//! [`crate::target`], and the decomposition records how much of it is present:
//!
//! ```text
//!   D(u) = f_{N,λ}(u) − τ(u)
//!   a₁   = ⟨D, g⟩ / ⟨g, g⟩
//!   R(u) = D(u) − a₁ g(u)
//! ```
//!
//! `a₁` and the residual are facts about one `(λ², N)` configuration. Laws
//! relating them across configurations — how `a₁` scales, where it changes
//! sign, what structure the residual retains — are not computed here.
//!
//! ## Two metrics, both recorded
//!
//! The distance functional is `d(N,λ) = ∫₁^λ |f − τ| u^{−1/2} du`, and there
//! are two defensible ways to read that weight as an inner product: applied to
//! each factor, or applied once to the product. They are not equivalent — the
//! choice moves derived norm ratios by several percent. Neither is privileged
//! here, so [`project`] takes the metric explicitly and callers are expected to
//! record which one produced a number. See [`DeviationMetric`].
//!
//! ## Behavior at a crossing
//!
//! `a₁` passes through zero at a cutoff-dependent `N`, where the deviation is
//! carried by other structure instead. That is a fact to record, not a failure:
//! the projection stays well defined, only `relative_residual` approaches one.
//! Nothing here rejects such a configuration, because those are precisely the
//! configurations that locate the crossing.

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
pub mod hp {
    use super::DeviationMetric;
    use anyhow::{bail, Result};
    use rug::Float;

    /// One projection of a deviation onto one reference, in one metric.
    #[derive(Clone, Debug)]
    pub struct DeviationProjection {
        /// `a₁ = ⟨D, g⟩ / ⟨g, g⟩`. Signed; passes through zero at a crossing.
        pub amplitude: Float,
        /// `‖D‖`.
        pub deviation_norm: Float,
        /// `‖g‖`.
        pub reference_norm: Float,
        /// `‖D − a₁ g‖`.
        pub residual_norm: Float,
        /// `‖D − a₁ g‖ / ‖D‖`, or zero when the deviation vanishes.
        pub relative_residual: Float,
    }

    fn weight(metric: DeviationMetric, u: &Float, prec: u32) -> Float {
        let u = Float::with_val(prec, u);
        match metric {
            DeviationMetric::FactorWeighted => u.recip(),
            DeviationMetric::IntegrandWeighted => u.sqrt().recip(),
        }
    }

    /// Trapezoidal `∫ a b w` over the supplied grid.
    ///
    /// The grid is the profile's own sample grid, so the rule is the one the
    /// profile was built for rather than an independent quadrature.
    fn inner_product(
        us: &[Float],
        a: &[Float],
        b: &[Float],
        metric: DeviationMetric,
        prec: u32,
    ) -> Float {
        let mut total = Float::with_val(prec, 0u32);
        for index in 0..us.len().saturating_sub(1) {
            let mut left = Float::with_val(prec, &a[index]);
            left *= &b[index];
            left *= &weight(metric, &us[index], prec);

            let mut right = Float::with_val(prec, &a[index + 1]);
            right *= &b[index + 1];
            right *= &weight(metric, &us[index + 1], prec);

            let mut step = Float::with_val(prec, &us[index + 1]);
            step -= &us[index];

            let mut cell = left;
            cell += &right;
            cell /= 2u32;
            cell *= &step;
            total += cell;
        }
        total
    }

    fn norm(us: &[Float], values: &[Float], metric: DeviationMetric, prec: u32) -> Float {
        inner_product(us, values, values, metric, prec).sqrt()
    }

    /// Project `deviation` onto `reference` over `us` in `metric`.
    ///
    /// All three slices must share the profile grid. `us` must be strictly
    /// ascending with at least two points, and `reference` must not be identically
    /// zero.
    pub fn project(
        us: &[Float],
        deviation: &[Float],
        reference: &[Float],
        metric: DeviationMetric,
        prec: u32,
    ) -> Result<DeviationProjection> {
        if us.len() < 2 {
            bail!("deviation projection requires at least two grid points");
        }
        if deviation.len() != us.len() || reference.len() != us.len() {
            bail!(
                "deviation projection sample counts disagree: grid {}, deviation {}, reference {}",
                us.len(),
                deviation.len(),
                reference.len()
            );
        }
        if us.windows(2).any(|pair| pair[1] <= pair[0]) {
            bail!("deviation projection requires a strictly ascending grid");
        }
        if us
            .iter()
            .chain(deviation)
            .chain(reference)
            .any(|v| !v.is_finite())
        {
            bail!("deviation projection requires finite samples");
        }

        let reference_square = inner_product(us, reference, reference, metric, prec);
        if reference_square <= 0u32 {
            bail!("deviation projection requires a reference of positive norm");
        }
        let overlap = inner_product(us, deviation, reference, metric, prec);
        let mut amplitude = overlap;
        amplitude /= &reference_square;

        let residual: Vec<Float> = deviation
            .iter()
            .zip(reference)
            .map(|(d, r)| {
                let mut scaled = Float::with_val(prec, r);
                scaled *= &amplitude;
                let mut value = Float::with_val(prec, d);
                value -= &scaled;
                value
            })
            .collect();

        let deviation_norm = norm(us, deviation, metric, prec);
        let residual_norm = norm(us, &residual, metric, prec);
        let relative_residual = if deviation_norm > 0u32 {
            let mut ratio = Float::with_val(prec, &residual_norm);
            ratio /= &deviation_norm;
            ratio
        } else {
            Float::with_val(prec, 0u32)
        };

        Ok(DeviationProjection {
            amplitude,
            deviation_norm,
            reference_norm: reference_square.sqrt(),
            residual_norm,
            relative_residual,
        })
    }
}

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
