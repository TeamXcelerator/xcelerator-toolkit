// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.
#![cfg(feature = "hp")]

use rug::{float::Special, Float, Rational};
use serde_json::Value;
use xc_numerics::interval::RationalInterval;
use xc_spectral::ccm::evidence::{
    finite_cutoff_decision, finite_cutoff_interval_decision, try_finite_cutoff_decision,
    ArchimedeanTailBudget, FiniteCutoffDecision,
};

fn parse(p: u32, s: &str) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn rational(s: &str) -> Rational {
    Rational::from_str_radix(s, 10).unwrap()
}

#[test]
fn explicit_budgets_enclose_independent_arb_bounds_at_all_requested_precisions() {
    let data: Value =
        serde_json::from_str(include_str!("fixtures/tail_budget_oracle.json")).unwrap();
    for p in [64, 128, 256] {
        for row in data["cases"].as_array().unwrap() {
            let c = row["c"].as_u64().unwrap();
            let Ok(n) = usize::try_from(row["n"].as_u64().unwrap()) else {
                continue;
            };
            let t = parse(p, row["t"].as_str().unwrap());
            let budget = ArchimedeanTailBudget::explicit(c, n, &t).unwrap();
            let upper = budget.upper_bound().to_rational().unwrap();
            let reference_lower = rational(row["lower"].as_str().unwrap());
            let reference_upper = rational(row["upper"].as_str().unwrap());
            assert!(upper >= reference_upper, "p={p},c={c},N={n}");
            let mut relative = upper - reference_lower.clone();
            relative /= &reference_lower;
            let mut tolerance = Rational::from(1);
            tolerance >>= p - 8;
            assert!(
                relative <= tolerance,
                "excessive enclosure inflation, p={p},c={c},N={n}"
            );
            assert_eq!(budget.integer_cutoff_c(), c);
            assert_eq!(budget.modes(), n);
            assert_eq!(budget.cutoff_t(), &t);
            assert!(budget.theorem_threshold() < &t);
            assert!(budget.log_cutoff().is_strictly_positive());
            assert!(budget.rho().is_strictly_positive());
        }
    }
}

#[test]
fn theorem_domain_rejects_wrapped_modes_nonfinite_values_and_unsupported_precision() {
    let t = Float::with_val(128, 100);
    for (c, n) in [(0, 1), (1, 1), (13, 0)] {
        assert!(ArchimedeanTailBudget::explicit(c, n, &t).is_err());
    }
    for value in [
        Float::with_val(128, 7),
        Float::with_val(128, -100),
        Float::with_val(128, Special::Infinity),
        Float::with_val(128, Special::Nan),
        Float::with_val(2, 100),
    ] {
        assert!(ArchimedeanTailBudget::explicit(13, 4, &value).is_err());
    }
    if let Ok(n) = usize::try_from(1u64 << 32) {
        assert!(ArchimedeanTailBudget::explicit(13, n, &t).is_err());
        assert!(ArchimedeanTailBudget::explicit(13, n + 1, &t).is_err());
    }
    let valid = ArchimedeanTailBudget::explicit(13, 4, &t).unwrap();
    let threshold = valid.theorem_threshold().clone();
    // A point just below the lower rho*N enclosure is certainly outside.
    let band = valid
        .rho()
        .mul(&xc_numerics::mpfr_interval::MpfrInterval::from_i64(4, 160));
    let mut outside = band.lower().clone();
    outside.next_down();
    assert!(ArchimedeanTailBudget::explicit(13, 4, &outside).is_err());
    assert!(threshold < t);
    for x in [Special::Infinity, Special::NegInfinity, Special::Nan] {
        let x = Float::with_val(128, x);
        assert!(try_finite_cutoff_decision(&x, &valid).is_err());
        assert_eq!(
            finite_cutoff_decision(&x, &valid),
            FiniteCutoffDecision::InconclusiveTailBand
        );
    }
}

#[test]
fn interval_decisions_use_the_correct_endpoint_and_strict_negative_boundary() {
    use FiniteCutoffDecision::*;
    let budget = ArchimedeanTailBudget::explicit(100, 200, &Float::with_val(192, 800)).unwrap();
    let b = budget.upper_bound().to_rational().unwrap();
    let minus_b = -b.clone();
    let cases = [
        (Rational::from(0), Rational::from(1), CutoffFreePositive),
        (Rational::from(-1), Rational::from(1), InconclusiveTailBand),
        (minus_b.clone() * 2, minus_b.clone(), InconclusiveTailBand),
        (minus_b.clone() * 3, minus_b.clone() * 2, CutoffFreeNegative),
        (minus_b.clone(), Rational::from(0), InconclusiveTailBand),
    ];
    for (lower, upper, expected) in cases {
        let interval = RationalInterval::new(lower, upper).unwrap();
        assert_eq!(
            finite_cutoff_interval_decision(&interval, &budget),
            expected
        );
    }
    assert_eq!(
        try_finite_cutoff_decision(&Float::with_val(64, 0), &budget).unwrap(),
        CutoffFreePositive
    );
    for (value, expected) in [
        ("1e200000000", CutoffFreePositive),
        ("-1e200000000", CutoffFreeNegative),
        ("-1e-200000000", InconclusiveTailBand),
    ] {
        assert_eq!(
            try_finite_cutoff_decision(&parse(128, value), &budget).unwrap(),
            expected
        );
    }
}
