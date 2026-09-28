#![cfg(feature = "hp")]
use rug::Float;
use xc_spectral::deviation::{hp::project, DeviationMetric};

fn f(p: u32, s: &str) -> Float {
    Float::with_val(p, Float::parse(s).unwrap())
}
fn shifts(values: &[Float], e: i32, sign: i32) -> Vec<Float> {
    values
        .iter()
        .map(|v| {
            let mut x = v.clone();
            x <<= e;
            x *= sign;
            x
        })
        .collect()
}

#[test]
fn discrete_projections_match_independent_forms_under_extreme_signed_scales() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/projection_oracle.json")).unwrap();
    let mut comparisons = 0;
    for case in oracle["cases"].as_array().unwrap() {
        for p in [64, 128, 256] {
            let read = |name: &str| {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f(p, v.as_str().unwrap()))
                    .collect::<Vec<_>>()
            };
            let us = read("us");
            let d = read("deviation");
            let g = read("reference");
            let expected: Vec<Float> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f(768, v.as_str().unwrap()))
                .collect();
            let metric = if case["metric"] == "factor" {
                DeviationMetric::FactorWeighted
            } else {
                DeviationMetric::IntegrandWeighted
            };
            for (de, ge, sign) in [
                (0, 0, 1),
                (-700_000_000, -700_000_000, 1),
                (700_000_000, 700_000_000, -1),
                (-600_000_000, 400_000_000, -1),
                (600_000_000, -400_000_000, 1),
                (100, -100, -1),
                (-100, 100, 1),
            ] {
                let got =
                    project(&us, &shifts(&d, de, sign), &shifts(&g, ge, 1), metric, p).unwrap();
                let results = [
                    got.amplitude,
                    got.deviation_norm,
                    got.reference_norm,
                    got.residual_norm,
                    got.relative_residual,
                ];
                let exponents = [de - ge, de, ge, de, 0];
                for (i, (actual, reference)) in results.iter().zip(&expected).enumerate() {
                    assert_eq!(actual.prec(), p);
                    assert!(actual.is_finite());
                    let mut restored = Float::with_val(768, actual);
                    restored >>= exponents[i];
                    if i == 0 {
                        restored *= sign;
                    }
                    let error = Float::with_val(768, &restored - reference).abs();
                    // Absolute tolerance at a zero projection/residual; otherwise relative.
                    let scale = reference.clone().abs().max(&Float::with_val(768, 1));
                    let tolerance = scale * (Float::with_val(768, 1) >> (p - 10));
                    assert!(
                        error < tolerance,
                        "{} p={p} shifts={de},{ge} field={i}: got={restored} expected={reference}",
                        case["name"]
                    );
                    comparisons += 1;
                }
            }
        }
    }
    assert_eq!(comparisons, 6300);
}

#[test]
fn invalid_projection_domains_are_errors_and_zero_deviation_is_valid() {
    let p = 128;
    let us = vec![f(p, "1"), f(p, "2")];
    let d = vec![f(p, "1"), f(p, "2")];
    let g = vec![f(p, "1"), f(p, "1")];
    for metric in [
        DeviationMetric::FactorWeighted,
        DeviationMetric::IntegrandWeighted,
    ] {
        for precision in [0, 1, 31, 1_000_001] {
            assert!(project(&us, &d, &g, metric, precision).is_err());
        }
        for bad in [
            vec![f(p, "0"), f(p, "1")],
            vec![f(p, "-2"), f(p, "-1")],
            vec![f(p, "0.5"), f(p, "1")],
            vec![f(p, "1"), f(p, "1")],
            vec![f(p, "2"), f(p, "1")],
        ] {
            assert!(project(&bad, &d, &g, metric, p).is_err());
        }
        assert!(project(&us, &d[..1], &g, metric, p).is_err());
        assert!(project(&us[..1], &d[..1], &g[..1], metric, p).is_err());
        let zero = vec![Float::with_val(p, 0); 2];
        assert!(project(&us, &d, &zero, metric, p).is_err());
        let got = project(&us, &zero, &g, metric, p).unwrap();
        assert!(
            got.amplitude.is_zero()
                && got.deviation_norm.is_zero()
                && got.residual_norm.is_zero()
                && got.relative_residual.is_zero()
        );
        assert!(got.reference_norm > 0);
        for bad in [
            rug::float::Special::Nan,
            rug::float::Special::Infinity,
            rug::float::Special::NegInfinity,
        ] {
            let mut samples = d.clone();
            samples[0] = Float::with_val(p, bad);
            assert!(project(&us, &samples, &g, metric, p).is_err());
            assert!(project(&us, &d, &samples, metric, p).is_err());
            assert!(project(&samples, &d, &g, metric, p).is_err());
        }
    }
}

#[test]
fn original_range_and_mixed_precision_counterexamples_are_resolved() {
    let p = 128;
    let us = vec![f(p, "1"), f(p, "2")];
    let g = vec![f(p, "1"), f(p, "1")];
    for metric in [
        DeviationMetric::FactorWeighted,
        DeviationMetric::IntegrandWeighted,
    ] {
        let tiny = vec![f(p, "1e-200000000"), f(p, "-1e-200000000")];
        let got = project(&us, &tiny, &g, metric, p).unwrap();
        assert!(got.deviation_norm > 0 && got.residual_norm > 0 && got.relative_residual > 0.9);
        let large = vec![f(p, "1e200000000"), f(p, "1e200000000")];
        let d = vec![f(p, "1"), f(p, "2")];
        let got = project(&us, &d, &large, metric, p).unwrap();
        assert!(got.amplitude > 0 && got.reference_norm.is_finite());
        let mixed = vec![
            Float::with_val(2, 3),
            f(512, "4.000000000000000000000000000000000000000001"),
        ];
        let got = project(&us, &mixed, &g, metric, 64).unwrap();
        let high = project(&us, &mixed, &g, metric, 512).unwrap();
        assert_eq!(got.amplitude, Float::with_val(64, &high.amplitude));
        let lost = vec![f(p, "1e200000000"), f(p, "1e-200000000")];
        assert!(project(&us, &lost, &g, metric, p).is_err());
    }
}

#[test]
fn grid_rescaling_obeys_each_discrete_metric_homogeneity() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/projection_oracle.json")).unwrap();
    let p = 128;
    let mut comparisons = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let read = |name: &str| {
            case[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f(p, v.as_str().unwrap()))
                .collect::<Vec<_>>()
        };
        let us = read("us");
        let d = read("deviation");
        let g = read("reference");
        let factor = case["metric"] == "factor";
        let metric = if factor {
            DeviationMetric::FactorWeighted
        } else {
            DeviationMetric::IntegrandWeighted
        };
        for exponent in [4, 400_000_000, 800_000_000] {
            let got = project(&shifts(&us, exponent, 1), &d, &g, metric, p).unwrap();
            let actual = [
                got.amplitude,
                got.deviation_norm,
                got.reference_norm,
                got.residual_norm,
                got.relative_residual,
            ];
            for (i, got) in actual.iter().enumerate() {
                let expected = f(768, case["expected"][i].as_str().unwrap());
                let mut value = Float::with_val(768, got);
                if !factor && (1..=3).contains(&i) {
                    value >>= exponent / 4;
                }
                let error = Float::with_val(768, &value - &expected).abs();
                let tolerance = expected.abs().max(&Float::with_val(768, 1))
                    * (Float::with_val(768, 1) >> (p - 10));
                assert!(
                    error < tolerance,
                    "{} grid scale={exponent} field={i}",
                    case["name"]
                );
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 900);
}
