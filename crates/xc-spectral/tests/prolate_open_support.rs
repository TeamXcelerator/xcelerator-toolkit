use xc_spectral::prolate::{compute_k_lambda_f64, ProlateConfig};

fn cases() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/prolate_open_support.json")).unwrap()
}

fn native_cutoff(text: &str) -> f64 {
    if let Some((a, b)) = text.split_once('/') {
        a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap()
    } else {
        text.parse().unwrap()
    }
}

#[test]
fn native_samples_include_pinned_and_interior_independent_oracles() {
    for case in cases()["cases"].as_array().unwrap() {
        let cutoff = native_cutoff(case["lambda_squared"].as_str().unwrap());
        let samples = case["n_sample"].as_u64().unwrap() as usize;
        let actual =
            compute_k_lambda_f64(&ProlateConfig::new(cutoff.sqrt(), 257).with_n_sample(samples))
                .unwrap();
        for (i, (value, reference)) in actual
            .k_values
            .iter()
            .zip(case["k_values"].as_array().unwrap())
            .enumerate()
        {
            let expected: f64 = reference.as_str().unwrap().parse().unwrap();
            assert!(
                (value - expected).abs() < 3e-13,
                "cutoff={cutoff} sample={i}: actual={value:e}, independent={expected:e}"
            );
        }
        assert_eq!(actual.k_values[samples - 1], 0.0);
    }
}

#[cfg(feature = "hp")]
mod hp {
    use super::*;
    use rug::Float;
    use xc_numerics::quadrature::CacheMode;
    use xc_spectral::ccm::{hp::HighPrecConfig, CcmParams};
    use xc_spectral::prolate::hp::{ccm_prolate_distance_hp, compute_k_lambda};

    fn cutoff(text: &str, p: u32) -> Float {
        if let Some((a, b)) = text.split_once('/') {
            Float::with_val(p, Float::parse(a).unwrap())
                / Float::with_val(p, Float::parse(b).unwrap())
        } else {
            Float::with_val(p, Float::parse(text).unwrap())
        }
    }

    #[test]
    fn precision_ladder_obeys_independent_exact_support_counts_and_series_values() {
        for case in cases()["cases"].as_array().unwrap() {
            let text = case["lambda_squared"].as_str().unwrap();
            let samples = case["n_sample"].as_u64().unwrap() as usize;
            for p in [255u32, 256, 257, 258, 512] {
                let lambda = cutoff(text, p).sqrt();
                let actual = compute_k_lambda(&lambda, 257, samples, p, CacheMode::Off).unwrap();
                let tolerance = Float::with_val(p, 1) >> (p - 9);
                for (i, (value, reference)) in actual
                    .k_values
                    .iter()
                    .zip(case["k_values"].as_array().unwrap())
                    .enumerate()
                {
                    let expected =
                        Float::with_val(p, Float::parse(reference.as_str().unwrap()).unwrap());
                    let error = Float::with_val(p, value - expected).abs();
                    assert!(
                        error < tolerance,
                        "cutoff={text} p={p} sample={i}: error={error:e}"
                    );
                }
                assert_eq!(actual.k_values[samples - 1], 0);
            }
        }
    }

    #[test]
    fn end_to_end_distance_is_stable_across_the_previously_flipping_precisions() {
        for (cutoff, n) in [(2u64, 4usize), (3, 5)] {
            let params = CcmParams::from_lambda_sq_integer(cutoff, n);
            let mut results = Vec::new();
            for p in [256u32, 257, 320, 512] {
                let mut cfg = HighPrecConfig::for_decimal_digits(60);
                cfg.precision_bits = p;
                cfg.cache_mode = CacheMode::Off;
                results.push(
                    ccm_prolate_distance_hp(&params, &cfg, 257, 64, CacheMode::Off)
                        .unwrap()
                        .relative_l2_distance,
                );
            }
            eprintln!(
                "cutoff={cutoff} N={n}: p256={}, p257={}, p320={}, p512={}",
                results[0], results[1], results[2], results[3]
            );
            assert!(
                Float::with_val(320, &results[0] - &results[1]).abs()
                    < (Float::with_val(320, 1) >> 250),
                "cutoff={cutoff} N={n}: p256={} p257={} difference={}",
                results[0],
                results[1],
                Float::with_val(320, &results[0] - &results[1]).abs()
            );
        }
    }
}
