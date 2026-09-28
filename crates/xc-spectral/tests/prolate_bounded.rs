use xc_spectral::prolate::{compute_k_lambda_f64, ProlateConfig};

fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/prolate_bounded_oracle.json")).unwrap()
}

#[test]
fn bounded_native_preflights_output_and_sampling_work() {
    let output =
        compute_k_lambda_f64(&ProlateConfig::new(2.0, 257).with_n_sample(u32::MAX as usize))
            .unwrap_err();
    assert!(output.to_string().contains("combined output"));
    let work = compute_k_lambda_f64(&ProlateConfig::new(999.0, 257)).unwrap_err();
    assert!(work.to_string().contains("total polynomial sampling work"));
}

#[cfg(feature = "hp")]
#[test]
fn bounded_hp_preflights_output_and_sampling_work() {
    use rug::Float;
    use xc_numerics::quadrature::CacheMode;
    use xc_spectral::prolate::hp::compute_k_lambda;
    let output = compute_k_lambda(
        &Float::with_val(128, 2),
        257,
        u32::MAX as usize,
        128,
        CacheMode::Off,
    )
    .unwrap_err();
    assert!(output.to_string().contains("combined output"));
    let work =
        compute_k_lambda(&Float::with_val(128, 999), 257, 256, 128, CacheMode::Off).unwrap_err();
    assert!(work.to_string().contains("total polynomial sampling work"));
}

#[test]
fn bounded_native_matches_independent_continuum_oracle() {
    for case in oracle()["cases"].as_array().unwrap() {
        let cutoff: f64 = case["lambda_squared"].as_str().unwrap().parse().unwrap();
        let result =
            compute_k_lambda_f64(&ProlateConfig::new(cutoff.sqrt(), 257).with_n_sample(9)).unwrap();
        assert_eq!(result.discretization, "prolate-bounded-legendre-even-v2");
        assert!(result.relative_operator_residual.unwrap() <= 2f64.powi(-39));
        for (value, key) in [
            (result.eigenvalue_0, "eigenvalue_0"),
            (result.eigenvalue_4, "eigenvalue_4"),
            (result.c_0, "c_0"),
        ] {
            let reference: f64 = case[key].as_str().unwrap().parse().unwrap();
            assert!(
                (value - reference).abs() < 2e-10 * (1.0 + reference.abs()),
                "c={cutoff} {key}: got={value} expected={reference}"
            );
        }
        for (value, reference) in result.k_values[1..8]
            .iter()
            .zip(case["interior_k_samples_of_nine"].as_array().unwrap())
        {
            let reference: f64 = reference.as_str().unwrap().parse().unwrap();
            assert!(
                (value - reference).abs() < 2e-9,
                "c={cutoff} sample got={value} expected={reference}"
            );
        }
        assert_eq!(result.k_values[8], 0.0);
    }
    assert!(
        compute_k_lambda_f64(&ProlateConfig::new(13f64.sqrt(), 17)).is_err(),
        "unresolved truncation must fail without FD fallback"
    );
}

#[cfg(feature = "hp")]
#[test]
fn bounded_hp_matches_oracle_and_refines() {
    use rug::Float;
    use xc_numerics::quadrature::CacheMode;
    use xc_spectral::prolate::hp::compute_k_lambda;
    for case in oracle()["cases"].as_array().unwrap() {
        let mut previous = None;
        for p in [128, 192] {
            let lambda = Float::with_val(
                p,
                Float::parse(case["lambda_squared"].as_str().unwrap()).unwrap(),
            )
            .sqrt();
            let result = compute_k_lambda(&lambda, 257, 9, p, CacheMode::Off).unwrap();
            assert_eq!(result.discretization, "prolate-bounded-legendre-even-v2");
            assert!(
                result.relative_operator_residual.as_ref().unwrap()
                    <= &(Float::with_val(p, 1) >> (p - 12))
            );
            let tolerance = Float::with_val(p, 1) >> 100;
            for (value, key) in [
                (&result.eigenvalue_0, "eigenvalue_0"),
                (&result.eigenvalue_4, "eigenvalue_4"),
                (&result.c_0, "c_0"),
            ] {
                let reference =
                    Float::with_val(p, Float::parse(case[key].as_str().unwrap()).unwrap());
                assert!(
                    Float::with_val(p, value - &reference).abs()
                        < Float::with_val(
                            p,
                            &tolerance * (Float::with_val(p, &reference).abs() + 1)
                        ),
                    "p={p} {key}"
                );
            }
            for (value, reference) in result.k_values[1..8]
                .iter()
                .zip(case["interior_k_samples_of_nine"].as_array().unwrap())
            {
                let reference =
                    Float::with_val(p, Float::parse(reference.as_str().unwrap()).unwrap());
                assert!(
                    Float::with_val(p, value - reference).abs() < tolerance,
                    "p={p} sample"
                );
            }
            if let Some(previous) = previous {
                assert!(Float::with_val(p, &result.c_0 - previous).abs() < tolerance);
            }
            previous = Some(result.c_0);
        }
    }
}
