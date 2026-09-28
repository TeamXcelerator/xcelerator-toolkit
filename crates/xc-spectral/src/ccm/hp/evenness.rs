use super::*;

#[test]
fn evenness_accepts_qualified_positive_ground_state() {
    let p = 128;
    let params = CcmParams::from_lambda_sq_integer(2, 1);
    let mut cfg = HighPrecConfig::for_decimal_digits(25);
    cfg.precision_bits = p;
    cfg.inverse_iter_steps = 64;
    // Same exact parity basis: even eigenvalues1,3, odd eigenvalue10.
    let matrix = [13, 0, -7, 0, 2, 0, -7, 0, 13]
        .into_iter()
        .map(|n| Float::with_val(p, n) / 2u32)
        .collect::<Vec<_>>();
    // Preserve the original work budget as a refusal control for a partial candidate.
    let error = measure_evenness_from_tau(&params, &cfg, matrix.clone())
        .err()
        .expect("64 steps do not reach the residual floor");
    assert!(error.to_string().contains("work budget"), "{error:#}");
    cfg.inverse_iter_steps = 256;
    let result = measure_evenness_from_tau(&params, &cfg, matrix).unwrap();
    let tolerance = Float::with_val(p, 1) >> 90u32;
    assert!((result.natural_eigenvalue - 1u32).abs() < tolerance);
    assert!((result.forced_eigenvalue - 1u32).abs() < tolerance);
    assert!(result.evenness_deviation < tolerance);
}

#[test]
fn evenness_must_qualify_algebraic_ground_state() {
    let p = 128;
    let params = CcmParams::from_lambda_sq_integer(2, 1);
    let mut cfg = HighPrecConfig::for_decimal_digits(25);
    cfg.precision_bits = p;
    cfg.inverse_iter_steps = 64;
    // Exact manufactured centrosymmetric source: (1,0,-1) has eigenvalue -10,
    // (0,1,0) has eigenvalue 1, and (1,0,1) has eigenvalue 3.
    let matrix = [-7, 0, 13, 0, 2, 0, 13, 0, -7]
        .into_iter()
        .map(|n| Float::with_val(p, n) / 2u32)
        .collect::<Vec<_>>();
    let error = measure_evenness_from_tau(&params, &cfg, matrix.clone())
        .err()
        .expect("64 steps do not qualify a ground state");
    assert!(error.to_string().contains("work budget"), "{error:#}");
    cfg.inverse_iter_steps = 256;
    match measure_evenness_from_tau(&params, &cfg, matrix) {
        Ok(result) => {
            eprintln!(
                "manufactured source: reported natural={}, forced={}, deviation={}",
                result.natural_eigenvalue, result.forced_eigenvalue, result.evenness_deviation
            );
            assert!(result.natural_eigenvalue < 0,
                "near-zero inverse iteration must not claim the positive eigenvalue is the algebraic ground state");
        }
        Err(error) => assert!(error.to_string().contains("ground"), "{error:#}"),
    }
}
