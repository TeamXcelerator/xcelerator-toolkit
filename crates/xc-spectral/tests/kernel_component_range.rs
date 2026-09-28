#[test]
fn representable_complex_values_survive_negligible_component_underflow() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/kernel_component_range.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let values: Vec<f64> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let (real, imaginary) = xc_spectral::yakaboylu::try_v_r_matrix_element_f64(
            values[0], values[1], values[2], values[3], values[4],
        )
        .unwrap();
        let expected_real = case["real"].as_f64().unwrap();
        let expected_imaginary = case["imaginary"].as_f64().unwrap();
        let norm = expected_real.hypot(expected_imaginary);
        let tolerance = (norm * 64.0 * f64::EPSILON * case["condition"].as_f64().unwrap())
            .max(f64::from_bits(1));
        assert!((real-expected_real).hypot(imaginary-expected_imaginary)<=tolerance,"inputs={values:?}: {real:e}+{imaginary:e}i, expected={expected_real:e}+{expected_imaginary:e}i");
    }
    // The whole value lies below binary64 range; this remains an error.
    assert!(
        xc_spectral::yakaboylu::try_v_r_matrix_element_f64(0.5, 0.0, 0.5, 1e300, 1e-300).is_err()
    );
}
