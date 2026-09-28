use super::*;
#[test]
fn native_discovery_resolves_high_ordinate_bracket() {
    // R=1/(x-500)+2/(x-700) has exact root 1700/3.
    let s = SecularFunctionF64::new(vec![500., 700.], vec![1., 2.]).unwrap();
    let roots = discover_roots_f64(&s, 501., 699., &DiscoveryOptionsF64::default()).unwrap();
    assert_eq!(roots.len(), 1);
    let actual: f64 = roots[0].midpoint.parse().unwrap();
    assert!((actual - 1700. / 3.).abs() < 1e-10);
    assert!(window_math::within_scaled_distance(512., 512f64.next_up(), 1e-13).unwrap());
    assert!(!window_math::within_distance(512., 512f64.next_up(), 1e-13, 1).unwrap());
}
#[test]
fn capture_builder_supports_current_lower_levels_only() {
    use crate::ccm::capture::{CcmCaptureLevel, CcmCapturePlan};
    let plan = CcmCapturePlan::resolve(CcmCaptureLevel::Research, 2, 3).unwrap();
    assert!(
        plan.clone()
            .with_reference_projection()
            .unwrap()
            .capture_reference_projection
    );
    let mut historical = plan;
    historical.semantics = "ccm-measurement-capture-plan-v1".into();
    historical.validate().unwrap();
    assert!(historical.with_reference_projection().is_err());
}
#[test]
fn cutoff_filename_is_canonical() {
    use crate::ccm::LambdaSq;
    for value in [1.01, 5.0, 13.5] {
        let cutoff = LambdaSq::fractional(value);
        assert_eq!(
            LambdaSq::from_filename_str(&cutoff.filename_str())
                .unwrap()
                .filename_str(),
            cutoff.filename_str()
        );
    }
    assert!(LambdaSq::from_filename_str("5d0").is_none());
    assert!(LambdaSq::from_filename_str("0005").is_none());
}

#[test]
fn plan_accepts_one_ulp_platform_height_but_rejects_material_change() {
    let request = CcmFirstKPlanningRequest {
        requested_roots: 50,
        target_uniform_digits: 20,
        precision_guard_digits: 10,
        minimum_reach_margin_modes: 10,
    };
    let candidate = CcmPlannerCandidate {
        candidate_id: "portable-height".into(),
        lambda_squared: "5".into(),
        n_modes: 120,
        precision_bits: 256,
        calibrated_root_count: 100,
        calibrated_uniform_digits: 30,
        calibration_evidence_digest: "a".repeat(64),
    };
    let mut plan = plan_first_k_ccm_observation(request, &[candidate]).unwrap();
    let height: f64 = plan.estimated_height.parse().unwrap();
    plan.estimated_height = height.next_up().to_string();
    validate_first_k_plan(&plan).unwrap();
    plan.estimated_height = (height * 1.0001).to_string();
    assert!(validate_first_k_plan(&plan).is_err());
}
