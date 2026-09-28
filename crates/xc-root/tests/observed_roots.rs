use xc_root::{
    discover_pole_aware_sign_changes_f64, MeromorphicFunctionF64, PoleAwareDiscoveryOptionsF64,
    RealFunctionF64, RootError,
};

struct EndpointRoots;
impl RealFunctionF64 for EndpointRoots {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(x * (x - 1.0))
    }
}
impl MeromorphicFunctionF64 for EndpointRoots {
    fn real_poles(&self) -> &[f64] {
        &[]
    }
}

#[test]
fn discovery_must_retain_both_exactly_observed_endpoint_roots() {
    let options = PoleAwareDiscoveryOptionsF64 {
        subdivisions_per_interval: 1,
        ..Default::default()
    };
    let roots = discover_pole_aware_sign_changes_f64(&EndpointRoots, 0.0, 1.0, &options).unwrap();
    let points: Vec<_> = roots.iter().map(|r| r.midpoint).collect();
    eprintln!("observed endpoint roots: {points:?}");
    assert_eq!(
        points,
        vec![0.0, 1.0],
        "discovery discarded an exactly sampled root"
    );
}

struct ThreeSampledRoots;
impl RealFunctionF64 for ThreeSampledRoots {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(x * (x - 0.5) * (x - 1.0))
    }
}
impl MeromorphicFunctionF64 for ThreeSampledRoots {
    fn real_poles(&self) -> &[f64] {
        &[]
    }
}
#[test]
fn adjacent_sampled_zeros_are_retained_once_each() {
    for subdivisions in [2, 4, 8] {
        let options = PoleAwareDiscoveryOptionsF64 {
            subdivisions_per_interval: subdivisions,
            ..Default::default()
        };
        let roots =
            discover_pole_aware_sign_changes_f64(&ThreeSampledRoots, 0.0, 1.0, &options).unwrap();
        assert_eq!(
            roots.iter().map(|root| root.midpoint).collect::<Vec<_>>(),
            vec![0.0, 0.5, 1.0]
        );
        assert!(roots
            .iter()
            .all(|root| root.bracket.lower == root.bracket.upper && root.residual == 0.0));
    }
}
