use xc_root::{
    discover_pole_aware_sign_changes_f64, MeromorphicFunctionF64, PoleAwareDiscoveryOptionsF64,
    RealFunctionF64, RootError,
};
struct Linear {
    root: f64,
    poles: Vec<f64>,
}
impl RealFunctionF64 for Linear {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        assert!(!self.poles.contains(&x), "evaluated a declared pole");
        Ok(x - self.root)
    }
}
impl MeromorphicFunctionF64 for Linear {
    fn real_poles(&self) -> &[f64] {
        &self.poles
    }
}
fn find(root: f64, lo: f64, hi: f64, poles: Vec<f64>) {
    let roots = discover_pole_aware_sign_changes_f64(
        &Linear { root, poles },
        lo,
        hi,
        &PoleAwareDiscoveryOptionsF64::default(),
    )
    .unwrap();
    assert_eq!(roots.len(), 1, "{roots:?}");
    assert_eq!(roots[0].midpoint, root);
}
#[test]
fn regular_window_endpoints_are_searchable() {
    find(0.0, 0.0, 1.0, vec![]);
    find(1.0, 0.0, 1.0, vec![]);
}
#[test]
fn narrow_regular_windows_are_not_discarded_as_pole_margins() {
    find(0.0, -1e-100, 1e-100, vec![]);
}
#[test]
fn actual_poles_at_window_boundaries_remain_excluded() {
    find(0.0, -1.0, 1.0, vec![-1.0, 1.0]);
}
