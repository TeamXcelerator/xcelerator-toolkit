//! Checkout-independent, short fixture paths for Git's Windows path budget.
pub(crate) use xc_core::test_support::TestDir;

/// A fresh scratch directory removed when the guard is dropped. The label is
/// hashed so nested Git fixtures stay within the Windows path budget.
pub(crate) fn temporary_root(name: &str) -> TestDir {
    let label = crate::ContentDigest::sha256(name.as_bytes());
    TestDir::new(&label.0[..8])
}

#[test]
fn fixtures_are_short_unique_and_outside_checkout() {
    let name = "long-checkout-independent-name".repeat(30);
    let a = temporary_root(&name);
    let b = temporary_root(&name);
    assert_ne!(a.path(), b.path());
    assert_eq!(a.parent().unwrap(), xc_core::test_support::test_root());
    assert!(a.file_name().unwrap().len() < 40);
    let path = a.to_path_buf();
    drop(a);
    assert!(!path.exists());
}
