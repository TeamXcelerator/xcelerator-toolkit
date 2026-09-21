//! Checkout-independent, short fixture paths for Git's Windows path budget.
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) fn temporary_root(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let label = crate::ContentDigest::sha256(name.as_bytes());
    std::env::temp_dir().join("xc-test").join(format!(
        "{}-{}-{}",
        &label.0[..8],
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn fixtures_are_short_unique_and_outside_checkout() {
    let name = "long-checkout-independent-name".repeat(30);
    let a = temporary_root(&name);
    let b = temporary_root(&name);
    assert_ne!(a, b);
    assert_eq!(a.parent().unwrap(), std::env::temp_dir().join("xc-test"));
    assert!(a.file_name().unwrap().len() < 40);
}
