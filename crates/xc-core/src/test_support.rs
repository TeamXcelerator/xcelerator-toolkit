// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Self-removing scratch directories for tests.
//!
//! Every toolkit test that needs files on disk uses [`TestDir`]. Directories
//! are created under `<OS temp>/xc-test/`, never inside a checkout, and are
//! removed when the guard is dropped, including when a test panics. Short
//! names keep nested cache layouts within Windows path limits. Directories
//! left behind by an aborted process are swept on the next run once they are
//! older than [`STALE_AFTER`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;
use std::time::Duration;

/// Age after which an abandoned scratch directory is removed by a later run.
pub const STALE_AFTER: Duration = Duration::from_secs(24 * 60 * 60);

/// Shared parent of all test scratch directories.
pub fn test_root() -> PathBuf {
    std::env::temp_dir().join("xc-test")
}

/// A unique scratch directory that is deleted when dropped.
#[derive(Debug)]
pub struct TestDir {
    path: PathBuf,
}

impl TestDir {
    /// Creates a fresh, empty directory. `tag` is a short label; only its
    /// first 24 path-safe characters are kept.
    pub fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        static SWEEP: Once = Once::new();
        let root = test_root();
        SWEEP.call_once(|| sweep_stale(&root));
        let label: String = tag
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .take(24)
            .collect();
        let path = root.join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test scratch directory");
        Self { path }
    }

    /// The directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::ops::Deref for TestDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TestDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn sweep_stale(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > STALE_AFTER);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_is_unique_outside_checkout_and_removed_on_drop() {
        let a = TestDir::new("unique/check");
        let b = TestDir::new("unique/check");
        assert_ne!(a.path(), b.path());
        assert!(a.is_dir() && b.is_dir());
        assert!(a.starts_with(test_root()));
        assert!(!a.starts_with(env!("CARGO_MANIFEST_DIR")));
        std::fs::create_dir_all(a.join("nested")).unwrap();
        std::fs::write(a.join("nested").join("file"), b"x").unwrap();
        let path = a.path().to_path_buf();
        drop(a);
        assert!(!path.exists());
    }

    #[test]
    fn directory_is_removed_when_a_test_panics() {
        let path = std::panic::catch_unwind(|| {
            let dir = TestDir::new("panic");
            let path = dir.path().to_path_buf();
            std::fs::write(dir.join("file"), b"x").unwrap();
            std::panic::panic_any(path)
        })
        .unwrap_err()
        .downcast::<PathBuf>()
        .unwrap();
        assert!(!path.exists());
    }
}
