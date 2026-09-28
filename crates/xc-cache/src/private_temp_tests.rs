use super::*;
use std::sync::{Arc, Barrier};

#[test]
fn private_staging_keeps_two_open_writers_isolated() {
    let root = crate::test_support::temporary_root("private-two-open-writers");
    fs::create_dir_all(&root).unwrap();
    let (first_path, mut first) = create_private_sibling_file(&root, "tmp").unwrap();
    let (second_path, mut second) = create_private_sibling_file(&root, "tmp").unwrap();
    assert_ne!(first_path, second_path);
    first.write_all(b"A").unwrap();
    first.sync_all().unwrap();
    drop(first);
    let a = root.join("A");
    let b = root.join("B");
    fs::rename(&first_path, &a).unwrap();
    // B's already-open handle must not alias A's published file.
    second.write_all(b"B").unwrap();
    second.sync_all().unwrap();
    drop(second);
    fs::rename(&second_path, &b).unwrap();
    assert_eq!(fs::read(a).unwrap(), b"A");
    assert_eq!(fs::read(b).unwrap(), b"B");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_atomic_writes_keep_each_canonical_payload() {
    let root = crate::test_support::temporary_root("private-concurrent-atomic-writes");
    fs::create_dir_all(&root).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    std::thread::scope(|scope| {
        for worker in 0..8u8 {
            let root = &root;
            let barrier = barrier.clone();
            scope.spawn(move || {
                barrier.wait();
                for index in 0..8u8 {
                    let bytes = vec![worker * 8 + index; 4096];
                    atomic_write(&root.join(format!("payload-{worker}-{index}")), &bytes).unwrap();
                }
            });
        }
    });
    for worker in 0..8u8 {
        for index in 0..8u8 {
            assert_eq!(
                fs::read(root.join(format!("payload-{worker}-{index}"))).unwrap(),
                vec![worker * 8 + index; 4096]
            );
        }
    }
    assert_eq!(fs::read_dir(&root).unwrap().count(), 64);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_atomic_replacements_use_private_staging() {
    let root = crate::test_support::temporary_root("private-concurrent-atomic-replaces");
    fs::create_dir_all(&root).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    std::thread::scope(|scope| {
        for worker in 0..8u8 {
            let root = &root;
            let barrier = barrier.clone();
            scope.spawn(move || {
                let path = root.join(format!("index-{worker}"));
                fs::write(&path, b"initial").unwrap();
                barrier.wait();
                for index in 0..8u8 {
                    atomic_replace(&path, &vec![worker * 8 + index; 4096]).unwrap();
                }
            });
        }
    });
    for worker in 0..8u8 {
        assert_eq!(
            fs::read(root.join(format!("index-{worker}"))).unwrap(),
            vec![worker * 8 + 7; 4096]
        );
    }
    assert_eq!(fs::read_dir(&root).unwrap().count(), 8);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_staging_parent_preserves_existing_bytes() {
    let root = crate::test_support::temporary_root("private-invalid-staging-parent");
    fs::create_dir_all(&root).unwrap();
    let blocker = root.join("file");
    fs::write(&blocker, b"sentinel").unwrap();
    assert!(create_private_sibling_file(&blocker, "tmp").is_err());
    assert!(atomic_write(&blocker.join("child"), b"new").is_err());
    assert!(atomic_replace(&blocker.join("child"), b"new").is_err());
    assert_eq!(fs::read(&blocker).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_encoded_adoption_reserves_distinct_owned_links() {
    let root = crate::test_support::temporary_root("private-concurrent-adoption");
    fs::create_dir_all(&root).unwrap();
    let source = root.join("source");
    fs::write(&source, b"verified bytes").unwrap();
    let barrier = Arc::new(Barrier::new(16));
    let paths = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let source = &source;
                let root = &root;
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    stage_encoded_object(source, root).unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let distinct: std::collections::HashSet<_> = paths.iter().collect();
    assert_eq!(distinct.len(), 16);
    for path in paths {
        assert_eq!(fs::read(&path).unwrap(), b"verified bytes");
        fs::remove_file(path).unwrap();
    }
    assert_eq!(fs::read(&source).unwrap(), b"verified bytes");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_encoded_staging_does_not_remove_other_owned_files() {
    let root = crate::test_support::temporary_root("private-failed-adoption");
    fs::create_dir_all(&root).unwrap();
    let source = root.join("source");
    fs::write(&source, b"sentinel").unwrap();
    let owned = stage_encoded_object(&source, &root).unwrap();
    assert!(stage_encoded_object(&root.join("missing"), &root).is_err());
    assert_eq!(fs::read(&owned).unwrap(), b"sentinel");
    assert_eq!(fs::read(&source).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
    fs::remove_dir_all(root).unwrap();
}
