use std::path::Path;
use std::process::Command;
use xc_cache::{GitCliRemoteStore, RemoteGitStore};
use xc_core::CancellationToken;

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "local Git fixture: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[test]
fn ordinary_canonical_and_owned_reader_paths_recover_exact_committed_bytes() {
    // Exclusively created test directory, never a caller-owned cache.
    let scratch = xc_core::test_support::TestDir::new("r2-git");
    let fixture = scratch.join("fixture");
    std::fs::create_dir(&fixture).unwrap();
    let source = fixture.join("source");
    std::fs::create_dir(&source).unwrap();
    git(&source, &["init", "-q"]);
    let payload = b"{\"value\":\"0.25\"}\n\0\xff";
    std::fs::write(source.join("payload.bin"), payload).unwrap();
    git(&source, &["config", "uploadpack.allowFilter", "true"]);
    git(
        &source,
        &["config", "uploadpack.allowAnySHA1InWant", "true"],
    );
    git(&source, &["add", "payload.bin"]);
    git(
        &source,
        &[
            "-c",
            "user.name=TeamXceleratorDev",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "Add manufactured payload",
        ],
    );
    let revision = git(&source, &["rev-parse", "HEAD"]);
    let canonical = std::fs::canonicalize(&fixture).unwrap();
    for (label, parent, owned) in [
        ("ordinary", &fixture, false),
        ("canonical", &canonical, false),
        ("Unicode space \u{03b1}", &canonical, false),
        ("owned canonical", &canonical, true),
    ] {
        let root = parent.join(label);
        let store = if owned {
            GitCliRemoteStore::new_read_session(
                root.join("git"),
                root.join("parts"),
                "TeamXceleratorDev",
                "fixture@example.invalid",
            )
        } else {
            GitCliRemoteStore::new(
                root.join("git"),
                root.join("parts"),
                "TeamXceleratorDev",
                "fixture@example.invalid",
            )
        }
        .unwrap();
        let mut bytes = Vec::new();
        let report = store
            .read_committed_path(
                source.to_str().unwrap(),
                &revision,
                "payload.bin",
                4096,
                &CancellationToken::new(),
                &mut bytes,
            )
            .unwrap();
        assert_eq!(bytes, payload, "{label}");
        assert_eq!(report.size_bytes, payload.len() as u64);
        if owned {
            store.finish_read_session().unwrap();
        }
    }
}
