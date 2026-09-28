use std::path::{Path, PathBuf};
use std::process::Command;
use xc_cache::{GitCliRemoteStore, RemoteGitStore};
use xc_core::CancellationToken;

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        // Exclusively created test directory, never a caller-owned cache.
        std::fs::remove_dir_all(&self.0).expect("remove local Git fixture");
    }
}

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
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let fixture =
        Fixture(std::env::temp_dir().join(format!("xc-r2-git-{}-{nanos}", std::process::id())));
    std::fs::create_dir(&fixture.0).unwrap();
    let source = fixture.0.join("source");
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
    let canonical = std::fs::canonicalize(&fixture.0).unwrap();
    for (label, parent, owned) in [
        ("ordinary", &fixture.0, false),
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
