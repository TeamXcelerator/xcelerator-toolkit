use std::env;
use std::path::Path;
use std::process::Command;

fn valid_git_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn git_revision(manifest_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD^{commit}"])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let revision = String::from_utf8(output.stdout).ok()?;
    let revision = revision.trim().to_ascii_lowercase();
    valid_git_revision(&revision).then_some(revision)
}

fn git_path(manifest_dir: &Path, name: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", name])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn symbolic_head(manifest_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=XC_SOURCE_REVISION");

    let manifest_dir = std::path::PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    if let Some(head_path) = git_path(&manifest_dir, "HEAD") {
        println!("cargo:rerun-if-changed={head_path}");
    }
    if let Some(head_ref) = symbolic_head(&manifest_dir) {
        if let Some(ref_path) = git_path(&manifest_dir, &head_ref) {
            println!("cargo:rerun-if-changed={ref_path}");
        }
    }

    let explicit = env::var("XC_SOURCE_REVISION").ok();
    let revision = match explicit {
        Some(revision) => {
            let revision = revision.trim().to_ascii_lowercase();
            assert!(
                valid_git_revision(&revision),
                "XC_SOURCE_REVISION must be a full 40-character lowercase hexadecimal Git commit"
            );
            Some(revision)
        }
        None => git_revision(&manifest_dir),
    };

    if let Some(revision) = revision {
        println!("cargo:rustc-env=XC_SOURCE_REVISION={revision}");
    }
}
