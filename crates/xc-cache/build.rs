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

// Content identity supplements the Git base revision, including uncommitted source.
fn source_tree_digest(manifest_dir: &Path) {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};

    fn walk(path: &Path, files: &mut Vec<PathBuf>, in_data: bool) {
        println!("cargo:rerun-if-changed={}", path.display());
        for entry in std::fs::read_dir(path).expect("source input directory") {
            let entry = entry.expect("source input entry");
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".private.json") {
                continue;
            }
            let kind = entry.file_type().expect("source input type");
            assert!(
                !kind.is_symlink(),
                "source input links need explicit admission"
            );
            if kind.is_dir() {
                if in_data && name.ends_with("_cache") {
                    continue;
                }
                walk(&path, files, in_data || name == "data");
            } else if kind.is_file() {
                files.push(path);
            }
        }
    }

    let root = manifest_dir.join("../..");
    let mut files = vec![root.join("Cargo.toml"), root.join("Cargo.lock")];
    let crates = root.join("crates");
    // Cargo's directory watches are recursive. They discover newly added source
    // inputs; runtime cache activity can rerun this script but cannot change its
    // digest. Runtime data are excluded from the content walk at every depth.
    println!("cargo:rerun-if-changed={}", crates.display());
    for entry in std::fs::read_dir(&crates).expect("workspace crates") {
        let entry = entry.expect("workspace crate entry");
        if !entry.file_type().expect("workspace crate type").is_dir() {
            continue;
        }
        let path = entry.path();
        for name in ["Cargo.toml", "build.rs"] {
            let file = path.join(name);
            if file.is_file() {
                files.push(file);
            }
        }
        for section in ["src", "examples", "tests", "benches", "data"] {
            let input = path.join(section);
            if input.is_dir() {
                walk(&input, &mut files, section == "data");
            }
        }
    }
    for name in [".cargo/config.toml", "rust-toolchain.toml"] {
        let path = root.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    files.dedup();
    let mut h = Sha256::new();
    h.update(b"xcelerator-workspace-source-byte-inputs-v2\0");
    h.update((files.len() as u64).to_le_bytes());
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .strip_prefix(&root)
            .expect("relative source input")
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::read(&path).expect("source input bytes");
        // Byte identity also covers include_str!/include_bytes! inputs, even
        // when their extension is .rs. Length framing is unambiguous for binary data.
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }

    println!("cargo:rustc-env=XC_SOURCE_TREE_DIGEST={:x}", h.finalize());
}

fn main() {
    println!("cargo:rerun-if-env-changed=XC_SOURCE_REVISION");

    let manifest_dir = std::path::PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    );
    source_tree_digest(&manifest_dir);
    if let Some(packed_refs) = git_path(&manifest_dir, "packed-refs") {
        println!("cargo:rerun-if-changed={packed_refs}");
    }
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
