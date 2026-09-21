fn main() {
    checkpoint_build_id();
    println!("cargo:rerun-if-changed=src/ccm/arb_bridge.c");
    if std::env::var_os("CARGO_FEATURE_ARB").is_none() {
        return;
    }

    cc::Build::new()
        .file("src/ccm/arb_bridge.c")
        .warnings(true)
        .compile("xc_spectral_arb_bridge");
    println!("cargo:rustc-link-lib=dylib=flint");
}

// Invalidate local execution state across releases AND amendments of one release.
fn checkpoint_build_id() {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};
    fn walk(path: &Path, files: &mut Vec<PathBuf>) {
        println!("cargo:rerun-if-changed={}", path.display());
        for entry in std::fs::read_dir(path).expect("checkpoint source directory") {
            let entry = entry.expect("checkpoint source entry");
            let p = entry.path();
            if entry.file_type().expect("source type").is_dir() {
                walk(&p, files);
            } else if matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("rs" | "c" | "h" | "txt")
            ) {
                files.push(p);
            }
        }
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let mut files = vec![root.join("Cargo.toml"), root.join("Cargo.lock")];
    for entry in std::fs::read_dir(root.join("crates")).expect("workspace crates") {
        let path = entry.unwrap().path();
        if path.join("src").is_dir() {
            walk(&path.join("src"), &mut files);
            files.push(path.join("Cargo.toml"));
            if path.join("build.rs").exists() {
                files.push(path.join("build.rs"));
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    for p in files {
        println!("cargo:rerun-if-changed={}", p.display());
        h.update(
            p.strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
                .as_bytes(),
        );
        h.update([0]);
        h.update(
            std::fs::read_to_string(&p)
                .expect("checkpoint source")
                .replace("\r\n", "\n")
                .as_bytes(),
        );
        h.update([0]);
    }
    h.update(std::env::var("TARGET").unwrap());
    println!("cargo:rustc-env=XC_CHECKPOINT_BUILD_ID={:x}", h.finalize());
}
