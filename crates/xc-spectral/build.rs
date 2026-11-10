fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() {
    checkpoint_build_id();
    println!("cargo:rerun-if-changed=src/ccm/arb_bridge.c");
    if std::env::var_os("CARGO_FEATURE_ARB").is_none() {
        return;
    }

    let mut bridge = cc::Build::new();
    configure_native_arb(&mut bridge);
    bridge
        .file("src/ccm/arb_bridge.c")
        .warnings(true)
        .compile("xc_spectral_arb_bridge");
    println!("cargo:rustc-link-lib=dylib=flint");
}

// Keep this prefix contract aligned with xc-numerics: its executed MPFR probe
// and both C bridges must select the same installed GMP/MPFR/FLINT headers.
fn configure_native_arb(build: &mut cc::Build) {
    println!("cargo:rerun-if-env-changed=XC_NATIVE_PREFIX");
    let prefix = std::env::var_os("XC_NATIVE_PREFIX").map(std::path::PathBuf::from);
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    assert!(
        !windows || prefix.is_some(),
        "Windows Arb builds require XC_NATIVE_PREFIX set to the absolute native GMP/MPFR/FLINT prefix"
    );
    if let Some(prefix) = &prefix {
        assert!(prefix.is_absolute(), "XC_NATIVE_PREFIX must be absolute");
        for header in ["gmp.h", "mpfr.h", "flint/flint.h"] {
            let path = prefix.join("include").join(header);
            assert!(path.is_file(), "missing native header: {}", path.display());
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let lib = prefix.join("lib");
        assert!(
            lib.is_dir(),
            "missing native library directory: {}",
            lib.display()
        );
        if windows {
            assert!(
                prefix.join("bin").is_dir(),
                "XC_NATIVE_PREFIX must contain the native DLL bin directory"
            );
        }
        build.include(prefix.join("include"));
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
}

// Invalidate local execution state across releases AND amendments of one release.
fn checkpoint_build_id() {
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

    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
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
    h.update(b"xcelerator-checkpoint-source-byte-inputs-v2\0");
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
    let target = std::env::var("TARGET").unwrap();
    h.update((target.len() as u64).to_le_bytes());
    h.update(target.as_bytes());
    println!(
        "cargo:rustc-env=XC_CHECKPOINT_BUILD_ID={}",
        hex_lower(&h.finalize())
    );
}
