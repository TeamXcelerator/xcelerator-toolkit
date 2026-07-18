fn main() {
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
