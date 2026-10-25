fn main() {
    println!("cargo:rerun-if-changed=src/quadrature_arb_bridge.c");
    #[cfg(feature = "arb")]
    {
        verify_system_mpfr();
        cc::Build::new()
            .file("src/quadrature_arb_bridge.c")
            .warnings(true)
            .compile("xc_numerics_arb_bridge");
        println!("cargo:rustc-link-lib=dylib=flint");
    }
    // CUDA/cuBLAS are optional runtime discoveries, never link dependencies.
}

// Rug permits Float values on parallel workers. Its bundled MPFR enables TLS;
// system-library builds must establish the same prerequisite for every HP route,
// including callers that never select the Arb quadrature generator.
#[cfg(feature = "arb")]
fn verify_system_mpfr() {
    assert_eq!(
        std::env::var_os("HOST"),
        std::env::var_os("TARGET"),
        "Arb requires a native build to verify the system MPFR thread-safety contract"
    );
    assert!(
        std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("musl"),
        "Arb requires shared system GMP/MPFR; static musl linkage is unsupported"
    );
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let source = out.join("mpfr_thread_safety.c");
    let executable = out.join("mpfr_thread_safety");
    std::fs::write(
        &source,
        "#include <mpfr.h>\nint main(void) { return mpfr_buildopt_tls_p() ? 0 : 1; }\n",
    )
    .expect("write MPFR thread-safety probe");
    let status = cc::Build::new()
        .get_compiler()
        .to_command()
        .arg(&source)
        .args(["-lmpfr", "-lgmp", "-o"])
        .arg(&executable)
        .status()
        .expect("compile MPFR thread-safety probe");
    assert!(status.success(), "could not link the system MPFR probe");
    let status = std::process::Command::new(&executable)
        .status()
        .expect("execute MPFR thread-safety probe");
    assert!(
        status.success(),
        "Arb requires system MPFR built with thread-local storage (--enable-thread-safe)"
    );
}
