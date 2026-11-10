fn main() {
    println!("cargo:rerun-if-changed=src/quadrature_arb_bridge.c");
    println!("cargo:rustc-check-cfg=cfg(xc_windows_gnu_ucrt)");
    #[cfg(feature = "arb")]
    {
        let mut bridge = cc::Build::new();
        let prefix = configure_native_arb(&mut bridge);
        if verify_system_mpfr(&bridge, prefix.as_deref()) {
            println!("cargo:rustc-cfg=xc_windows_gnu_ucrt");
        }
        bridge
            .file("src/quadrature_arb_bridge.c")
            .warnings(true)
            .compile("xc_numerics_arb_bridge");
        println!("cargo:rustc-link-lib=dylib=flint");
    }
    // CUDA/cuBLAS are optional runtime discoveries, never link dependencies.
}

// One explicit prefix keeps the C bridge, probe, and native linker on the same
// system installation. The launcher must give gmp-mpfr-sys/pkg-config that same
// prefix; generated DEP_GMP_* paths need not contain the system headers.
#[cfg(feature = "arb")]
fn configure_native_arb(build: &mut cc::Build) -> Option<std::path::PathBuf> {
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
    prefix
}

// Rug permits Float values on parallel workers. Its bundled MPFR enables TLS;
// system-library builds must establish the same prerequisite for every HP route,
// including callers that never select the Arb quadrature generator.
#[cfg(feature = "arb")]
fn verify_system_mpfr(build: &cc::Build, prefix: Option<&std::path::Path>) -> bool {
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
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    let executable = out.join(if windows {
        "mpfr_thread_safety.exe"
    } else {
        "mpfr_thread_safety"
    });
    std::fs::write(
        &source,
        concat!(
            "#include <gmp.h>\n#include <mpfr.h>\n#include <string.h>\n#include <stdio.h>\n",
            "#define XC_STR_(x) #x\n#define XC_STR(x) XC_STR_(x)\n",
            "int main(void) {\n",
            "if (!mpfr_buildopt_tls_p()) return 1;\n",
            "if (strcmp(mpfr_get_version(), MPFR_VERSION_STRING)) return 2;\n",
            "if (strcmp(gmp_version, XC_STR(__GNU_MP_VERSION) \".\" ",
            "XC_STR(__GNU_MP_VERSION_MINOR) \".\" XC_STR(__GNU_MP_VERSION_PATCHLEVEL))) return 3;\n",
            // GNU is not enough: the legacy MSVCRT target has the same Rust
            // target triple. Admit only the native x64 UCRT C compiler lane.
            "#if defined(_WIN64) && defined(__MINGW64__) && defined(_UCRT)\n",
            "puts(\"xc_windows_gnu_ucrt\");\n",
            "#endif\n",
            "return 0;\n}\n",
        ),
    )
    .expect("write MPFR thread-safety probe");
    let compiler = build.get_compiler();
    let mut compile = compiler.to_command();
    compile.arg(&source);
    if compiler.is_like_msvc() {
        let mut output = std::ffi::OsString::from("/Fe");
        output.push(&executable);
        let mut object = std::ffi::OsString::from("/Fo");
        object.push(out.join("mpfr_thread_safety.obj"));
        compile.arg(output).arg(object).arg("/link");
        if let Some(prefix) = prefix {
            let mut lib = std::ffi::OsString::from("/LIBPATH:");
            lib.push(prefix.join("lib"));
            compile.arg(lib);
        }
        compile.args(["mpfr.lib", "gmp.lib"]);
    } else {
        if let Some(prefix) = prefix {
            compile.arg("-L").arg(prefix.join("lib"));
        }
        compile.args(["-lmpfr", "-lgmp", "-o"]).arg(&executable);
    }
    let output = compile.output().expect("compile MPFR thread-safety probe");
    assert!(
        output.status.success(),
        "could not link the system MPFR probe: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut run = std::process::Command::new(&executable);
    if windows {
        if let Some(prefix) = prefix {
            let mut paths = vec![prefix.join("bin")];
            paths.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            run.env(
                "PATH",
                std::env::join_paths(paths).expect("native DLL search path"),
            );
        }
    }
    let output = run.output().expect("execute MPFR thread-safety probe");
    assert!(
        output.status.success(),
        "Arb requires system MPFR built with thread-local storage and matching GMP/MPFR headers/runtime (probe status {})",
        output.status
    );
    windows
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        && String::from_utf8_lossy(&output.stdout).trim() == "xc_windows_gnu_ucrt"
}
