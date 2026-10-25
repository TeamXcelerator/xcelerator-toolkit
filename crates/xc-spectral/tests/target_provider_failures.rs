use std::{ffi::OsString, fs, path::PathBuf, process::Command, sync::Mutex};
use xc_spectral::target::{ExternalProfileSpec, TargetEvaluatorF64, TargetProfileSpec};

// This integration-test executable is isolated from the library's environment
// tests. Serialize its own cases and restore every variable even after a panic.
static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct Environment(Vec<(&'static str, Option<OsString>)>);

impl Environment {
    fn capture() -> Self {
        Self(
            [
                "XC_TARGET_PROVIDER_EXECUTABLE",
                "XC_TARGET_PROVIDER_PURE",
                "XC_TARGET_PROVIDER_TIMEOUT_SECONDS",
                "XC_TARGET_PROVIDER_STDERR_DIR",
                "XC_TEST_PROVIDER_COUNTER",
                "XC_TEST_PROVIDER_BAD_SCALAR",
            ]
            .into_iter()
            .map(|key| (key, std::env::var_os(key)))
            .collect(),
        )
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

struct Fixture {
    _environment: Environment,
    root: xc_core::test_support::TestDir,
    digest: String,
    counter: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let environment = Environment::capture();
        let root = xc_core::test_support::TestDir::new("provider-failure-recovery");
        let source = root.join("provider.rs");
        fs::write(&source, include_str!("fixtures/provider_failures.rs")).unwrap();
        let executable = root.join(if cfg!(windows) {
            "provider.exe"
        } else {
            "provider"
        });
        let output = Command::new("rustc")
            .args(["--edition=2021", "--crate-name", "provider_failure_fixture"])
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let digest = xc_cache::ContentDigest::sha256(&fs::read(&executable).unwrap()).0;
        let counter = root.join("requests.txt");
        std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", &executable);
        std::env::set_var("XC_TARGET_PROVIDER_TIMEOUT_SECONDS", "5");
        std::env::remove_var("XC_TARGET_PROVIDER_STDERR_DIR");
        std::env::set_var("XC_TEST_PROVIDER_COUNTER", &counter);
        Self {
            _environment: environment,
            root,
            digest,
            counter,
        }
    }

    fn spec(&self, bad_scalar: &str) -> TargetProfileSpec {
        std::env::set_var("XC_TEST_PROVIDER_BAD_SCALAR", bad_scalar);
        TargetProfileSpec {
            schema_version: 3,
            profile_id: "provider-failure-recovery".into(),
            base_series: None,
            auxiliary_series: None,
            external_profile: Some(ExternalProfileSpec {
                lambda_squared: "13".into(),
                evaluation_precision_bits: 256,
                provider_sha256: self.digest.clone(),
                input: serde_json::json!({"bad_scalar": bad_scalar}),
            }),
        }
    }

    fn requests(&self) -> u32 {
        fs::read_to_string(&self.counter).unwrap().parse().unwrap()
    }
}

#[test]
fn native_failed_scalars_are_retried_then_memoized() {
    let _lock = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());
    let fixture = Fixture::new();
    for bad in [
        "malformed-once",
        "NaN",
        "1e400",
        "1e-400",
        "4.9406564584124654e-324",
    ] {
        let spec = fixture.spec(bad);
        for pure in [true, false] {
            std::env::set_var("XC_TARGET_PROVIDER_PURE", if pure { "1" } else { "0" });
            let target = TargetEvaluatorF64::from_spec(&spec).unwrap();
            assert!(target.try_value(2.0).is_err(), "{bad}");
            assert_eq!(fixture.requests(), 1);
            assert_eq!(target.try_value(2.0).unwrap(), 2.0);
            assert_eq!(fixture.requests(), 2);
            assert_eq!(target.try_value(2.0).unwrap(), 2.0);
            assert_eq!(fixture.requests(), if pure { 2 } else { 3 });
        }
    }
}

#[cfg(feature = "hp")]
#[test]
fn hp_failed_scalars_are_retried_then_memoized() {
    let _lock = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());
    let fixture = Fixture::new();
    for bad in ["malformed-once", "NaN", "1e400000000", "1e-400000000"] {
        let spec = fixture.spec(bad);
        for pure in [true, false] {
            std::env::set_var("XC_TARGET_PROVIDER_PURE", if pure { "1" } else { "0" });
            let target = xc_spectral::target::hp::TargetEvaluator::from_spec(&spec, 128).unwrap();
            let point = rug::Float::with_val(128, 2);
            assert!(target.try_value(&point).is_err(), "{bad}");
            assert_eq!(fixture.requests(), 1);
            assert_eq!(target.try_value(&point).unwrap(), 2);
            assert_eq!(fixture.requests(), 2);
            assert_eq!(target.try_value(&point).unwrap(), 2);
            assert_eq!(fixture.requests(), if pure { 2 } else { 3 });
        }
    }
}

#[test]
fn invalid_purity_is_rejected_before_provider_side_effects() {
    let _lock = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());
    let fixture = Fixture::new();
    let spec = fixture.spec("malformed-once");
    let stderr = fixture.root.join("stderr");
    fs::create_dir(&stderr).unwrap();
    std::env::set_var("XC_TARGET_PROVIDER_STDERR_DIR", &stderr);
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "true");
    let error = TargetEvaluatorF64::from_spec(&spec).unwrap_err();
    assert_eq!(error.to_string(), "XC_TARGET_PROVIDER_PURE must be 0 or 1");
    // Opening the diagnostic log precedes spawn, so this is deterministic:
    // no sleep or race against a child's startup marker is needed.
    assert_eq!(fs::read_dir(&stderr).unwrap().count(), 0);
    // Positive control: a permitted start really does create its private log.
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
    let target = TargetEvaluatorF64::from_spec(&spec).unwrap();
    assert_eq!(target.try_value(1.0).unwrap(), 1.0);
    drop(target);
    assert_eq!(fs::read_dir(&stderr).unwrap().count(), 1);
}
