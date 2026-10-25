use super::*;
use std::{ffi::OsString, fs, path::Path, process::Command};

// Keep process-global provider configuration out of the parent test process.
fn run_in_child(name: &str) -> bool {
    const CHILD: &str = "XC_QUADRATURE_PROVIDER_TEST_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(name) {
        return false;
    }
    let full_name = format!("distance::hp::quadrature_provider::tests::{name}");
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &full_name, "--test-threads=1", "--nocapture"])
        .env(CHILD, name)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

struct Environment(Vec<(&'static str, Option<OsString>)>);
impl Environment {
    fn save() -> Self {
        Self(
            [
                "XC_TARGET_PROVIDER_EXECUTABLE",
                "XC_TARGET_PROVIDER_PURE",
                "XC_TARGET_PROVIDER_WORKERS",
                "XC_TARGET_SPEC_FILE",
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

fn provider(root: &Path) -> std::path::PathBuf {
    let source = root.join("provider.rs");
    fs::write(&source, r#"
use std::io::{self,BufRead,Write};
fn field<'a>(line:&'a str,name:&str)->Option<&'a str>{
 let marker=format!("\"{}\":",name);
 Some(line.split_once(&marker)?.1.split([',','}']).next().unwrap())
}
fn scalar(line:&str,name:&str)->Option<f64>{field(line,name)?.trim_matches('"').parse().ok()}
fn main(){
 let(mut fail,mut zero)=(None,None);
 for line in io::stdin().lock().lines(){
  let line=line.unwrap();
  let id=field(&line,"request_id").unwrap();let nonce=field(&line,"request_nonce").unwrap();
  let operation=field(&line,"operation").unwrap();let bits=field(&line,"precision_bits").unwrap();
  if operation=="\"initialize\""{fail=scalar(&line,"fail_u");zero=scalar(&line,"zero_u");}
  let u=field(&line,"u").unwrap_or("\"1\"");let point=scalar(&line,"u");
  let value=if point.is_some() && point==fail{"\"invalid-point-scalar\""}
   else if point.is_some()&&point==zero{"\"-0\""}else{u};
  let ready=fail!=Some(-999.0);
  println!("{{\"protocol_version\":2,\"request_id\":{id},\"request_nonce\":{nonce},\"operation\":{operation},\"precision_bits\":{bits},\"ready\":{ready},\"value\":{value}}}");
  io::stdout().flush().unwrap();
 }
}
"#).unwrap();
    let executable = root.join(format!("provider{}", std::env::consts::EXE_SUFFIX));
    let output = Command::new("rustc")
        .args(["--edition=2021", "-C", "debuginfo=0"])
        .arg(source)
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    executable
}

fn configure(
    root: &Path,
    executable: &Path,
    tag: &str,
    fail: Option<&str>,
    zero: Option<&str>,
) -> String {
    let spec = crate::target::TargetProfileSpec::from_json(
        &serde_json::to_vec(&serde_json::json!({
            "schema_version":3,"profile_id":"quadrature-provider-fixture",
            "external_profile":{"lambda_squared":"16","evaluation_precision_bits":320,
              "provider_sha256":xc_cache::ContentDigest::sha256(&fs::read(executable).unwrap()).0,
              "input":{"tag":tag,"fail_u":fail,"zero_u":zero}}
        }))
        .unwrap(),
    )
    .unwrap();
    let spec_file = root.join("target.json");
    fs::write(&spec_file, serde_json::to_vec(&spec).unwrap()).unwrap();
    std::env::set_var("XC_TARGET_SPEC_FILE", spec_file);
    std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", executable);
    spec.digest().unwrap()
}

fn trap(steps: usize) -> WeightedIntegrationRule {
    WeightedIntegrationRule::UniformGrid {
        scheme: UniformGridScheme::Trapezoid,
        variable: GridVariable::U,
        steps,
    }
}

fn rendered(
    result: Result<WeightedGridValueHp>,
) -> std::result::Result<(u32, String, bool), String> {
    result
        .map(|result| {
            (
                result.value.prec(),
                result.value.to_string_radix(16, None),
                result.value.is_sign_negative(),
            )
        })
        .map_err(|error| format!("{error:#}"))
}

fn injected_f(kind: usize, u: &Float) -> Float {
    match kind {
        0 if u == &1 => Float::with_val(256, f64::NAN),
        1 if u == &4 => Float::with_val(256, f64::NAN),
        2 if u != &1 && u != &4 => Float::with_val(256, f64::NAN),
        // Each L1 endpoint is finite; adding the two endpoint values
        // overflows before the trapezoid visits its interior provider failure.
        3 => Float::with_val(256, 0.75) << rug::float::exp_max(),
        4 => Float::with_val(256, -0.0),
        _ => Float::with_val(256, u),
    }
}

#[test]
fn quadrature_provider_deferred_errors_and_order() {
    if run_in_child("quadrature_provider_deferred_errors_and_order") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-errors");
    let executable = provider(&root);
    let lambda = Float::with_val(256, 4);
    let alpha = Float::with_val(256, 0);
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 4).unwrap();
    let _scope = owner.enter();
    for (name, kind, fail) in [
        ("weighted-before-provider", 0, "4"),
        ("same-point-provider-wins", 1, "4"),
        ("provider-before-weighted", 2, "4"),
        ("accumulation-before-provider", 3, "1.75"),
        ("negative-zero-f", 4, "1.75"),
    ] {
        for workers in [2, 4] {
            configure(
                &root,
                &executable,
                &format!("{name}-{workers}"),
                Some(fail),
                None,
            );
            std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
            let expected = rendered(distance_to_target_with_tables_bound(
                |u| injected_f(kind, u),
                &lambda,
                &alpha,
                trap(4),
                192,
                None,
                None,
            ));
            assert!(expected.is_err(), "fixture must fail: {name}");
            if kind == 3 {
                assert!(
                    expected
                        .as_ref()
                        .unwrap_err()
                        .contains("HP uniform-grid evaluation or accumulation is nonfinite"),
                    "finite endpoint samples must fail in the integrator: {expected:?}"
                );
            }
            std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
            std::env::set_var("XC_TARGET_PROVIDER_WORKERS", workers.to_string());
            let bound = BoundTarget::new(&lambda, 192, None).unwrap();
            let prepared = Prepared::try_new(&bound, &lambda, trap(4), 192, None)
                .unwrap()
                .unwrap();
            let actual = rendered(replay(
                |u| injected_f(kind, u),
                &bound,
                Some(&prepared),
                &lambda,
                &alpha,
                trap(4),
                192,
                None,
            ));
            assert_eq!(actual, expected, "{name}, workers{workers}");
            drop(prepared);
            assert_eq!(owner.usage().working_bytes, 0);
        }
    }
    let outcomes = indexed_outcomes(
        9,
        4,
        |first| -> Result<()> {
            if first == 2 {
                anyhow::bail!("deferred worker setup failure");
            }
            Ok(())
        },
        |_, index| Ok(Float::with_val(256, index)),
    );
    assert_eq!(
        outcomes[2]
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_err()
            .to_string(),
        "deferred worker setup failure"
    );
    assert!(outcomes[6].is_none());
    eprintln!("PASS weighted/accumulation/provider precedence, endpoint-first order, and indexed setup failure");
}

#[test]
fn quadrature_provider_coordinate_and_admission_fallbacks() {
    if run_in_child("quadrature_provider_coordinate_and_admission_fallbacks") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-fallback");
    let executable = provider(&root);
    configure(&root, &executable, "fallback", Some("4"), None);
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "4");
    let lambda = Float::with_val(256, 4);
    let bound = BoundTarget::new(&lambda, 192, None).unwrap();
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 4).unwrap();
    {
        let _scope = owner.enter();
        let prepared = Prepared::try_new(&bound, &lambda, trap(4), 192, None)
            .unwrap()
            .unwrap();
        let before = owner.usage().working_bytes;
        assert!(before > 0);
        // An unplanned point must not discard the already attempted failure4.
        prepared.value(&Float::with_val(256, 1.125)).unwrap();
        let four = Float::with_val(256, 4);
        let padded = Float::with_val(257, 4);
        assert_eq!(
            provider_point_key(&bound, &four),
            provider_point_key(&bound, &padded)
        );
        let expected_failure = format!(
            "{:#}",
            prepared.buffers.borrow().outcomes[1]
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap_err()
        );
        assert_eq!(
            format!("{:#}", prepared.value(&padded).unwrap_err()),
            expected_failure
        );
        assert_eq!(owner.usage().working_bytes, before);
        drop(prepared);
        assert_eq!(owner.usage().working_bytes, 0);
    }
    // Distinct wire precision is genuinely unplanned; its call cannot erase a
    // later failure at the original wire coordinate.
    configure(&root, &executable, "precision", None, None);
    let bound = BoundTarget::new(&lambda, 192, None).unwrap();
    {
        let _scope = owner.enter();
        let prepared = Prepared::try_new(&bound, &lambda, trap(4), 192, None)
            .unwrap()
            .unwrap();
        let mut buffers = prepared.buffers.borrow_mut();
        buffers.outcomes[1] = Some(Err(anyhow::anyhow!("known failure original precision")));
        drop(buffers);
        let wide = Float::with_val(512, 4);
        assert_ne!(
            provider_point_key(&bound, &wide),
            provider_point_key(&bound, &Float::with_val(256, 4))
        );
        prepared.value(&wide).unwrap();
        assert_eq!(
            prepared
                .value(&Float::with_val(256, 4))
                .unwrap_err()
                .to_string(),
            "known failure original precision"
        );
        // A consumed/empty grid falls back without a panic.
        prepared
            .buffers
            .borrow_mut()
            .outcomes
            .iter_mut()
            .for_each(|slot| *slot = None);
        prepared.value(&Float::with_val(256, 1.75)).unwrap();
    }
    configure(&root, &executable, "repeat", None, Some("4"));
    let bound = BoundTarget::new(&lambda, 192, None).unwrap();
    {
        let _scope = owner.enter();
        let prepared = Prepared::try_new(&bound, &lambda, trap(4), 192, None)
            .unwrap()
            .unwrap();
        assert!(prepared
            .value(&Float::with_val(256, 4))
            .unwrap()
            .is_sign_negative());
        let mut buffers = prepared.buffers.borrow_mut();
        buffers.points[2] = Float::with_val(256, 4);
        buffers.keys[2] = provider_point_key(&bound, &buffers.points[2]);
        buffers.outcomes[2] = Some(Err(anyhow::anyhow!("second occurrence failure")));
        drop(buffers);
        assert_eq!(
            prepared
                .value(&Float::with_val(256, 4))
                .unwrap_err()
                .to_string(),
            "second occurrence failure"
        );
    }
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "unused-invalid-policy");
    std::env::remove_var("XC_TARGET_PROVIDER_EXECUTABLE");
    assert!(xc_core::RunResources::current().is_none());
    assert!(Prepared::try_new(&bound, &lambda, trap(4), 192, None)
        .unwrap()
        .is_none());
    let tiny = xc_core::RunResources::with_worker_limit(1, 4).unwrap();
    {
        let _scope = tiny.enter();
        assert!(Prepared::try_new(&bound, &lambda, trap(4), 192, None)
            .unwrap()
            .is_none());
        assert_eq!(tiny.usage().working_bytes, 0);
    }
    // Stored lambda precision affects point limbs and exact provider wire keys.
    // This budget exceeds the former low-precision estimate but must reject
    // the larger buffers before inspecting the deliberately malformed policy.
    let wide_lambda = Float::with_val(1 << 20, 4);
    let limited = xc_core::RunResources::with_worker_limit(1 << 20, 4).unwrap();
    {
        let _scope = limited.enter();
        assert!(Prepared::try_new(&bound, &wide_lambda, trap(4), 192, None)
            .unwrap()
            .is_none());
        assert_eq!(limited.usage().working_bytes, 0);
        let tables = SharedGlTables::new();
        assert!(Prepared::try_new(
            &bound,
            &lambda,
            WeightedIntegrationRule::GaussLegendre {
                points: 4,
                variable: GridVariable::U
            },
            192,
            Some(&tables)
        )
        .unwrap()
        .is_none());
        assert!(
            Prepared::try_new(&bound, &lambda, trap(usize::MAX), 192, None)
                .unwrap()
                .is_none()
        );
    }
    assert!(Prepared::try_new(&bound, &lambda, trap(0), 192, None)
        .unwrap()
        .is_none());
    eprintln!("PASS mismatch/repeated/wire-precision/zero/admission/ownerless fallbacks without retrying a known failure");
}

#[test]
fn quadrature_provider_bound_setup_errors() {
    if run_in_child("quadrature_provider_bound_setup_errors") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-setup");
    let executable = provider(&root);
    let lambda = Float::with_val(256, 4);
    let alpha = Float::with_val(192, 0);
    configure(&root, &executable, "setup-error", Some("-999"), None);
    let expected = rendered(distance_to_target_with_tables_bound(
        |_| panic!("setup must precede f"),
        &lambda,
        &alpha,
        trap(4),
        192,
        None,
        None,
    ))
    .unwrap_err();
    let actual = BoundTarget::new(&lambda, 192, None)
        .err()
        .map(|error| format!("{error:#}"))
        .unwrap();
    assert_eq!(actual, expected);
    configure(&root, &executable, "wrong-bound", None, None);
    let expected = rendered(distance_to_target_with_tables_bound(
        |_| panic!("identity must precede f"),
        &lambda,
        &alpha,
        trap(4),
        192,
        None,
        Some("wrong-digest"),
    ))
    .unwrap_err();
    let actual = BoundTarget::new(&lambda, 192, Some("wrong-digest"))
        .err()
        .map(|error| format!("{error:#}"))
        .unwrap();
    assert_eq!(actual, expected);
    eprintln!("PASS unchanged actual provider setup and bound identity error chains");
}

#[test]
fn quadrature_provider_exact_supported_visits() {
    if run_in_child("quadrature_provider_exact_supported_visits") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-visits");
    let executable = provider(&root);
    configure(&root, &executable, "exact-visits", None, None);
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "4");
    let lambda = Float::with_val(256, 4);
    let alpha = Float::with_val(192, 0);
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 4).unwrap();
    let _scope = owner.enter();
    let mut tables = SharedGlTables::new();
    tables.get_or_build(8, 256).unwrap();
    let mut rules = [
        UniformGridScheme::LeftRiemann,
        UniformGridScheme::RightRiemann,
        UniformGridScheme::Midpoint,
        UniformGridScheme::Trapezoid,
    ]
    .into_iter()
    .map(|scheme| WeightedIntegrationRule::UniformGrid {
        scheme,
        variable: GridVariable::U,
        steps: 8,
    })
    .collect::<Vec<_>>();
    rules.push(WeightedIntegrationRule::GaussLegendre {
        points: 8,
        variable: GridVariable::U,
    });
    for rule in rules {
        let bound = BoundTarget::new(&lambda, 192, None).unwrap();
        let prepared = Prepared::try_new(&bound, &lambda, rule, 192, Some(&tables))
            .unwrap()
            .unwrap();
        let expected = prepared
            .buffers
            .borrow()
            .points
            .iter()
            .map(|point| provider_point_key(&bound, point))
            .collect::<Vec<_>>();
        let visits = RefCell::new(Vec::new());
        replay(
            |u| {
                visits.borrow_mut().push(provider_point_key(&bound, u));
                Float::with_val(256, u)
            },
            &bound,
            Some(&prepared),
            &lambda,
            &alpha,
            rule,
            192,
            Some(&mut tables),
        )
        .unwrap();
        assert_eq!(
            visits.into_inner(),
            expected,
            "exact ordered visits for {rule:?}"
        );
        assert!(
            prepared
                .buffers
                .borrow()
                .outcomes
                .iter()
                .all(Option::is_none),
            "all planned results consumed"
        );
    }
    drop(tables);
    assert_eq!(owner.usage().working_bytes, 0);
    eprintln!(
        "PASS exact planned/original callback wire visits for all U uniform schemes and U GL"
    );
}

#[test]
fn quadrature_provider_thread_range_fallback() {
    const NAME: &str =
        "distance::hp::quadrature_provider::tests::quadrature_provider_thread_range_fallback";
    if std::env::var_os("QUADRATURE_EXPONENT_CHILD").is_none() {
        assert!(Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
            .env("QUADRATURE_EXPONENT_CHILD", "1")
            .status()
            .unwrap()
            .success());
        return;
    }
    unsafe extern "C" {
        fn mpfr_set_emin(value: std::os::raw::c_long) -> i32;
        fn mpfr_set_emax(value: std::os::raw::c_long) -> i32;
        fn mpfr_buildopt_tls_p() -> i32;
    }
    if unsafe { mpfr_buildopt_tls_p() } == 0 {
        eprintln!("SKIP isolated range mismatch: MPFR has no thread-local exponent ranges");
        return;
    }
    struct RestoreRange(i32, i32);
    impl Drop for RestoreRange {
        fn drop(&mut self) {
            assert_eq!(unsafe { mpfr_set_emin(self.0.into()) }, 0);
            assert_eq!(unsafe { mpfr_set_emax(self.1.into()) }, 0);
        }
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-range");
    let executable = provider(&root);
    configure(&root, &executable, "range", None, None);
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "4");
    let lambda = Float::with_val(256, 4);
    let bound = BoundTarget::new(&lambda, 192, None).unwrap();
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 4).unwrap();
    let _scope = owner.enter();
    let original = (rug::float::exp_min(), rug::float::exp_max());
    for (minimum, maximum) in [(-1000, original.1), (original.0, 1000)] {
        let restore = RestoreRange(original.0, original.1);
        assert_eq!(unsafe { mpfr_set_emin(minimum.into()) }, 0);
        assert_eq!(unsafe { mpfr_set_emax(maximum.into()) }, 0);
        assert!(!fresh_threads_match());
        assert!(Prepared::try_new(&bound, &lambda, trap(4), 192, None)
            .unwrap()
            .is_none());
        assert_eq!(owner.usage().working_bytes, 0);
        drop(restore);
    }
    eprintln!("PASS fresh std-thread minimum/maximum exponent mismatch chooses serial before bulk planning");
}

fn full_payload(
    eigenfunction: &WeilEigenfunction,
    lambda: &Float,
    cutoff: &str,
    alpha: &Float,
    rules: &[WeightedIntegrationRule],
    prec: u32,
    candidate: bool,
) -> Result<Vec<u8>> {
    let digest = crate::target::TargetProfileSpec::from_environment()?.digest()?;
    let mut tables = SharedGlTables::new();
    for rule in rules {
        if let WeightedIntegrationRule::GaussLegendre { points, .. } = rule {
            tables.get_or_build(*points, prec + 64)?;
        }
    }
    let mut distances = Vec::new();
    let mut measurements = Vec::new();
    for rule in rules {
        let points = weighted_rule_abscissae(*rule, lambda, prec, Some(&tables));
        let values = PrecomputedEigenfunctionValues::from_points(eigenfunction, points);
        let recorded = RefCell::new(Vec::new());
        let distance = if candidate {
            ccm_distance(
                eigenfunction,
                Some(&values),
                Some(&recorded),
                lambda,
                alpha,
                *rule,
                prec,
                Some(&mut tables),
                Some(&digest),
            )?
        } else {
            distance_to_target_with_tables_bound(
                |u| {
                    let value = values.eval(u);
                    recorded.borrow_mut().push(value.clone());
                    value
                },
                lambda,
                alpha,
                *rule,
                prec,
                Some(&mut tables),
                Some(&digest),
            )?
        };
        let recorded = recorded.into_inner();
        let cursor = std::cell::Cell::new(0usize);
        let norm = weighted_alpha_norm_with_tables(
            |_| {
                let i = cursor.get();
                cursor.set(i + 1);
                recorded[i].clone()
            },
            lambda,
            alpha,
            *rule,
            prec,
            Some(&mut tables),
        )?;
        assert_eq!(cursor.get(), recorded.len());
        measurements.push(PortableRuleMeasurement {
            rule_family: rule.family().into(),
            quadrature_rule: rule.rule().into(),
            grid_variable: rule.variable().as_str().into(),
            resolution: rule.resolution(),
            distance_to_target: retained_decimal(&distance.value, prec),
            eigenfunction_norm: retained_decimal(&norm.value, prec),
        });
        distances.push(distance);
    }
    let resolution = if candidate {
        compute_distance_resolution_evidence(
            eigenfunction,
            lambda,
            alpha,
            rules,
            &distances,
            &digest,
            cutoff,
            prec,
        )?
    } else {
        quadrature_serial_resolution(
            eigenfunction,
            lambda,
            alpha,
            rules,
            ResolutionEvidenceEvaluationSource {
                base_distances: &distances,
                base_value_samples: None,
            },
            &digest,
            cutoff,
            prec,
        )?
    };
    // Eigenvalue is common synthetic fixture metadata, not a claimed eigenpair.
    let payload = PortableTargetDistance {
        schema_version: 2,
        target_definition_digest: digest,
        lambda_squared: cutoff.into(),
        n_modes: eigenfunction.normalized_coefficients().len() - 1,
        precision_bits: prec,
        alpha: alpha_identity(alpha, prec),
        measurements,
        eigenvalue: "0".into(),
    };
    Ok(serde_json::to_vec(&(payload, resolution))?)
}

#[test]
fn quadrature_provider_complete_payloads() {
    if run_in_child("quadrature_provider_complete_payloads") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("quadrature-payload");
    let executable = provider(&root);
    let lambda = Float::with_val(256, 4);
    let alpha = Float::with_val(192, 0.5);
    let eigenfunction = WeilEigenfunction::from_normalized_coefficients(
        &[Float::with_val(192, 1), Float::with_val(192, 0.125)],
        &lambda,
        192,
    )
    .unwrap();
    let rules = [
        trap(8),
        WeightedIntegrationRule::GaussLegendre {
            points: 8,
            variable: GridVariable::U,
        },
    ];
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 4).unwrap();
    let _scope = owner.enter();
    for workers in [1, 2, 4] {
        configure(
            &root,
            &executable,
            &format!("payload-{workers}"),
            None,
            None,
        );
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
        let expected =
            full_payload(&eigenfunction, &lambda, "16", &alpha, &rules, 192, false).unwrap();
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
        std::env::set_var("XC_TARGET_PROVIDER_WORKERS", workers.to_string());
        let actual =
            full_payload(&eigenfunction, &lambda, "16", &alpha, &rules, 192, true).unwrap();
        assert_eq!(
            actual, expected,
            "complete base/norm/resolution payloads workers{workers}"
        );
        let (_, resolution): (PortableTargetDistance, PortableDistanceResolutionEvidence) =
            serde_json::from_slice(&actual).unwrap();
        assert_eq!(resolution.refinements.len(), 1);
        let refinement = &resolution.refinements[0];
        assert_eq!(refinement.base_resolution, 8);
        assert_eq!(refinement.twice_resolution, 16);
        assert_eq!(refinement.four_times_resolution, Some(32));
        assert_eq!(refinement.final_resolution, 32);
        assert!(!refinement.tolerance_met);

        // With f(u)=1, target(u)=u, and alpha=0, |1-u| is linear on [1,4].
        // Trapezoid is exact already at Q, so the policy stops at 2Q.
        let constant = WeilEigenfunction::from_normalized_coefficients(
            &[Float::with_val(192, 1)],
            &lambda,
            192,
        )
        .unwrap();
        let zero_alpha = Float::with_val(192, 0);
        configure(
            &root,
            &executable,
            &format!("resolved-payload-{workers}"),
            None,
            None,
        );
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
        let expected = full_payload(
            &constant,
            &lambda,
            "16",
            &zero_alpha,
            &[trap(4)],
            192,
            false,
        )
        .unwrap();
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
        std::env::set_var("XC_TARGET_PROVIDER_WORKERS", workers.to_string());
        let actual =
            full_payload(&constant, &lambda, "16", &zero_alpha, &[trap(4)], 192, true).unwrap();
        assert_eq!(actual, expected, "resolved 2Q payload workers{workers}");
        let (_, resolution): (PortableTargetDistance, PortableDistanceResolutionEvidence) =
            serde_json::from_slice(&actual).unwrap();
        assert_eq!(resolution.refinements.len(), 1);
        let refinement = &resolution.refinements[0];
        assert_eq!(refinement.base_resolution, 4);
        assert_eq!(refinement.twice_resolution, 8);
        assert_eq!(refinement.four_times_resolution, None);
        assert_eq!(refinement.four_times_distance, None);
        assert_eq!(refinement.final_resolution, 8);
        assert!(refinement.tolerance_met);

        // These points first occur at 2Q16 and 4Q32, respectively, for Q8.
        for (stage, point) in [("2q", "1.1875"), ("4q", "1.09375")] {
            configure(
                &root,
                &executable,
                &format!("refinement-failure-{stage}-{workers}"),
                Some(point),
                None,
            );
            std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
            let expected = full_payload(&eigenfunction, &lambda, "16", &alpha, &rules, 192, false)
                .unwrap_err();
            std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
            std::env::set_var("XC_TARGET_PROVIDER_WORKERS", workers.to_string());
            let actual =
                full_payload(&eigenfunction, &lambda, "16", &alpha, &rules, 192, true).unwrap_err();
            assert_eq!(
                format!("{actual:#}"),
                format!("{expected:#}"),
                "{stage} provider failure workers{workers}"
            );
            assert_eq!(owner.usage().working_bytes, 0);
        }
    }
    assert_eq!(owner.usage().working_bytes, 0);
    eprintln!("PASS complete serialized distance/norm and conditional2Q/4Q resolution fixture at1/2/4workers");
}

#[test]
fn profile_provider_complete_payloads_and_order() {
    if run_in_child("profile_provider_complete_payloads_and_order") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("profile-provider-payload");
    let executable = provider(&root);
    let p = 192;
    let lambda = Float::with_val(p, 4);
    let alpha = Float::with_val(p, 0);
    let eigenfunction = WeilEigenfunction::from_normalized_coefficients(
        &[Float::with_val(p, 1), Float::with_val(p, 0.125)],
        &lambda,
        p,
    )
    .unwrap();
    let us = (0..=32)
        .map(|j| Float::with_val(p, 1) + Float::with_val(p, j) * 3 / 32)
        .collect::<Vec<_>>();
    let fs = us
        .iter()
        .enumerate()
        .map(|(j, u)| Float::with_val(p, u) + Float::with_val(p, (j % 3) as i32 - 1) / 8)
        .collect::<Vec<_>>();
    let rules = [trap(16)];
    let owner = xc_core::RunResources::with_worker_limit(1 << 28, 16).unwrap();
    let _scope = owner.enter();
    for workers in [1, 8, 16] {
        configure(
            &root,
            &executable,
            &format!("profile-{workers}"),
            None,
            Some("2.5"),
        );
        let mut spec = crate::target::TargetProfileSpec::from_environment().unwrap();
        spec.auxiliary_series = Some(crate::target::GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: vec!["0".into(), "1".into()],
            polynomial_scale: crate::target::ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: vec!["1".into()],
            parameter_polynomial_scale: crate::target::ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 1000,
        });
        fs::write(root.join("target.json"), serde_json::to_vec(&spec).unwrap()).unwrap();
        let digest = spec.digest().unwrap();
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
        let distances = rules
            .iter()
            .map(|rule| {
                distance_to_target_with_tables_bound(
                    |u| eigenfunction.eval(u),
                    &lambda,
                    &alpha,
                    *rule,
                    p,
                    None,
                    Some(&digest),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let residual = || {
            compute_target_residual_analysis(TargetResidualAnalysisSource {
                eigenfunction: &eigenfunction,
                lambda: &lambda,
                alpha: &alpha,
                rules: &rules,
                base_distances: &distances,
                u_values: &us,
                f_values: &fs,
                precomputed_signed_residuals: None,
                target_definition_digest: &digest,
                lambda_squared: "16",
                sampling_variable: GridVariable::U,
                precision_bits: p,
            })
            .map(|value| serde_json::to_vec(&value).unwrap())
        };
        let profile = PortableEigenfunctionProfile {
            schema_version: 1,
            lambda_squared: "16".into(),
            n_modes: 1,
            precision_bits: p,
            grid_variable: "uniform_u".into(),
            sample_count: us.len(),
            normalization: "f(1)=1".into(),
            u_values: us.iter().map(|u| decimal(u, p)).collect(),
            f_values: fs.iter().map(|f| decimal(f, p)).collect(),
            normalized_coefficients: vec![],
        };
        let expected = residual().unwrap();
        let expected_deviation =
            serde_json::to_vec(&compute_deviation_decomposition_payload(&profile, p).unwrap())
                .unwrap();
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
        std::env::set_var("XC_TARGET_PROVIDER_WORKERS", workers.to_string());
        assert_eq!(
            residual().unwrap(),
            expected,
            "complete residual workers={workers}"
        );
        assert_eq!(
            serde_json::to_vec(&compute_deviation_decomposition_payload(&profile, p).unwrap())
                .unwrap(),
            expected_deviation,
            "complete deviation workers={workers}"
        );
        assert_eq!(owner.usage().working_bytes, 0);
    }
    for (tag, bad_u, bad_f, fail) in [
        ("parse-u-first", Some(2), None, "4"),
        ("parse-f-first", None, Some(2), "4"),
        ("provider-first", Some(30), Some(31), "1.75"),
    ] {
        configure(&root, &executable, tag, Some(fail), None);
        let mut profile = PortableEigenfunctionProfile {
            schema_version: 1,
            lambda_squared: "16".into(),
            n_modes: 1,
            precision_bits: p,
            grid_variable: "uniform_u".into(),
            sample_count: us.len(),
            normalization: "f(1)=1".into(),
            u_values: us.iter().map(|u| decimal(u, p)).collect(),
            f_values: fs.iter().map(|f| decimal(f, p)).collect(),
            normalized_coefficients: vec![],
        };
        // An auxiliary failure at every point would mask the intended failures.
        let path = root.join("target.json");
        let mut spec = crate::target::TargetProfileSpec::from_environment().unwrap();
        spec.auxiliary_series = Some(crate::target::GaussianPolynomialSeriesSpec {
            term_input_power: 0,
            polynomial_coefficients: vec!["0".into(), "1".into()],
            polynomial_scale: crate::target::ScalarScaleSpec::default(),
            parameter_polynomial_coefficients: vec!["1".into()],
            parameter_polynomial_scale: crate::target::ScalarScaleSpec::default(),
            minimum_terms: 2,
            maximum_terms: 1000,
        });
        fs::write(path, serde_json::to_vec(&spec).unwrap()).unwrap();
        if let Some(index) = bad_u {
            profile.u_values[index] = "invalid".into();
        }
        if let Some(index) = bad_f {
            profile.f_values[index] = "invalid".into();
        }
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
        let target = crate::target::hp::TargetEvaluator::from_spec(&spec, p + 64).unwrap();
        let expected = format!(
            "{:#}",
            deviation_samples_serial_reference(&profile, p, &target).unwrap_err()
        );
        std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
        let target = crate::target::hp::TargetEvaluator::from_spec(&spec, p + 64).unwrap();
        let actual = format!("{:#}", deviation_samples(&profile, p, &target).unwrap_err());
        assert_eq!(actual, expected, "{tag}");
        assert_eq!(owner.usage().working_bytes, 0);
    }
}

#[test]
#[ignore = "explicit retained-state/provider replay, with an isolated output directory"]
fn profile_provider_retained_source_replay() {
    let _environment = Environment::save();
    let output = std::path::PathBuf::from(std::env::var_os("XC_PROFILE_REPLAY_OUTPUT").unwrap());
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var_os("XC_PROFILE_REPLAY_STATE").unwrap()).unwrap(),
    )
    .unwrap();
    let p = state["precision_bits"].as_u64().unwrap() as u32;
    let n = state["n_modes"].as_u64().unwrap() as usize;
    let cutoff = state["lambda_squared"].as_str().unwrap();
    let lambda = Float::with_val(p + 64, Float::parse(cutoff).unwrap()).sqrt();
    let xi = state["eigenvector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| Float::with_val(p, Float::parse(v.as_str().unwrap()).unwrap()))
        .collect::<Vec<_>>();
    let eigenfunction = WeilEigenfunction::from_v_basis(&xi, n, &lambda, p).unwrap();
    let target = crate::target::TargetProfileSpec::from_environment().unwrap();
    let digest = target.digest().unwrap();
    let owner = xc_core::RunResources::with_worker_limit(1 << 30, 16).unwrap();
    let _scope = owner.enter();
    let us = (0..=64)
        .map(|i: u32| (Float::with_val(p, &lambda).ln() * i / 64u32).exp())
        .collect::<Vec<_>>();
    let fs = us.iter().map(|u| eigenfunction.eval(u)).collect::<Vec<_>>();
    let profile = PortableEigenfunctionProfile {
        schema_version: 1,
        lambda_squared: cutoff.into(),
        n_modes: n,
        precision_bits: p,
        grid_variable: "uniform_log_u".into(),
        sample_count: us.len(),
        normalization: "f(1)=1".into(),
        u_values: us.iter().map(|u| decimal(u, p)).collect(),
        f_values: fs.iter().map(|f| decimal(f, p)).collect(),
        normalized_coefficients: vec![],
    };
    let rules = [trap(16)];
    let alpha = Float::with_val(p, 0.5);
    let distances = rules
        .iter()
        .map(|rule| {
            distance_to_target_with_tables_bound(
                |u| eigenfunction.eval(u),
                &lambda,
                &alpha,
                *rule,
                p,
                None,
                Some(&digest),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let started = std::time::Instant::now();
    let residual = compute_target_residual_analysis(TargetResidualAnalysisSource {
        eigenfunction: &eigenfunction,
        lambda: &lambda,
        alpha: &alpha,
        rules: &rules,
        base_distances: &distances,
        u_values: &us,
        f_values: &fs,
        precomputed_signed_residuals: None,
        target_definition_digest: &digest,
        lambda_squared: cutoff,
        sampling_variable: GridVariable::LogU,
        precision_bits: p,
    })
    .unwrap();
    let residual_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let deviation = compute_deviation_decomposition_payload(&profile, p).unwrap();
    let deviation_seconds = started.elapsed().as_secs_f64();
    let bytes = serde_json::to_vec(&(residual, deviation)).unwrap();
    fs::write(output.join("payload.json"), &bytes).unwrap();
    let record = serde_json::json!({"target_digest":digest,"precision_bits":p,"n_modes":n,"lambda_squared":cutoff,
        "points":65,"workers":std::env::var("XC_TARGET_PROVIDER_WORKERS").unwrap(),
        "payload_sha256":xc_cache::ContentDigest::sha256(&bytes).0,"residual_seconds":residual_seconds,
        "deviation_seconds":deviation_seconds,"working_bytes_after":owner.usage().working_bytes});
    fs::write(
        output.join("timing.json"),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .unwrap();
    eprintln!("PROFILE_REPLAY {record}");
    assert_eq!(owner.usage().working_bytes, 0);
}

#[test]
fn profile_provider_warm_and_resource_fallbacks() {
    if run_in_child("profile_provider_warm_and_resource_fallbacks") {
        return;
    }
    let _environment = Environment::save();
    let root = xc_core::test_support::TestDir::new("profile-provider-fallbacks");
    let executable = provider(&root);
    configure(&root, &executable, "profile-fallback", None, None);
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "1");
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "4");
    let spec = crate::target::TargetProfileSpec::from_environment().unwrap();
    let target = crate::target::hp::TargetEvaluator::from_spec(&spec, 256).unwrap();
    let us = [Float::with_val(192, 1.25), Float::with_val(192, 2.5)];
    let points = us.iter().map(Some).collect::<Vec<_>>();
    assert!(ProfileTargets::try_new(&target, &points, 256)
        .unwrap()
        .is_none());
    let tiny = xc_core::RunResources::with_worker_limit(1, 4).unwrap();
    {
        let _scope = tiny.enter();
        assert!(ProfileTargets::try_new(&target, &points, 256)
            .unwrap()
            .is_none());
    }
    assert_eq!(tiny.usage().working_bytes, 0);
    let expected = us
        .iter()
        .map(|u| target.try_value(u).unwrap().to_string())
        .collect::<Vec<_>>();
    let owner = xc_core::RunResources::with_worker_limit(1 << 20, 4).unwrap();
    let _scope = owner.enter();
    std::env::remove_var("XC_TARGET_PROVIDER_EXECUTABLE");
    assert!(
        ProfileTargets::try_new(&target, &points, 256)
            .unwrap()
            .is_none(),
        "warm phase must not spawn providers"
    );
    assert_eq!(
        us.iter()
            .map(|u| target.try_value(u).unwrap().to_string())
            .collect::<Vec<_>>(),
        expected
    );
    std::env::set_var("XC_TARGET_PROVIDER_PURE", "0");
    std::env::set_var("XC_TARGET_PROVIDER_WORKERS", "not-used");
    assert!(ProfileTargets::try_new(&target, &points, 256)
        .unwrap()
        .is_none());
    assert_eq!(owner.usage().working_bytes, 0);
}
