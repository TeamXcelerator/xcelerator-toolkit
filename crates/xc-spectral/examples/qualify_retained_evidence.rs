//! Bounded synthetic qualification with analytic spectra, not CCM research data.
//! The dense reflector fixture has distinct eigenvalues and exercises a full
//! reduction; rank-one and tridiagonal fixtures are structural regressions.
#[cfg(feature = "hp")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, ensure};
    use rug::{Float, Integer, Rational};
    use serde_json::json;
    use std::{
        collections::BTreeMap,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };
    use xc_cache::*;
    use xc_spectral::ccm::{
        capture::*,
        prefix::{RetainedEvenEigenpair, RetainedEvenMatrix},
    };
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=5).contains(&args.len()) {
        bail!("usage: qualify_retained_evidence DIMENSION BITS WORKERS [dense-reflector|rank-one|toeplitz] [legacy|third-moment|third-moment-no-cancellation] (dimension power of two, 2..1024)");
    }
    let n: usize = args[0].parse()?;
    let bits: u32 = args[1].parse()?;
    let workers: usize = args[2].parse()?;
    let fixture = args.get(3).map(String::as_str).unwrap_or("dense-reflector");
    let diagnostic_mode = args.get(4).map(String::as_str).unwrap_or("legacy");
    let diagnostics = match diagnostic_mode {
        "legacy" => xc_core::PrefixDiagnosticPolicy::default(),
        "third-moment" => xc_core::PrefixDiagnosticPolicy::full(),
        "third-moment-no-cancellation" => xc_core::PrefixDiagnosticPolicy {
            third_inverse_moment: true,
            innovation_cancellation: false,
        },
        _ => bail!("unknown diagnostic mode"),
    };
    ensure!(
        matches!(fixture, "dense-reflector" | "rank-one" | "toeplitz"),
        "unknown fixture"
    );
    ensure!(
        n.is_power_of_two()
            && (2..=1024).contains(&n)
            && (128..=1024).contains(&bits)
            && (1..=8).contains(&workers),
        "qualification dimensions, precision or workers outside budget"
    );
    let started = Instant::now();
    // H = I - 2uu^T/s, u_i=i+1, D_ii=1+i/n, A=HDH. Construct each
    // entry as an exact rational, then round once to the source precision.
    // Reference eigenvalues belong to the ideal rational matrix; source
    // rounding is therefore included in the measured spectrum discrepancy.
    let s: u64 = (1..=n as u64).map(|u| u * u).sum();
    let weighted: u64 = (0..n as u64).map(|j| (j + 1).pow(2) * (n as u64 + j)).sum();
    let denominator = Integer::from(n) * s * s;
    let off = Float::with_val(bits, 1) / Float::with_val(bits, n);
    let mut diag = off.clone();
    diag += 1;
    let off = xc_numerics::prefix::lossless_decimal(&off);
    let diag = xc_numerics::prefix::lossless_decimal(&diag);
    let entries = (0..n * n)
        .map(|i| {
            if fixture == "dense-reflector" {
                let row = (i / n) as u64;
                let col = (i % n) as u64;
                let uv = (row + 1) * (col + 1);
                let mut numerator = Integer::from(4 * uv) * weighted
                    - Integer::from(2 * uv) * (2 * n as u64 + row + col) * s;
                if row == col {
                    numerator += Integer::from(n as u64 + row) * s * s;
                }
                xc_numerics::prefix::lossless_decimal(&Float::with_val(
                    bits,
                    Rational::from((numerator, denominator.clone())),
                ))
            } else if fixture == "toeplitz" {
                match (i / n).abs_diff(i % n) {
                    0 => "2",
                    1 => "-0.5",
                    _ => "0",
                }
                .to_owned()
            } else if i / n == i % n {
                diag.clone()
            } else {
                off.clone()
            }
        })
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(
        &json!({"schema_version":1,"lambda_squared":"13","n_modes":n-1,"precision_bits":bits,"dimension":n,"entries":entries}),
    )?;
    drop(entries);
    let digest = ContentDigest::sha256(&bytes);
    let source = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new(
            "ccm_even_sector_matrix",
            format!("synthetic-{fixture}-qualification"),
            format!("{fixture}/{n}/{bits}").as_bytes(),
        )?,
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.13.0")?,
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::from([(
            "fixture".into(),
            "synthetic matrix with exact analytic spectrum; not a CCM construction".into(),
        )]),
        provenance_digest: None,
    };
    let matrix = RetainedEvenMatrix::from_payload(&source, &bytes, &[digest])?;
    drop(bytes);
    let nonzero_entries = matrix.entries().iter().filter(|v| !v.is_zero()).count();
    if fixture == "dense-reflector" {
        ensure!(
            nonzero_entries == n * n,
            "dense fixture contains exact zeros"
        );
    }
    let reference_bits = bits + 64;
    let angle = Float::with_val(reference_bits, rug::float::Constant::Pi)
        / Float::with_val(reference_bits, n + 1);
    let reference_values: Vec<Float> = (0..n)
        .map(|j| {
            if fixture == "rank-one" {
                Float::with_val(reference_bits, if j + 1 == n { 2 } else { 1 })
            } else if fixture == "dense-reflector" {
                Float::with_val(reference_bits, Rational::from((n + j, n)))
            } else {
                Float::with_val(reference_bits, 2)
                    - Float::with_val(reference_bits, &angle * (j + 1)).cos()
            }
        })
        .collect();
    let even_vector: Vec<Float> = (0..n)
        .map(|j| {
            if fixture == "rank-one" {
                Float::with_val(bits, 1)
            } else if fixture == "dense-reflector" {
                let mut entry = -Rational::from((2 * (j as u64 + 1), s));
                if j == 0 {
                    entry += 1;
                }
                Float::with_val(bits, entry)
            } else {
                Float::with_val(
                    bits,
                    Float::with_val(reference_bits, &angle * (j + 1)).sin(),
                )
            }
        })
        .collect();
    let eigenvalue = if fixture == "rank-one" {
        &reference_values[n - 1]
    } else {
        &reference_values[0]
    };
    let full_vector = xc_spectral::ccm::hp::expand_even_sector_vector(&even_vector, n - 1, bits);
    let eigenbytes = serde_json::to_vec(&json!({
        "schema_version":2,"lambda_squared":"13","n_modes":n-1,"precision_bits":bits,
        "eigenvalue":xc_numerics::prefix::lossless_decimal(&Float::with_val(bits, eigenvalue)),
        "eigenvector":full_vector.iter().map(xc_numerics::prefix::lossless_decimal).collect::<Vec<_>>()
    }))?;
    let mut eigenmanifest = source.clone();
    eigenmanifest.key = ArtifactKey::new(
        "ccm_weil_eigenpair",
        format!("synthetic-{fixture}-eigenstate"),
        format!("{fixture}/{n}/{bits}").as_bytes(),
    )?;
    eigenmanifest.content_digest = ContentDigest::sha256(&eigenbytes);
    eigenmanifest.size_bytes = eigenbytes.len() as u64;
    eigenmanifest.objects = vec![CacheObjectRef {
        content_digest: eigenmanifest.content_digest.clone(),
        size_bytes: eigenmanifest.size_bytes,
    }];
    let eigenpair = RetainedEvenEigenpair::from_payload(
        &eigenmanifest,
        &eigenbytes,
        std::slice::from_ref(&eigenmanifest.content_digest),
    )?;
    let eigenpairs = [eigenpair];
    let root = std::env::temp_dir().join(format!(
        "xc-retained-qualification-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let resolver = CacheResolver::new(vec![CacheLayer {
        precedence: 0,
        store: Box::new(FilesystemCacheStore::new(
            "qualification",
            root.clone(),
            true,
            CacheVisibility::Local,
        )),
    }]);
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    let context = |mode: ArtifactExecutionCacheMode| ArtifactCacheContext {
        resolver: Some(&resolver),
        reference_resolver: None,
        acceptance: Some(&policy),
        ordered_overlays: vec!["qualification".into()],
        mode,
        write_on_miss: !mode.requires_reuse(),
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    };
    let plan = CcmCapturePlan::resolve(CcmCaptureLevel::Claim, 2, n)?
        .with_prefix_checkpoints(vec![n])?
        .with_prefix_diagnostics(diagnostics)?;
    let tolerance = format!("1e-{}", (bits - 64) * 3 / 10);
    let budget = RetainedReductionRequest {
        working_precision_bits: bits,
        maximum_dimension: n,
        relative_tolerance: tolerance.clone(),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let run = |mode| {
        pool.install(|| {
            plan.execute_with_receipt_and_reduction(
                Some((&matrix, &eigenpairs)),
                Some(&budget),
                &context(mode),
                |_| unreachable!("only retained diagnostics requested"),
            )
        })
    };
    let cold_start = Instant::now();
    let cold = run(ArtifactExecutionCacheMode::PreferReuse)?;
    let cold_seconds = cold_start.elapsed().as_secs_f64();
    let warm_start = Instant::now();
    let warm = run(ArtifactExecutionCacheMode::RequireReuse)?;
    let warm_seconds = warm_start.elapsed().as_secs_f64();
    let warm_byte_identity = serde_json::to_vec(&cold.value)? == serde_json::to_vec(&warm.value)?;
    ensure!(warm_byte_identity, "warm receipt differs from cold receipt");
    ensure!(
        cold.produced_manifest == warm.reused_manifest,
        "warm manifest differs from cold manifest"
    );
    cold.value.validate()?;
    let prefix = &cold.value.measurements["prefix_ladder"].value;
    ensure!(
        prefix["ladder"]["stopped"].is_null(),
        "prefix ladder stopped"
    );
    let trace = prefix["ladder"]["rows"].as_array().unwrap().last().unwrap()["inverse_trace"]
        .as_str()
        .unwrap();
    let mut trace_error = Float::with_val(bits, Float::parse(trace)?);
    let expected_trace = if fixture == "rank-one" {
        Float::with_val(reference_bits, n) - Float::with_val(reference_bits, 0.5)
    } else {
        reference_values
            .iter()
            .fold(Float::with_val(reference_bits, 0), |sum, value| {
                sum + Float::with_val(reference_bits, 1) / value
            })
    };
    trace_error.set_prec(reference_bits);
    trace_error -= &expected_trace;
    trace_error.abs_mut();
    let tol = Float::with_val(bits, Float::parse(&tolerance)?);
    ensure!(
        trace_error <= tol,
        "inverse trace disagrees with analytic reference"
    );
    let reduction = &cold.value.measurements["retained_reduction"].value;
    ensure!(
        reduction["checks_passed"] == true,
        "reduction residual screen failed"
    );
    let spectrum = reduction["computed_eigenvalues"].as_array().unwrap();
    ensure!(spectrum.len() == n, "spectrum comparison length mismatch");
    let mut maximum_spectrum_error = Float::with_val(reference_bits, 0);
    let mut errors = Vec::with_capacity(n);
    for (i, value) in spectrum.iter().enumerate() {
        let mut error = Float::with_val(bits, Float::parse(value.as_str().unwrap())?);
        error.set_prec(reference_bits);
        error -= &reference_values[i];
        error.abs_mut();
        errors.push(xc_numerics::prefix::lossless_decimal(&error));
        if error > maximum_spectrum_error {
            maximum_spectrum_error = error;
        }
    }
    ensure!(
        maximum_spectrum_error <= tol,
        "spectrum disagrees with exact analytic eigenvalues"
    );
    let final_third = prefix["ladder"]["rows"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .get("third_inverse_moment")
        .cloned();
    ensure!(
        final_third.is_some() == diagnostics.third_inverse_moment,
        "third-moment capture does not match the requested policy"
    );
    let cube_error = if let Some(third) = &final_third {
        let mut expected = Float::with_val(reference_bits, 0);
        for value in &reference_values {
            let mut inverse = Float::with_val(reference_bits, 1) / value;
            inverse = inverse.clone().square() * inverse;
            expected += inverse;
        }
        let mut actual = Float::with_val(
            bits,
            Float::parse(third["inverse_cube_trace"].as_str().unwrap())?,
        );
        actual.set_prec(reference_bits);
        let error = (actual - expected).abs();
        ensure!(
            error <= tol,
            "third inverse moment disagrees with analytic reference"
        );
        Some(xc_numerics::prefix::lossless_decimal(&error))
    } else {
        None
    };
    let checkpoint = &cold.value.measurements[&format!("prefix_checkpoint_{n}")].value;
    ensure!(
        checkpoint["status"] == "export_checks_passed",
        "checkpoint exports unresolved"
    );
    let overlap = Float::with_val(
        reference_bits,
        Float::parse(checkpoint["squared_overlap"].as_str().unwrap())?,
    );
    ensure!((0..=1).contains(&overlap), "invalid overlap");
    let overlap_error = if fixture == "rank-one" {
        // The LDLT innovation differs from the top eigenvector. Its squared
        // normalized overlap is 1/(4n-3), not 1.
        let error = Float::with_val(
            reference_bits,
            overlap
                - Float::with_val(reference_bits, 1) / Float::with_val(reference_bits, 4 * n - 3),
        )
        .abs();
        ensure!(
            error <= tol,
            "innovation/eigenvector overlap differs from analytic reference"
        );
        Some(xc_numerics::prefix::lossless_decimal(&error))
    } else {
        None
    };
    // Separate replay for profiling; timings never enter numerical identities.
    let stage_seconds = pool.install(|| -> anyhow::Result<_> {
        let start = Instant::now();
        let options = plan.prefix_options(bits)?.unwrap();
        xc_spectral::ccm::prefix::analyze_retained_prefixes(&matrix, &options, &eigenpairs)?;
        let prefix_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let (d, e, q) = xc_numerics::eigen::householder_tridiag_hp_stable(matrix.entries(), n, bits)?;
        let householder_seconds = start.elapsed().as_secs_f64();
        let nonzero_subdiagonals = e.iter().filter(|v| !v.is_zero()).count();
        if fixture == "dense-reflector" {
            ensure!(nonzero_subdiagonals == n - 1, "dense reduction has an unexpected split");
        }
        let start = Instant::now();
        xc_numerics::eigen::assess_symmetric_reduction_hp(matrix.entries(), &d, &e, &q, bits)?;
        let assessment_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        xc_numerics::eigen::tridiag_eigenvalues_hp(&d, &e, bits)?;
        Ok(json!({"prefix":prefix_seconds,"householder":householder_seconds,"assessment":assessment_seconds,"qr":start.elapsed().as_secs_f64(),"nonzero_subdiagonals":nonzero_subdiagonals}))
    })?;
    let peak_rss_bytes = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version":1,"toolkit_version":env!("CARGO_PKG_VERSION"),"fixture":fixture,"diagnostic_mode":diagnostic_mode,"synthetic":true,
            "source_nonzero_entries":nonzero_entries,
            "dimension":n,"precision_bits":bits,"workers":workers,"preparation_seconds":preparation_seconds,"cold_seconds":cold_seconds,"warm_seconds":warm_seconds,
            "peak_rss_bytes":peak_rss_bytes,"receipt_digest":xc_core::research_digest(&cold.value)?.0,"warm_byte_identity":warm_byte_identity,
            "checkpoint":{
                "status":checkpoint["status"],"accepted_significant_digits":checkpoint["accepted_significant_digits"],
                "squared_overlap":checkpoint["squared_overlap"],"decoded_eigenpair_backward_error":checkpoint["decoded_eigenpair_backward_error"],
                "decoded_innovation_backward_error":checkpoint["decoded_innovation_backward_error"],"sign_convention":checkpoint["sign_convention"]
            },"checkpoint_overlap_absolute_error":overlap_error,"stage_replay_seconds":stage_seconds,
            "spectrum_values_checked":spectrum.len(),"spectrum_absolute_errors":errors,"reference_precision_bits":reference_bits,
            "spectrum_error_interpretation":"Difference of computed point eigenvalues from analytic references; exact zero is possible and is not a certified error bound.",
            "third_inverse_moment":final_third,"inverse_cube_trace_absolute_error":cube_error,"inverse_trace_absolute_error":xc_numerics::prefix::lossless_decimal(&trace_error),"maximum_spectrum_absolute_error":xc_numerics::prefix::lossless_decimal(&maximum_spectrum_error),
            "relative_similarity_residual":reduction["diagnostics"]["relative_similarity_residual"],"relative_orthogonality_residual":reduction["diagnostics"]["relative_orthogonality_residual"]
        }))?
    );
    drop(resolver);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("qualify_retained_evidence requires --features hp");
    std::process::exit(2);
}
