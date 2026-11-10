use super::*;
use std::io::{Cursor, Write};
use std::sync::OnceLock;

fn digest(label: &str) -> ContentDigest {
    ContentDigest::sha256(label.as_bytes())
}
fn fixture() -> PortableCcmSectorGapCertificate {
    static FIXTURE: OnceLock<PortableCcmSectorGapCertificate> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let precision_bits = 256;
            let matrix = crate::ccm::cutoff_free::assemble(
                &crate::ccm::cutoff_free::CutoffFreeConfig::new(5, 2, precision_bits),
            )
            .unwrap();
            super::super::build_certificate_from_tau(
                "5".into(),
                5,
                2,
                precision_bits,
                matrix.config.geometric_terms,
                matrix.scalar_backend.clone(),
                matrix.component_evidence_digest().unwrap(),
                matrix.portable_inertia_certificate().unwrap(),
                &Float::with_val(precision_bits, 1),
                &Float::with_val(precision_bits, 2),
                &Float::with_val(precision_bits, 3),
                digest("compact-independent-even"),
                digest("compact-independent-odd"),
                digest("compact-independent-gap"),
                CcmSectorGapCertificationOptions::default(),
            )
            .unwrap()
        })
        .clone()
}
fn compact() -> CompactCcmSectorGapCertificate {
    CompactCcmSectorGapCertificate::from_portable(
        &fixture(),
        ReconstructionSourceIdentity::current("qualification-dirty-workspace"),
    )
    .unwrap()
}

#[test]
fn compact_run_handle_reuses_only_exact_payload_and_public_replay_stays_fresh() {
    let c = compact();
    let resources = xc_core::RunResources::new(8 << 30).unwrap();
    let _resources = resources.enter();
    let memo = retained::Memo::new(resources.clone());
    let _memo = memo.enter();
    let before = retained::FRESH_REPLAYS.with(std::cell::Cell::get);
    let first = retained::reconstruct(&c).unwrap();
    let second = retained::reconstruct(&c).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 1
    );
    assert!(resources.usage().retained_bytes > 0);
    assert_eq!(
        reconstruct_verified_compact_sector_gap(&c)
            .unwrap()
            .canonical_tau(),
        first.canonical_tau()
    );
    assert!(verify_compact_ccm_sector_gap_certificate(&c).valid);
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 3
    );
    let replayed = retained::freshly(|| retained::reconstruct(&c)).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first, &replayed));
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 4
    );
    // Provenance participates in exact payload binding even though it is not a
    // mathematical premise. A changed compatible payload gets its own replay.
    let mut other = c.clone();
    other
        .source_identity
        .toolkit_revision
        .push_str("-different-retained-snapshot");
    let changed = retained::reconstruct(&other).unwrap();
    assert_eq!(changed.canonical_tau(), first.canonical_tau());
    assert!(!std::sync::Arc::ptr_eq(&first, &changed));
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 5
    );
    let mut wrong = other.clone();
    wrong.cutoff_free_tau_content_digest = digest("tampered canonical matrix");
    assert!(retained::reconstruct(&wrong).is_err());
    assert!(retained::reconstruct(&wrong).is_err());
    // Failed proofs are never retained; eviction releases only memo ownership.
    let last = retained::reconstruct(&c).unwrap();
    let held_bytes = resources.usage().retained_bytes;
    memo.eviction_callback()();
    assert_eq!(resources.usage().retained_bytes, held_bytes);
    drop((first, second, changed, last, replayed));
    assert_eq!(resources.usage().retained_bytes, 0);
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
fn compact_run_handle_rejects_changed_exponent_context_and_recovers_after_budget_refusal() {
    unsafe extern "C" {
        fn mpfr_set_emin(value: std::os::raw::c_long) -> i32;
    }
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            assert_eq!(unsafe { mpfr_set_emin(self.0 as _) }, 0);
        }
    }
    let c = compact();
    let resources = xc_core::RunResources::new(8 << 30).unwrap();
    let _resources = resources.enter();
    let memo = retained::Memo::new(resources.clone());
    let _memo = memo.enter();
    let pressure = resources
        .try_reserve_retained(resources.maximum_bytes() - 1)
        .unwrap();
    assert!(retained::reconstruct(&c)
        .err()
        .unwrap()
        .to_string()
        .contains("budget"));
    drop(pressure);
    let first = retained::reconstruct(&c).unwrap();
    {
        let p = c
            .precision_bits
            .max(c.even_ground.inertia_precision_bits)
            .max(c.even_first_excited.inertia_precision_bits)
            .max(c.odd_ground.inertia_precision_bits);
        let scratch = super::super::certificate_workspace_bytes(
            c.n_modes,
            p,
            c.integer_cutoff_c,
            c.geometric_terms,
        )
        .unwrap();
        let constrained =
            xc_core::RunResources::new(scratch + first._retention.bytes() - 1).unwrap();
        let _constrained = constrained.enter();
        // Scratch fits, but transferring the returned matrix cannot fit. The
        // public API must fail rather than return an uncharged live allocation.
        assert!(reconstruct_verified_compact_sector_gap(&c)
            .err()
            .unwrap()
            .to_string()
            .contains("budget"));
        assert_eq!(constrained.usage().retained_bytes, 0);
        assert_eq!(constrained.usage().working_bytes, 0);
    }
    let before = retained::FRESH_REPLAYS.with(std::cell::Cell::get);
    {
        let _restore = Restore(rug::float::exp_min());
        assert_eq!(unsafe { mpfr_set_emin(-100000) }, 0);
        // A new environment must replay (and may fail the uniform-worker gate).
        // Either outcome is acceptable; reusing the old handle is not.
        if let Ok(changed) = retained::reconstruct(&c) {
            assert!(!std::sync::Arc::ptr_eq(&first, &changed));
        }
        assert_eq!(
            retained::FRESH_REPLAYS.with(std::cell::Cell::get),
            before + 1
        );
    }
    let restored = retained::reconstruct(&c).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first, &restored));
    assert_eq!(first.canonical_tau(), restored.canonical_tau());
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 2
    );
    drop((first, restored));
    memo.eviction_callback()();
    assert_eq!(resources.usage().retained_bytes, 0);
    assert_eq!(resources.usage().working_bytes, 0);
}
fn zip_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(6))
        .last_modified_time(zip::DateTime::default())
        .unix_permissions(0o644);
    zip.start_file("payload.json", options).unwrap();
    zip.write_all(bytes).unwrap();
    zip.finish().unwrap().into_inner()
}
#[test]
fn compact_real_analytic_replay_preserves_assertions_and_measures_storage() {
    let full = fixture();
    let compact = compact();
    let original = verify_portable_ccm_sector_gap_certificate_with_reassembly(&full);
    assert!(original.valid, "{:?}", original.errors);
    let verified = reconstruct_verified_compact_sector_gap(&compact).unwrap();
    assert_eq!(verified.canonical_tau(), full.cutoff_free_tau);
    assert_eq!(verified.certificate(), &compact);
    let report = verify_compact_ccm_sector_gap_certificate(&compact);
    assert!(
        report.valid && report.mathematical_claim_verified,
        "{:?}",
        report.errors
    );
    assert_eq!(
        compact.even_ground,
        CompactSelectedEigenvalue::from_selected(&full.even_ground).unwrap()
    );
    assert_eq!(
        compact.even_first_excited,
        CompactSelectedEigenvalue::from_selected(&full.even_first_excited).unwrap()
    );
    assert_eq!(
        compact.odd_ground,
        CompactSelectedEigenvalue::from_selected(&full.odd_ground).unwrap()
    );
    assert_eq!(
        compact.full_positive,
        full.full_matrix_inertia_certificate.positive
    );
    assert_eq!(
        compact.full_negative,
        full.full_matrix_inertia_certificate.negative
    );
    let full_bytes = serde_json::to_vec(&full).unwrap();
    let compact_bytes = serde_json::to_vec(&compact).unwrap();
    let fzip = zip_bytes(&full_bytes);
    let czip = zip_bytes(&compact_bytes);
    assert!(compact_bytes.len() < full_bytes.len());
    assert!(czip.len() < fzip.len());
    let tree = serde_json::to_value(&compact).unwrap();
    fn no_arrays(value: &serde_json::Value) {
        match value {
            serde_json::Value::Array(_) => panic!("compact certificate contains an array"),
            serde_json::Value::Object(map) => {
                for child in map.values() {
                    no_arrays(child)
                }
            }
            _ => {}
        }
    }
    no_arrays(&tree);
    assert_eq!(
        serde_json::from_slice::<CompactCcmSectorGapCertificate>(&compact_bytes).unwrap(),
        compact
    );
    println!(
        "{}",
        serde_json::json!({"qualification":"compact_real_analytic","c":5,"n":2,"precision_bits":256,"full_json_bytes":full_bytes.len(),"compact_json_bytes":compact_bytes.len(),"full_deflate6_zip_bytes":fzip.len(),"compact_deflate6_zip_bytes":czip.len(),"full_json_sha256":ContentDigest::sha256(&full_bytes),"compact_json_sha256":ContentDigest::sha256(&compact_bytes),"scope":"small newly generated analytic finite proof; no large-N extrapolation; ZIP comparison uses identical payload.json name and options"})
    );
}
#[test]
fn compact_scalar_tampering_and_malformed_schema_are_rejected() {
    let c = compact();
    let mut cases = Vec::new();
    let mut m = c.clone();
    m.schema_version += 1;
    cases.push(m);
    let mut m = c.clone();
    m.n_modes = usize::MAX;
    cases.push(m);
    let mut m = c.clone();
    m.precision_bits = 1_000_001;
    cases.push(m);
    let mut m = c.clone();
    m.reconstruction_semantics = "foreign".into();
    cases.push(m);
    let mut m = c.clone();
    m.assembly_semantics = "foreign".into();
    cases.push(m);
    let mut m = c.clone();
    m.full_inertia_semantics = "foreign".into();
    cases.push(m);
    let mut m = c.clone();
    m.selected_inertia_semantics = "foreign".into();
    cases.push(m);
    let mut m = c.clone();
    m.parity_invariance_premise = "none".into();
    cases.push(m);
    let mut m = c.clone();
    m.claim_scope = "infinite operator".into();
    cases.push(m);
    let mut m = c.clone();
    m.even_ground.lower_count_below += 1;
    cases.push(m);
    let mut m = c.clone();
    m.full_positive += 1;
    cases.push(m);
    let mut m = c.clone();
    m.even_simplicity_gap_lower.numerator = "-1".into();
    cases.push(m);
    let mut m = c.clone();
    m.certified_finite_ground_parity = "odd".into();
    cases.push(m);
    let mut m = c.clone();
    m.certifies_finite_ground_state_simple = false;
    cases.push(m);
    let mut m = c.clone();
    m.source_identity.build_identity.clear();
    cases.push(m);
    for m in cases {
        assert!(m.validate_structure().is_err());
        assert!(!verify_compact_ccm_sector_gap_certificate(&m).valid);
    }
    let mut value = serde_json::to_value(&c).unwrap();
    value["extra_matrix"] = serde_json::json!([]);
    assert!(serde_json::from_value::<CompactCcmSectorGapCertificate>(value).is_err());
    let bytes = serde_json::to_vec(&c).unwrap();
    assert!(
        serde_json::from_slice::<CompactCcmSectorGapCertificate>(&bytes[..bytes.len() / 2])
            .is_err()
    );
    let mut old = fixture();
    old.schema_version = 3;
    assert!(
        CompactCcmSectorGapCertificate::from_portable(&old, c.source_identity.clone()).is_err()
    );
    let mut unresolved = fixture();
    unresolved
        .full_matrix_inertia_certificate
        .zero_or_unresolved = 1;
    assert!(CompactCcmSectorGapCertificate::from_portable(&unresolved, c.source_identity).is_err());
}
#[test]
fn compact_fresh_assembly_rejects_hash_backend_configuration_and_valid_count_changes() {
    let c = compact();
    let mut cases = Vec::new();
    let mut m = c.clone();
    m.raw_matrix_content_digest = digest("foreign raw");
    cases.push(m);
    let mut m = c.clone();
    m.component_evidence_digest = digest("foreign components");
    cases.push(m);
    let mut m = c.clone();
    m.cutoff_free_tau_content_digest = digest("foreign canonical");
    cases.push(m);
    let mut m = c.clone();
    m.scalar_backend = "foreign backend".into();
    cases.push(m);
    let mut m = c.clone();
    m.integer_cutoff_c = 6;
    cases.push(m);
    let mut m = c.clone();
    m.even_ground.matrix_digest = digest("foreign even");
    m.even_first_excited.matrix_digest = m.even_ground.matrix_digest.clone();
    cases.push(m);
    let mut m = c.clone();
    m.odd_ground.matrix_digest = digest("foreign odd");
    cases.push(m);
    let mut m = c.clone();
    if m.full_positive > 0 {
        m.full_positive -= 1;
        m.full_negative += 1;
    } else {
        m.full_negative -= 1;
        m.full_positive += 1;
    }
    m.certifies_finite_matrix_positive_definite = m.full_positive == 5;
    cases.push(m);
    for m in cases {
        m.validate_structure().unwrap();
        assert!(!verify_compact_ccm_sector_gap_certificate(&m).valid);
    }
}
#[test]
fn compact_provenance_is_explicitly_not_executable_attestation() {
    let mut c = compact();
    c.source_identity = ReconstructionSourceIdentity {
        toolkit_revision: "alternate producer provenance".into(),
        source_tree_digest: digest("unattested executable"),
        build_identity: "unattested label".into(),
    };
    let report = verify_compact_ccm_sector_gap_certificate(&c);
    assert!(report.valid, "{:?}", report.errors);
    assert!(report
        .warnings
        .iter()
        .any(|s| s.contains("not the producer executable")));
}
#[test]
fn compact_structure_consistent_wrong_bracket_fails_fresh_selected_proof() {
    let mut c = compact();
    c.even_ground.lower = rational_record(&(parse(&c.even_ground.lower).unwrap() - 100));
    c.even_ground.upper = rational_record(&(parse(&c.even_ground.upper).unwrap() - 100));
    c.even_simplicity_gap_lower = rational_record(
        &(parse(&c.even_first_excited.lower).unwrap() - parse(&c.even_ground.upper).unwrap()),
    );
    c.even_odd_ordering_gap_lower = rational_record(
        &(parse(&c.odd_ground.lower).unwrap() - parse(&c.even_ground.upper).unwrap()),
    );
    c.even_odd_ordering_gap_upper = rational_record(
        &(parse(&c.odd_ground.upper).unwrap() - parse(&c.even_ground.lower).unwrap()),
    );
    c.certified_finite_ground_parity = "even".into();
    c.certifies_finite_ground_state_simple = true;
    c.validate_structure().unwrap();
    assert!(!verify_compact_ccm_sector_gap_certificate(&c).valid);
}
#[test]
fn compact_consumers_match_full_arithmetic_and_bind_actual_certificate_identity() {
    use crate::ccm::{
        sector_gap_certificate::{root_budget, SectorGapCertificateRef},
        state_geometry::RetainedState,
        transform_enclosure,
    };
    let full = fixture();
    let compact = compact();
    let dimension = 2 * full.n_modes + 1;
    let midpoint = parse_records(&full.cutoff_free_tau)
        .unwrap()
        .into_iter()
        .map(|x| ((x.lower().clone() + x.upper()) / 2_i32).to_f64())
        .collect::<Vec<_>>();
    let eigen = nalgebra::SymmetricEigen::new(nalgebra::DMatrix::from_row_slice(
        dimension, dimension, &midpoint,
    ));
    let index = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let version = serde_json::json!({"major":0,"minor":17,"patch":1,"prerelease":null});
    let manifest = serde_json::from_value(serde_json::json!({
        "schema_version":1,"key":{"kind":"ccm_weil_eigenpair","logical_key":"compact analytic consumer equality","parameters_digest":digest("consumer key")},
        "content_digest":digest("consumer state"),"size_bytes":0,"objects":[],"created_unix_seconds":1,
        "producer_toolkit_version":version,"minimum_reader_version":version,"maximum_reader_version":null,"quality":"validated","visibility":"local","immutable":true,"dependencies":[],"tags":{},"provenance_digest":null
    })).unwrap();
    let state = RetainedState {
        manifest,
        cutoff: "5".into(),
        modes: 2,
        precision: 256,
        coefficients: eigen
            .eigenvectors
            .column(index)
            .iter()
            .map(|v| Float::with_val(256, *v))
            .collect(),
        eigenvalue: Float::with_val(256, eigen.eigenvalues[index]).to_string(),
        selection_policy: Some(
            "numerical trial vector; analytic residual independently bounded".into(),
        ),
    };
    let resources = xc_core::RunResources::new(8 << 30).unwrap();
    let _resources = resources.enter();
    let memo = retained::Memo::new(resources);
    let _memo = memo.enter();
    let before = retained::FRESH_REPLAYS.with(std::cell::Cell::get);
    let error_full = transform_enclosure::verified_state_error_ref(
        &state,
        SectorGapCertificateRef::Full(&full),
        256,
    )
    .unwrap();
    let error_compact = transform_enclosure::verified_state_error_ref(
        &state,
        SectorGapCertificateRef::Compact(&compact),
        256,
    )
    .unwrap();
    assert_eq!(error_full.lower(), error_compact.lower());
    assert_eq!(error_full.upper(), error_compact.upper());
    assert!(error_compact.contains_zero());
    let options = crate::ccm::extended_research::ExtensionOptions::for_source(&state);
    let full_transform =
        transform_enclosure::analyze(&state, None, &options, None, Some(&full), 256).unwrap();
    let compact_transform = transform_enclosure::analyze_with_certificate_ref(
        &state,
        None,
        &options,
        None,
        Some(SectorGapCertificateRef::Compact(&compact)),
        256,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(full_transform).unwrap(),
        serde_json::to_value(compact_transform).unwrap()
    );
    let request = root_budget::Request {
        precision_bits: 256,
        windows: vec![root_budget::Window {
            requested_index: 1,
            bracket: xc_solver::convergence::ExactBounds {
                lower: "1".into(),
                upper: "2".into(),
            },
            center: "1.5".into(),
        }],
    };
    let token = xc_core::CancellationToken::new();
    let full_report = root_budget::analyze(&state, &full, &request, &token).unwrap();
    let compact_report = root_budget::analyze_with_certificate_ref(
        &state,
        SectorGapCertificateRef::Compact(&compact),
        &request,
        &token,
    )
    .unwrap();
    assert_eq!(
        full_report.unit_state_l2_error_upper,
        compact_report.unit_state_l2_error_upper
    );
    assert_eq!(full_report.rows, compact_report.rows);
    assert_eq!(
        compact_report.matrix_certificate_digest,
        ContentDigest::sha256_finite_json(&compact).unwrap()
    );
    assert_ne!(
        full_report.matrix_certificate_digest,
        compact_report.matrix_certificate_digest
    );
    assert_ne!(full_report.request_digest, compact_report.request_digest);
    assert_eq!(full_report.semantics, "ccm-analytic-local-root-budget-v1");
    assert_eq!(
        compact_report.semantics,
        "ccm-analytic-local-root-budget-compact-v1"
    );
    // State-error, transform and root-budget consumers share one reconstructed
    // matrix without changing any directed endpoints or primary numerical rows.
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 1
    );
    root_budget::verify_with_certificate_ref(
        &state,
        SectorGapCertificateRef::Compact(&compact),
        &request,
        &compact_report,
        &token,
    )
    .unwrap();
    assert_eq!(
        retained::FRESH_REPLAYS.with(std::cell::Cell::get),
        before + 2
    );
    assert!(root_budget::verify(&state, &full, &request, &compact_report, &token).is_err());
    let mut wrong = compact.clone();
    wrong.raw_matrix_content_digest = digest("tampered source matrix");
    assert!(transform_enclosure::verified_state_error_ref(
        &state,
        SectorGapCertificateRef::Compact(&wrong),
        256
    )
    .is_err());
    assert!(root_budget::analyze_with_certificate_ref(
        &state,
        SectorGapCertificateRef::Compact(&wrong),
        &request,
        &token
    )
    .is_err());
    let mut wrong_state = state.clone();
    wrong_state.cutoff = "6".into();
    assert!(transform_enclosure::verified_state_error_ref(
        &wrong_state,
        SectorGapCertificateRef::Compact(&compact),
        256
    )
    .is_err());
    // Passing the larger-N admission gate never substitutes for certificate
    // authentication: a relabeled small proof cannot certify a larger source.
    let mut larger_state = state.clone();
    larger_state.modes = 513;
    larger_state
        .coefficients
        .resize(1027, Float::with_val(256, 0));
    let mut relabeled = compact.clone();
    relabeled.n_modes = 513;
    let error = root_budget::analyze_with_certificate_ref(
        &larger_state,
        SectorGapCertificateRef::Compact(&relabeled),
        &request,
        &token,
    )
    .unwrap_err()
    .to_string();
    assert!(!error.contains("root budget dimension, precision or row limit"));
    assert!(error.contains("compact"), "{error}");
}
