#![cfg(feature = "hp")]

use rug::{float::Constant, Float, Rational};
use serde_json::json;
use std::collections::BTreeMap;
use xc_cache::*;
use xc_spectral::ccm::{extended_research::*, state_geometry::RetainedState};

fn state(coefficients: &[i32], p: u32) -> (RetainedState, ContentDigest) {
    let bytes = serde_json::to_vec(&json!({
        "schema_version":3,"lambda_squared":"9","n_modes":coefficients.len()/2,
        "precision_bits":p,"force_even":false,"eigenvalue":"0",
        "eigenvector":coefficients.iter().map(i32::to_string).collect::<Vec<_>>()
    }))
    .unwrap();
    let digest = ContentDigest::sha256(&bytes);
    let manifest = ArtifactManifest {
        schema_version: 1,
        key: ArtifactKey::new("ccm_weil_eigenpair", "fresh-math-oracle", &bytes).unwrap(),
        content_digest: digest.clone(),
        size_bytes: bytes.len() as u64,
        objects: vec![CacheObjectRef {
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
        }],
        created_unix_seconds: 1,
        producer_toolkit_version: ToolkitVersion::parse("0.15.2").unwrap(),
        minimum_reader_version: ToolkitVersion::parse("0.15.2").unwrap(),
        maximum_reader_version: None,
        quality: CacheQuality::Validated,
        visibility: CacheVisibility::Local,
        immutable: true,
        dependencies: vec![],
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    (
        RetainedState::from_payload(&manifest, &bytes, std::slice::from_ref(&digest)).unwrap(),
        digest,
    )
}

fn context() -> ArtifactCacheContext<'static> {
    ArtifactCacheContext {
        resolver: None,
        reference_resolver: None,
        acceptance: None,
        ordered_overlays: vec!["disabled".into()],
        mode: ArtifactExecutionCacheMode::Disabled,
        write_on_miss: false,
        write_visibility: CacheVisibility::Local,
        requested_assurance: xc_core::AssuranceLevel::Computed,
        certification_failure_policy: CertificationFailurePolicy::RetainComputedFailRun,
        production_sink: None,
    }
}

fn number(text: &str, p: u32) -> Float {
    Float::with_val(p, Float::parse(text).unwrap())
}

#[test]
fn weighted_fourier_norm_matches_independent_pairwise_defining_integrals() {
    // Directly integrate every pair of basis functions. The production code
    // instead uses lag correlations and normalized recurrence intermediates.
    // For k != 0, integration of exp(2a|x|)*exp(2pi*i*k*(x/L+1/2))
    // over [-L/2,L/2], divided by L, is
    // 4a*(9^a-(-1)^k)/(L*((2a)^2+(2pi*k/L)^2)).
    let work = 1024;
    let l = Float::with_val(work, 9).ln();
    for p in [64, 128, 257] {
        for coefficients in [[0, 0, 1, 0, 0], [1, -3, 7, 2, -1], [2, 1, -5, 1, 2]] {
            let (s, _) = state(&coefficients, p);
            let mut options = ExtensionOptions::for_source(&s);
            options.working_precision_bits = p;
            options.exponential_rates = vec!["0".into(), "1".into(), "2".into()];
            let report = capture_extended(
                "compactness",
                &s,
                None,
                None,
                None,
                &options,
                &[],
                &context(),
            )
            .unwrap()
            .value
            .data;
            let norm_squared: i32 = coefficients.iter().map(|v| v * v).sum();
            for (row, a) in report.rows.iter().zip([0, 1, 2]) {
                let mut expected = Float::with_val(work, 0);
                for (j, x) in coefficients.iter().enumerate() {
                    for (k, y) in coefficients.iter().enumerate() {
                        let lag = j.abs_diff(k);
                        let integral = if a == 0 {
                            Float::with_val(work, u32::from(lag == 0))
                        } else if lag == 0 {
                            Float::with_val(work, 9i32.pow(a as u32) - 1)
                                / (Float::with_val(work, &l) * a)
                        } else {
                            let frequency =
                                Float::with_val(work, Constant::Pi) * (2 * lag as u32) / &l;
                            let denominator = (frequency.square() + 4 * a * a) * &l;
                            Float::with_val(
                                work,
                                4 * a * (9i32.pow(a as u32) - if lag % 2 == 0 { 1 } else { -1 }),
                            ) / denominator
                        };
                        expected += integral * (x * y);
                    }
                }
                expected /= norm_squared;
                assert_eq!(
                    number(&row.values["analytic_integral"], p),
                    Float::with_val(p, expected),
                    "p={p}, rate={a}, coefficients={coefficients:?}"
                );
            }
        }
    }
}

#[test]
fn finite_signed_atoms_match_exact_rational_cutoff_and_inverse_moments() {
    for p in [64, 128, 257] {
        let (s, digest) = state(&[0, 1, 0], p);
        let input: ExternalResearchInputs = serde_json::from_value(json!({
            "schema_version":1,"source_eigenpair":digest,"lambda_squared":"9",
            "n_modes":1,"precision_bits":p,"convention_id":"fresh exact rational atom control",
            "definition_digest":ContentDigest::sha256(b"signed finite measure oracle"),
            "approximation_scope":"finite supplied points only",
            "atoms":[
                {"ordinal":1,"coordinate":"1","weight":"5","family":"zero","partition":"oracle"},
                {"ordinal":2,"coordinate":"2","weight":"-3","family":"zero","partition":"oracle"},
                {"ordinal":3,"coordinate":"2","weight":"1","family":"zero","partition":"oracle"},
                {"ordinal":4,"coordinate":"4","weight":"-2","family":"zero","partition":"oracle"}
            ],
            "atom_coordinate":"exact positive point","atom_coverage":"four supplied atoms",
            "tail_checkpoints":["0","1","2","4"]
        }))
        .unwrap();
        let mut options = ExtensionOptions::for_source(&s);
        options.working_precision_bits = p;
        let report = capture_extended(
            "weighted_tail",
            &s,
            None,
            None,
            Some(&input),
            &options,
            &[],
            &context(),
        )
        .unwrap()
        .value
        .data;
        assert_eq!(report.rows.len(), 4);
        let atoms: [(i32, i32); 4] = [(1, 5), (2, -3), (2, 1), (4, -2)];
        for (row, cutoff) in report.rows.iter().zip([0, 1, 2, 4]) {
            let included = atoms.iter().filter(|(coordinate, _)| *coordinate <= cutoff);
            let mass: i32 = included.clone().map(|(_, weight)| *weight).sum();
            let absolute: i32 = included.clone().map(|(_, weight)| weight.abs()).sum();
            assert_eq!(number(&row.values["included_mass"], p), mass);
            assert_eq!(number(&row.values["included_absolute_mass"], p), absolute);
            assert_eq!(number(&row.values["remaining_supplied_mass"], p), 1 - mass);
            assert_eq!(
                number(&row.values["included_count"], p),
                included.clone().count()
            );
            for power in 1..=3 {
                let exact = included
                    .clone()
                    .fold(Rational::from(0), |sum, (x, weight)| {
                        sum + Rational::from((*weight, i32::pow(*x, power)))
                    });
                assert_eq!(
                    number(&row.values[&format!("weighted_inverse_moment_{power}")], p),
                    Float::with_val(p, exact)
                );
            }
        }
    }
}
