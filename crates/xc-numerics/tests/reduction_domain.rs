#![cfg(feature = "hp")]
use rug::Float;
use std::collections::{BTreeMap, BTreeSet};
use xc_core::{
    ConfigDigest, DecimalLiteral, DeterministicReductionPolicy, ExecutionFingerprint,
    PrecisionFingerprint, Reproducibility, ThreadPolicyFingerprint,
    DETERMINISTIC_REDUCTION_SCHEDULING_V1,
};
use xc_numerics::reduction::{
    compare_hp_reduction_artifacts, deterministic_parallel_sum_hp, HpReductionEquivalenceCriterion,
};
fn fingerprint(thread_count: usize, policy: &DeterministicReductionPolicy) -> ExecutionFingerprint {
    ExecutionFingerprint {
        schema_version: 1,
        toolkit_revision: "reduction-fixture-v1".to_owned(),
        dependency_revisions: BTreeMap::from([("Cargo.lock".to_owned(), "fixture".to_owned())]),
        compiler: "rustc-fixture".to_owned(),
        target_triple: "x86_64-unknown-linux-gnu".to_owned(),
        native_libraries: BTreeMap::from([("mpfr".to_owned(), "fixture".to_owned())]),
        scalar_backend: "rug_mpfr".to_owned(),
        scalar_backend_version: "fixture".to_owned(),
        precision: PrecisionFingerprint {
            working_precision_bits: 256,
            guard_bits: 32,
            rounding_policy: "nearest".to_owned(),
        },
        algorithm_semantics_versions: BTreeMap::from([(
            "parallel_reduction".to_owned(),
            "deterministic-indexed-chunks-pairwise-v1".to_owned(),
        )]),
        cpu_feature_policy: "portable".to_owned(),
        thread_policy: ThreadPolicyFingerprint {
            thread_count,
            scheduling_policy: DETERMINISTIC_REDUCTION_SCHEDULING_V1.to_owned(),
            reduction_policy: policy.fingerprint_name(),
        },
        feature_flags: BTreeSet::from(["hp".to_owned()]),
        effective_configuration_digest: ConfigDigest("a".repeat(64)),
        resolved_resource_policy_digest: ConfigDigest("b".repeat(64)),
        reproducibility: Reproducibility::Bitwise,
    }
}

#[test]
fn rounded_difference_must_not_pass_an_exact_absolute_tolerance() {
    let policy = DeterministicReductionPolicy { chunk_elements: 17 };
    let left_fp = fingerprint(1, &policy);
    let right_fp = fingerprint(2, &policy);
    let (_, left) =
        deterministic_parallel_sum_hp(&[Float::with_val(256, 1)], &policy, &left_fp).unwrap();
    let (_, right) =
        deterministic_parallel_sum_hp(&[-(Float::with_val(256, 1) >> 400_i32)], &policy, &right_fp)
            .unwrap();
    // Exact source difference is 1+2^-400>1. The former p+64 point subtraction
    // rounded to 1 and falsely accepted the comparison.
    assert!(compare_hp_reduction_artifacts(
        &left,
        &left_fp,
        &right,
        &right_fp,
        &HpReductionEquivalenceCriterion {
            absolute_tolerance: DecimalLiteral::new("1").unwrap()
        }
    )
    .is_err());
}
#[test]
fn finite_inputs_that_overflow_cannot_create_reproducible_scalar_evidence() {
    let policy = DeterministicReductionPolicy { chunk_elements: 17 };
    let fp = fingerprint(1, &policy);
    let huge = Float::with_val(256, 1) << (rug::float::exp_max() - 1) as u32;
    assert!(deterministic_parallel_sum_hp(&[huge.clone(), huge], &policy, &fp).is_err());
}
