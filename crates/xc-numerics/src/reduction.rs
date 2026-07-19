//! Deterministic parallel MPFR reductions and cross-fingerprint evidence.

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use rug::Float;
use serde::{Deserialize, Serialize};
use xc_core::{
    CrossFingerprintComparison, DecimalLiteral, DeterministicReductionPolicy, ExecutionFingerprint,
    ReproducibilityComparisonPlan, ReproducibleReductionArtifact,
};

pub const MPFR_DECIMAL_REDUCTION_ENCODING_V1: &str = "mpfr_decimal_roundtrip_significant_digits_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpReductionEquivalenceCriterion {
    pub absolute_tolerance: DecimalLiteral,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpReductionComparison {
    pub plan: ReproducibilityComparisonPlan,
    pub left_payload_sha256: String,
    pub right_payload_sha256: String,
    pub absolute_difference: String,
    pub absolute_tolerance: DecimalLiteral,
    pub accepted: bool,
}

fn roundtrip_decimal_digits(precision_bits: u32) -> usize {
    // ceil(bits * log10(2)) + two guard digits, with 30103/100000 an
    // upward rational bound for log10(2).
    (u64::from(precision_bits)
        .saturating_mul(30_103)
        .div_ceil(100_000)
        .saturating_add(2)) as usize
}

fn encode(value: &Float, precision_bits: u32) -> String {
    value.to_string_radix(10, Some(roundtrip_decimal_digits(precision_bits)))
}

fn parse(value: &str, precision_bits: u32, name: &str) -> Result<Float> {
    let parsed = Float::parse(value).with_context(|| format!("parse {name}"))?;
    let value = Float::with_val(precision_bits, parsed);
    if !value.is_finite() {
        return Err(anyhow!("{name} must be finite"));
    }
    Ok(value)
}

/// Combine an indexed set through the canonical adjacent pairwise tree.
/// Callers may compute the indexed leaves in parallel, but must collect them
/// in source order before invoking this function.
pub fn deterministic_pairwise_sum_hp(values: &[Float], precision_bits: u32) -> Float {
    if values.is_empty() {
        return Float::with_val(precision_bits, 0);
    }
    let mut level: Vec<Float> = values
        .iter()
        .map(|value| Float::with_val(precision_bits, value))
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let mut value = pair[0].clone();
            if let Some(right) = pair.get(1) {
                value += right;
            }
            next.push(value);
        }
        level = next;
    }
    level.pop().expect("nonempty input produces one root")
}

/// Consume MPFR leaves through the same canonical adjacent-pair tree as
/// [`deterministic_pairwise_sum_hp`]. Moving each left leaf into the next
/// level avoids cloning every MPFR allocation while preserving every
/// rounding operation and the complete reduction-tree shape.
pub fn deterministic_pairwise_sum_hp_owned(mut values: Vec<Float>, precision_bits: u32) -> Float {
    if values.is_empty() {
        return Float::with_val(precision_bits, 0);
    }
    for value in &mut values {
        if value.prec() != precision_bits {
            *value = Float::with_val(precision_bits, &*value);
        }
    }
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut leaves = values.into_iter();
        while let Some(mut left) = leaves.next() {
            if let Some(right) = leaves.next() {
                left += &right;
            }
            next.push(left);
        }
        values = next;
    }
    values.pop().expect("nonempty input produces one root")
}

/// Sum MPFR values with parallel fixed-index leaf chunks and a serial,
/// adjacent pairwise tree whose shape is independent of worker scheduling.
pub fn deterministic_parallel_sum_hp(
    values: &[Float],
    policy: &DeterministicReductionPolicy,
    fingerprint: &ExecutionFingerprint,
) -> Result<(Float, ReproducibleReductionArtifact)> {
    fingerprint.validate()?;
    policy.validate()?;
    let precision_bits = fingerprint.precision.working_precision_bits;
    if values.is_empty() {
        return Err(anyhow!(
            "deterministic HP reduction requires at least one value"
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || value.prec() != precision_bits)
    {
        return Err(anyhow!(
            "deterministic HP reduction values must be finite and use the fingerprint precision"
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fingerprint.thread_policy.thread_count)
        .build()
        .context("build deterministic HP reduction thread pool")?;
    let level: Vec<Float> = pool.install(|| {
        values
            .par_chunks(policy.chunk_elements)
            .map(|chunk| {
                let mut subtotal = Float::with_val(precision_bits, 0);
                for value in chunk {
                    subtotal += value;
                }
                subtotal
            })
            .collect()
    });
    let value = deterministic_pairwise_sum_hp(&level, precision_bits);
    let artifact = ReproducibleReductionArtifact::new(
        fingerprint,
        policy.clone(),
        MPFR_DECIMAL_REDUCTION_ENCODING_V1,
        encode(&value, precision_bits),
    )?;
    Ok((value, artifact))
}

/// Verify two saved reductions and apply their declared cross-fingerprint
/// absolute-equivalence criterion entirely in MPFR.
pub fn compare_hp_reduction_artifacts(
    left: &ReproducibleReductionArtifact,
    left_fingerprint: &ExecutionFingerprint,
    right: &ReproducibleReductionArtifact,
    right_fingerprint: &ExecutionFingerprint,
    criterion: &HpReductionEquivalenceCriterion,
) -> Result<HpReductionComparison> {
    left.verify(left_fingerprint)?;
    right.verify(right_fingerprint)?;
    if left.scalar_encoding != MPFR_DECIMAL_REDUCTION_ENCODING_V1
        || right.scalar_encoding != MPFR_DECIMAL_REDUCTION_ENCODING_V1
    {
        return Err(anyhow!("unsupported HP reduction scalar encoding"));
    }
    let plan = left_fingerprint.comparison_plan(right_fingerprint)?;
    let precision_bits = left
        .precision_bits
        .max(right.precision_bits)
        .saturating_add(64);
    let left_value = parse(&left.value, precision_bits, "left HP reduction value")?;
    let right_value = parse(&right.value, precision_bits, "right HP reduction value")?;
    let tolerance = parse(
        criterion.absolute_tolerance.as_str(),
        precision_bits,
        "HP reduction absolute tolerance",
    )?;
    if tolerance < 0 {
        return Err(anyhow!(
            "HP reduction absolute tolerance must be nonnegative"
        ));
    }
    let mut difference = left_value;
    difference -= right_value;
    difference.abs_mut();
    let accepted = match plan.required_comparison {
        CrossFingerprintComparison::ByteIdentity => {
            left.payload_sha256 == right.payload_sha256 && left.value == right.value
        }
        CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure => {
            difference <= tolerance
        }
    };
    if !accepted {
        return Err(anyhow!(
            "HP reduction artifacts do not satisfy their required comparison"
        ));
    }
    Ok(HpReductionComparison {
        plan,
        left_payload_sha256: left.payload_sha256.clone(),
        right_payload_sha256: right.payload_sha256.clone(),
        absolute_difference: encode(&difference, precision_bits),
        absolute_tolerance: criterion.absolute_tolerance.clone(),
        accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;
    use std::collections::{BTreeMap, BTreeSet};
    use xc_core::{
        ConfigDigest, PrecisionFingerprint, Reproducibility, ThreadPolicyFingerprint,
        DETERMINISTIC_REDUCTION_SCHEDULING_V1,
    };

    fn fingerprint(
        thread_count: usize,
        policy: &DeterministicReductionPolicy,
    ) -> ExecutionFingerprint {
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

    fn cancellation_fixture() -> Vec<Float> {
        let precision_bits = 256;
        (0..258)
            .map(|index| {
                let mut value = Float::with_val(precision_bits, index as i64 - 128);
                value *= Float::with_val(precision_bits, 2).pow(-180);
                if index % 3 == 0 {
                    value += 1;
                } else if index % 3 == 1 {
                    value -= 1;
                }
                value
            })
            .collect()
    }

    #[test]
    fn owned_pairwise_reduction_is_bit_identical_to_borrowed_tree() {
        for length in 0..67 {
            let values = cancellation_fixture()
                .into_iter()
                .take(length)
                .collect::<Vec<_>>();
            let borrowed = deterministic_pairwise_sum_hp(&values, 256);
            let owned = deterministic_pairwise_sum_hp_owned(values, 256);
            assert_eq!(borrowed, owned, "tree changed at length {length}");
            assert_eq!(
                encode(&borrowed, 256),
                encode(&owned, 256),
                "serialized value changed at length {length}"
            );
        }
    }

    #[test]
    fn frozen_fingerprint_repeats_identical_hp_value_bytes_and_hash() {
        let policy = DeterministicReductionPolicy { chunk_elements: 17 };
        let fingerprint = fingerprint(4, &policy);
        let values = cancellation_fixture();
        let (first_value, first) =
            deterministic_parallel_sum_hp(&values, &policy, &fingerprint).unwrap();
        let (second_value, second) =
            deterministic_parallel_sum_hp(&values, &policy, &fingerprint).unwrap();

        assert!(!first_value.is_zero());
        assert_eq!(first_value, second_value);
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        first.verify(&fingerprint).unwrap();
        let comparison = compare_hp_reduction_artifacts(
            &first,
            &fingerprint,
            &second,
            &fingerprint,
            &HpReductionEquivalenceCriterion {
                absolute_tolerance: DecimalLiteral::new("0").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(
            comparison.plan.required_comparison,
            CrossFingerprintComparison::ByteIdentity
        );
    }

    #[test]
    fn cross_fingerprint_comparison_uses_declared_mpfr_bound() {
        let policy = DeterministicReductionPolicy { chunk_elements: 17 };
        let left_fingerprint = fingerprint(2, &policy);
        let right_fingerprint = fingerprint(5, &policy);
        let values = cancellation_fixture();
        let (_, left) = deterministic_parallel_sum_hp(&values, &policy, &left_fingerprint).unwrap();
        let (_, right) =
            deterministic_parallel_sum_hp(&values, &policy, &right_fingerprint).unwrap();
        assert_ne!(
            left.execution_fingerprint_digest,
            right.execution_fingerprint_digest
        );
        let comparison = compare_hp_reduction_artifacts(
            &left,
            &left_fingerprint,
            &right,
            &right_fingerprint,
            &HpReductionEquivalenceCriterion {
                absolute_tolerance: DecimalLiteral::new("1e-70").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(
            comparison.plan.required_comparison,
            CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure
        );
        assert!(comparison.accepted);

        let mut outside_bound_values = values.clone();
        outside_bound_values[0] += Float::with_val(256, 2).pow(-100);
        let (_, outside_bound) =
            deterministic_parallel_sum_hp(&outside_bound_values, &policy, &right_fingerprint)
                .unwrap();
        assert!(compare_hp_reduction_artifacts(
            &left,
            &left_fingerprint,
            &outside_bound,
            &right_fingerprint,
            &HpReductionEquivalenceCriterion {
                absolute_tolerance: DecimalLiteral::new("1e-70").unwrap(),
            },
        )
        .is_err());

        let mut tampered = right;
        tampered.value = "1".to_owned();
        assert!(compare_hp_reduction_artifacts(
            &left,
            &left_fingerprint,
            &tampered,
            &right_fingerprint,
            &HpReductionEquivalenceCriterion {
                absolute_tolerance: DecimalLiteral::new("1e-70").unwrap(),
            },
        )
        .is_err());
    }
}
