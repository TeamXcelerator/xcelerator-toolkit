//! Deterministically ordered sequential and parallel operator batches.

use crate::{LinearOperator, OperatorError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use xc_core::{
    CancellationToken, CrossFingerprintComparison, ExecutionFingerprint,
    ReproducibilityComparisonPlan,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorBatchTask<S> {
    pub task_id: String,
    pub input: Vec<S>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorBatchOutcome<S> {
    pub input_ordinal: usize,
    pub task_id: String,
    pub output: Vec<S>,
}

pub fn apply_operator_batch_sequential<S>(
    operator: &dyn LinearOperator<S>,
    tasks: &[OperatorBatchTask<S>],
    cancellation: &CancellationToken,
) -> Result<Vec<OperatorBatchOutcome<S>>, OperatorError>
where
    S: Clone + Default,
{
    validate_tasks(operator.dimension(), tasks)?;
    tasks
        .iter()
        .enumerate()
        .map(|(input_ordinal, task)| apply_one(operator, task, input_ordinal, cancellation))
        .collect()
}

pub fn apply_operator_batch_parallel<S>(
    operator: &dyn LinearOperator<S>,
    tasks: &[OperatorBatchTask<S>],
    cancellation: &CancellationToken,
) -> Result<Vec<OperatorBatchOutcome<S>>, OperatorError>
where
    S: Clone + Default + Send + Sync,
{
    validate_tasks(operator.dimension(), tasks)?;
    let attempts = tasks
        .par_iter()
        .enumerate()
        .map(|(input_ordinal, task)| apply_one(operator, task, input_ordinal, cancellation))
        .collect::<Vec<_>>();
    // Indexed Rayon collection preserves input ordinals. Resolve failures in
    // that same order so worker completion timing is never observable.
    attempts.into_iter().collect()
}

fn validate_tasks<S>(
    dimension: usize,
    tasks: &[OperatorBatchTask<S>],
) -> Result<(), OperatorError> {
    if tasks.is_empty() {
        return Err(OperatorError::InvalidData(
            "operator batch requires at least one task".to_owned(),
        ));
    }
    let mut identifiers = BTreeSet::new();
    for task in tasks {
        if task.task_id.trim().is_empty() || !identifiers.insert(task.task_id.clone()) {
            return Err(OperatorError::InvalidData(
                "operator batch task identifiers must be nonempty and unique".to_owned(),
            ));
        }
        if task.input.len() != dimension {
            return Err(OperatorError::DimensionMismatch {
                expected: dimension,
                actual: task.input.len(),
            });
        }
    }
    Ok(())
}

fn apply_one<S>(
    operator: &dyn LinearOperator<S>,
    task: &OperatorBatchTask<S>,
    input_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<OperatorBatchOutcome<S>, OperatorError>
where
    S: Clone + Default,
{
    cancellation
        .check()
        .map_err(|error| OperatorError::ApplicationFailed(error.to_string()))?;
    let mut output = vec![S::default(); operator.dimension()];
    operator.apply(&task.input, &mut output)?;
    Ok(OperatorBatchOutcome {
        input_ordinal,
        task_id: task.task_id.clone(),
        output,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct F64BatchReproducibilityReport {
    pub comparison_plan: ReproducibilityComparisonPlan,
    pub left_payload_sha256: String,
    pub right_payload_sha256: String,
    pub exact_payload_identity: bool,
    pub maximum_absolute_difference: f64,
    pub declared_absolute_tolerance: f64,
    pub accepted: bool,
}

pub fn compare_f64_operator_batches(
    left: &[OperatorBatchOutcome<f64>],
    left_fingerprint: &ExecutionFingerprint,
    right: &[OperatorBatchOutcome<f64>],
    right_fingerprint: &ExecutionFingerprint,
    declared_absolute_tolerance: f64,
) -> Result<F64BatchReproducibilityReport, OperatorError> {
    if !declared_absolute_tolerance.is_finite() || declared_absolute_tolerance < 0.0 {
        return Err(OperatorError::InvalidData(
            "batch comparison tolerance must be finite and nonnegative".to_owned(),
        ));
    }
    if left.len() != right.len() {
        return Err(OperatorError::DimensionMismatch {
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut maximum_absolute_difference = 0.0f64;
    for (ordinal, (left, right)) in left.iter().zip(right).enumerate() {
        if left.input_ordinal != ordinal
            || right.input_ordinal != ordinal
            || left.task_id != right.task_id
            || left.output.len() != right.output.len()
        {
            return Err(OperatorError::InvalidData(
                "batch comparison requires identical input ordering and output dimensions"
                    .to_owned(),
            ));
        }
        for (&left, &right) in left.output.iter().zip(&right.output) {
            if !left.is_finite() || !right.is_finite() {
                return Err(OperatorError::InvalidData(
                    "batch comparison rejects nonfinite output".to_owned(),
                ));
            }
            maximum_absolute_difference = maximum_absolute_difference.max((left - right).abs());
        }
    }
    let comparison_plan = left_fingerprint
        .comparison_plan(right_fingerprint)
        .map_err(|error| OperatorError::InvalidData(error.to_string()))?;
    let left_payload_sha256 = f64_batch_sha256(left);
    let right_payload_sha256 = f64_batch_sha256(right);
    let exact_payload_identity = left_payload_sha256 == right_payload_sha256;
    let accepted = match comparison_plan.required_comparison {
        CrossFingerprintComparison::ByteIdentity => exact_payload_identity,
        CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure => {
            maximum_absolute_difference <= declared_absolute_tolerance
        }
    };
    Ok(F64BatchReproducibilityReport {
        comparison_plan,
        left_payload_sha256,
        right_payload_sha256,
        exact_payload_identity,
        maximum_absolute_difference,
        declared_absolute_tolerance,
        accepted,
    })
}

fn f64_batch_sha256(batch: &[OperatorBatchOutcome<f64>]) -> String {
    let mut hasher = Sha256::new();
    for outcome in batch {
        hasher.update((outcome.input_ordinal as u64).to_le_bytes());
        hasher.update((outcome.task_id.len() as u64).to_le_bytes());
        hasher.update(outcome.task_id.as_bytes());
        hasher.update((outcome.output.len() as u64).to_le_bytes());
        for value in &outcome.output {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiagonalF64, MatrixStructure, OperatorMetadata};
    use std::collections::{BTreeMap, BTreeSet};
    use xc_core::{
        ConfigDigest, PrecisionFingerprint, Reproducibility, ResourcePolicy,
        ThreadPolicyFingerprint,
    };

    fn fingerprint(thread_count: usize, scheduling: &str) -> ExecutionFingerprint {
        let resources = ResourcePolicy::default();
        ExecutionFingerprint {
            schema_version: 1,
            toolkit_revision: env!("CARGO_PKG_VERSION").to_owned(),
            dependency_revisions: BTreeMap::from([("rayon".to_owned(), "1.11".to_owned())]),
            compiler: "rustc-test".to_owned(),
            target_triple: "x86_64-test".to_owned(),
            native_libraries: BTreeMap::from([("libm".to_owned(), "system".to_owned())]),
            scalar_backend: "f64".to_owned(),
            scalar_backend_version: "ieee754".to_owned(),
            precision: PrecisionFingerprint {
                working_precision_bits: 53,
                guard_bits: 0,
                rounding_policy: "nearest".to_owned(),
            },
            algorithm_semantics_versions: BTreeMap::from([(
                "operator_batch".to_owned(),
                "v1".to_owned(),
            )]),
            cpu_feature_policy: "portable".to_owned(),
            thread_policy: ThreadPolicyFingerprint {
                thread_count,
                scheduling_policy: scheduling.to_owned(),
                reduction_policy: "independent_tasks_input_order".to_owned(),
            },
            feature_flags: BTreeSet::new(),
            effective_configuration_digest: ConfigDigest("a".repeat(64)),
            resolved_resource_policy_digest: resources.digest().unwrap(),
            reproducibility: Reproducibility::Bitwise,
        }
    }

    #[test]
    fn parallel_batch_preserves_order_and_applies_fingerprint_comparison_policy() {
        let operator = DiagonalF64::new("diagonal", vec![2.0, -1.0, 0.5]).unwrap();
        let tasks = vec![
            OperatorBatchTask {
                task_id: "third-label-first".to_owned(),
                input: vec![1.0, 2.0, 3.0],
            },
            OperatorBatchTask {
                task_id: "first-label-second".to_owned(),
                input: vec![-4.0, 5.0, 6.0],
            },
            OperatorBatchTask {
                task_id: "middle-label-last".to_owned(),
                input: vec![7.0, 8.0, -9.0],
            },
        ];
        let cancellation = CancellationToken::new();
        let sequential = apply_operator_batch_sequential(&operator, &tasks, &cancellation).unwrap();
        let parallel = apply_operator_batch_parallel(&operator, &tasks, &cancellation).unwrap();
        assert_eq!(sequential, parallel);
        assert_eq!(
            parallel
                .iter()
                .map(|outcome| outcome.task_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "third-label-first",
                "first-label-second",
                "middle-label-last"
            ]
        );

        let sequential_fingerprint = fingerprint(1, "serial");
        let parallel_fingerprint = fingerprint(4, "rayon_indexed");
        let cross_policy = compare_f64_operator_batches(
            &sequential,
            &sequential_fingerprint,
            &parallel,
            &parallel_fingerprint,
            0.0,
        )
        .unwrap();
        assert!(cross_policy.accepted);
        assert!(cross_policy.exact_payload_identity);
        assert_eq!(
            cross_policy.comparison_plan.required_comparison,
            CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure
        );

        let repeated_parallel =
            apply_operator_batch_parallel(&operator, &tasks, &cancellation).unwrap();
        let same_policy = compare_f64_operator_batches(
            &parallel,
            &parallel_fingerprint,
            &repeated_parallel,
            &parallel_fingerprint,
            0.0,
        )
        .unwrap();
        assert!(same_policy.accepted);
        assert!(same_policy.exact_payload_identity);
        assert_eq!(
            same_policy.comparison_plan.required_comparison,
            CrossFingerprintComparison::ByteIdentity
        );

        let mut numerically_equivalent = parallel.clone();
        numerically_equivalent[1].output[2] += 1.0e-12;
        let bounded_cross_policy = compare_f64_operator_batches(
            &parallel,
            &parallel_fingerprint,
            &numerically_equivalent,
            &sequential_fingerprint,
            1.0e-10,
        )
        .unwrap();
        assert!(bounded_cross_policy.accepted);
        assert!(!bounded_cross_policy.exact_payload_identity);
        assert!(bounded_cross_policy.maximum_absolute_difference > 0.0);
        assert_eq!(
            bounded_cross_policy.comparison_plan.required_comparison,
            CrossFingerprintComparison::DeclaredNumericalEquivalenceOrEnclosure
        );

        let same_fingerprint_drift = compare_f64_operator_batches(
            &parallel,
            &parallel_fingerprint,
            &numerically_equivalent,
            &parallel_fingerprint,
            1.0,
        )
        .unwrap();
        assert!(!same_fingerprint_drift.accepted);
        assert!(!same_fingerprint_drift.exact_payload_identity);
        assert_eq!(
            same_fingerprint_drift.comparison_plan.required_comparison,
            CrossFingerprintComparison::ByteIdentity
        );
    }

    struct SelectiveFailure;

    impl LinearOperator<f64> for SelectiveFailure {
        fn dimension(&self) -> usize {
            1
        }

        fn apply(&self, input: &[f64], _output: &mut [f64]) -> Result<(), OperatorError> {
            Err(OperatorError::ApplicationFailed(format!(
                "failure marker {}",
                input[0]
            )))
        }

        fn metadata(&self) -> OperatorMetadata {
            OperatorMetadata::new("selective failure", 1, MatrixStructure::MatrixFree, "f64")
        }
    }

    #[test]
    fn parallel_batch_reports_the_earliest_input_failure() {
        let tasks = vec![
            OperatorBatchTask {
                task_id: "first".to_owned(),
                input: vec![1.0],
            },
            OperatorBatchTask {
                task_id: "second".to_owned(),
                input: vec![2.0],
            },
        ];
        let error =
            apply_operator_batch_parallel(&SelectiveFailure, &tasks, &CancellationToken::new())
                .unwrap_err();
        assert!(error.to_string().contains("failure marker 1"));
    }
}
