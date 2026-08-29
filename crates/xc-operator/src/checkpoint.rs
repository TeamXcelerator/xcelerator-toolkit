//! Restartable, resource-bounded operator construction.

use crate::OperatorError;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use xc_core::{CancellationToken, ConfigDigest, ResourcePolicy};

pub const OPERATOR_CONSTRUCTION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstructionError {
    Invalid(String),
    IncompatibleCheckpoint(String),
    ResourceLimit(String),
    Cancelled(String),
    Operator(OperatorError),
}

impl Display for ConstructionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid operator construction: {message}"),
            Self::IncompatibleCheckpoint(message) => {
                write!(f, "incompatible operator checkpoint: {message}")
            }
            Self::ResourceLimit(message) => write!(f, "operator resource limit: {message}"),
            Self::Cancelled(message) => write!(f, "operator construction cancelled: {message}"),
            Self::Operator(error) => Display::fmt(error, f),
        }
    }
}

impl Error for ConstructionError {}

impl From<OperatorError> for ConstructionError {
    fn from(value: OperatorError) -> Self {
        Self::Operator(value)
    }
}

/// Stable identity and entry callback for one symmetric matrix construction.
/// Callers are responsible for making `entry` deterministic for this exact
/// descriptor; the checkpoint binds all semantic compatibility fields.
pub trait RestartableSymmetricAssemblerF64: Send + Sync {
    fn builder_id(&self) -> &str;
    fn builder_version(&self) -> u32;
    fn operator_identity(&self) -> &str;
    fn configuration_digest(&self) -> &ConfigDigest;
    fn dimension(&self) -> usize;
    fn entry(&self, row: usize, column: usize) -> Result<f64, OperatorError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorConstructionCheckpointF64 {
    pub schema_version: u32,
    pub builder_id: String,
    pub builder_version: u32,
    pub operator_identity: String,
    pub configuration_digest: ConfigDigest,
    pub dimension: usize,
    pub block_rows: usize,
    pub completed_rows: usize,
    /// Complete row-major rows `0..completed_rows`; no half-written row is
    /// checkpoint-visible.
    pub assembled_rows: Vec<f64>,
}

impl OperatorConstructionCheckpointF64 {
    pub fn retained_state_bytes(&self) -> u64 {
        (self.assembled_rows.len() as u64).saturating_mul(8)
    }

    pub fn validate(
        &self,
        assembler: &dyn RestartableSymmetricAssemblerF64,
        block_rows: usize,
    ) -> Result<(), ConstructionError> {
        if self.schema_version != OPERATOR_CONSTRUCTION_CHECKPOINT_SCHEMA_VERSION {
            return Err(ConstructionError::IncompatibleCheckpoint(
                "schema version differs".to_owned(),
            ));
        }
        if self.builder_id != assembler.builder_id()
            || self.builder_version != assembler.builder_version()
            || self.operator_identity != assembler.operator_identity()
            || &self.configuration_digest != assembler.configuration_digest()
            || self.dimension != assembler.dimension()
            || self.block_rows != block_rows
        {
            return Err(ConstructionError::IncompatibleCheckpoint(
                "builder, operator, configuration, dimension, or block policy differs".to_owned(),
            ));
        }
        if self.completed_rows == 0
            || self.completed_rows >= self.dimension
            || !self.completed_rows.is_multiple_of(self.block_rows)
            || self.assembled_rows.len() != self.completed_rows.saturating_mul(self.dimension)
            || self.assembled_rows.iter().any(|value| !value.is_finite())
        {
            return Err(ConstructionError::IncompatibleCheckpoint(
                "retained row state is malformed or not resumable".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum OperatorConstructionOutcomeF64 {
    Complete {
        operator_identity: String,
        dimension: usize,
        matrix_row_major: Vec<f64>,
    },
    Checkpointed {
        checkpoint: Box<OperatorConstructionCheckpointF64>,
    },
}

/// Assemble a deterministic number of complete row blocks.  `maximum_blocks`
/// bounds work in this invocation; reaching it returns a resumable checkpoint.
pub fn assemble_symmetric_operator_f64(
    assembler: &dyn RestartableSymmetricAssemblerF64,
    block_rows: usize,
    maximum_blocks: usize,
    resume: Option<&OperatorConstructionCheckpointF64>,
    resources: &ResourcePolicy,
    cancellation: &CancellationToken,
) -> Result<OperatorConstructionOutcomeF64, ConstructionError> {
    validate_descriptor(assembler, block_rows, maximum_blocks)?;
    let dimension = assembler.dimension();
    let matrix_bytes = (dimension as u64)
        .saturating_mul(dimension as u64)
        .saturating_mul(8);
    if resources
        .maximum_memory_bytes
        .is_some_and(|maximum| matrix_bytes > maximum)
    {
        return Err(ConstructionError::ResourceLimit(format!(
            "dense output requires {matrix_bytes} bytes"
        )));
    }

    let (mut completed_rows, mut assembled_rows) = if let Some(checkpoint) = resume {
        checkpoint.validate(assembler, block_rows)?;
        (checkpoint.completed_rows, checkpoint.assembled_rows.clone())
    } else {
        (0, Vec::with_capacity(dimension.saturating_mul(dimension)))
    };
    let mut completed_blocks = 0usize;
    while completed_rows < dimension && completed_blocks < maximum_blocks {
        if cancellation.is_cancelled() {
            return if completed_rows == 0 {
                Err(ConstructionError::Cancelled(
                    "cancelled before the first complete checkpoint block".to_owned(),
                ))
            } else {
                Ok(OperatorConstructionOutcomeF64::Checkpointed {
                    checkpoint: Box::new(make_checkpoint(
                        assembler,
                        block_rows,
                        completed_rows,
                        assembled_rows,
                    )),
                })
            };
        }
        let end_row = completed_rows.saturating_add(block_rows).min(dimension);
        for row in completed_rows..end_row {
            for column in 0..dimension {
                let value = assembler.entry(row, column)?;
                if !value.is_finite() {
                    return Err(ConstructionError::Invalid(format!(
                        "builder returned a non-finite entry at ({row}, {column})"
                    )));
                }
                assembled_rows.push(value);
            }
        }
        completed_rows = end_row;
        completed_blocks += 1;
    }

    if completed_rows < dimension {
        return Ok(OperatorConstructionOutcomeF64::Checkpointed {
            checkpoint: Box::new(make_checkpoint(
                assembler,
                block_rows,
                completed_rows,
                assembled_rows,
            )),
        });
    }
    verify_symmetric(&assembled_rows, dimension)?;
    Ok(OperatorConstructionOutcomeF64::Complete {
        operator_identity: assembler.operator_identity().to_owned(),
        dimension,
        matrix_row_major: assembled_rows,
    })
}

fn validate_descriptor(
    assembler: &dyn RestartableSymmetricAssemblerF64,
    block_rows: usize,
    maximum_blocks: usize,
) -> Result<(), ConstructionError> {
    if assembler.dimension() == 0
        || block_rows == 0
        || maximum_blocks == 0
        || assembler.builder_id().trim().is_empty()
        || assembler.builder_version() == 0
        || assembler.operator_identity().trim().is_empty()
        || !assembler.configuration_digest().is_sha256()
    {
        return Err(ConstructionError::Invalid(
            "descriptor identities, versions, dimensions, block sizes, and digest must be valid"
                .to_owned(),
        ));
    }
    Ok(())
}

fn make_checkpoint(
    assembler: &dyn RestartableSymmetricAssemblerF64,
    block_rows: usize,
    completed_rows: usize,
    assembled_rows: Vec<f64>,
) -> OperatorConstructionCheckpointF64 {
    OperatorConstructionCheckpointF64 {
        schema_version: OPERATOR_CONSTRUCTION_CHECKPOINT_SCHEMA_VERSION,
        builder_id: assembler.builder_id().to_owned(),
        builder_version: assembler.builder_version(),
        operator_identity: assembler.operator_identity().to_owned(),
        configuration_digest: assembler.configuration_digest().clone(),
        dimension: assembler.dimension(),
        block_rows,
        completed_rows,
        assembled_rows,
    }
}

fn verify_symmetric(matrix: &[f64], dimension: usize) -> Result<(), ConstructionError> {
    for row in 0..dimension {
        for column in 0..row {
            if matrix[row * dimension + column].to_bits()
                != matrix[column * dimension + row].to_bits()
            {
                return Err(ConstructionError::Invalid(format!(
                    "assembled operator is not bitwise symmetric at ({row}, {column})"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixtureAssembler {
        identity: String,
        digest: ConfigDigest,
        dimension: usize,
    }

    impl RestartableSymmetricAssemblerF64 for FixtureAssembler {
        fn builder_id(&self) -> &str {
            "symmetric_fixture_builder"
        }

        fn builder_version(&self) -> u32 {
            1
        }

        fn operator_identity(&self) -> &str {
            &self.identity
        }

        fn configuration_digest(&self) -> &ConfigDigest {
            &self.digest
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn entry(&self, row: usize, column: usize) -> Result<f64, OperatorError> {
            Ok((row.min(column) * 10 + row.max(column)) as f64)
        }
    }

    fn assembler(identity: &str) -> FixtureAssembler {
        FixtureAssembler {
            identity: identity.to_owned(),
            digest: ConfigDigest("22".repeat(32)),
            dimension: 7,
        }
    }

    #[test]
    fn resumed_construction_is_identical_and_bounded() {
        let resources = ResourcePolicy::default();
        let token = CancellationToken::new();
        let fixture = assembler("sha256:operator-v1");
        let cold =
            assemble_symmetric_operator_f64(&fixture, 2, 10, None, &resources, &token).unwrap();
        let first =
            assemble_symmetric_operator_f64(&fixture, 2, 1, None, &resources, &token).unwrap();
        let OperatorConstructionOutcomeF64::Checkpointed { checkpoint } = first else {
            panic!("one block should produce a checkpoint");
        };
        assert_eq!(checkpoint.completed_rows, 2);
        assert_eq!(checkpoint.retained_state_bytes(), 2 * 7 * 8);
        let resumed =
            assemble_symmetric_operator_f64(&fixture, 2, 10, Some(&checkpoint), &resources, &token)
                .unwrap();
        assert_eq!(resumed, cold);
    }

    #[test]
    fn resume_rejects_identity_drift_and_resource_overflow() {
        let resources = ResourcePolicy::default();
        let token = CancellationToken::new();
        let original = assembler("sha256:original");
        let first =
            assemble_symmetric_operator_f64(&original, 2, 1, None, &resources, &token).unwrap();
        let OperatorConstructionOutcomeF64::Checkpointed { checkpoint } = first else {
            panic!("expected checkpoint");
        };
        let changed = assembler("sha256:changed");
        let error =
            assemble_symmetric_operator_f64(&changed, 2, 1, Some(&checkpoint), &resources, &token)
                .unwrap_err();
        assert!(matches!(
            error,
            ConstructionError::IncompatibleCheckpoint(_)
        ));

        let mut tiny = resources;
        tiny.maximum_memory_bytes = Some(8);
        let error =
            assemble_symmetric_operator_f64(&original, 2, 1, None, &tiny, &token).unwrap_err();
        assert!(matches!(error, ConstructionError::ResourceLimit(_)));
    }
}
