use crate::ConfigError;
use serde::{Deserialize, Serialize};

/// Invariant or user-declared subspace in which a problem is solved.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "subspace")]
pub enum Subspace {
    #[default]
    Full,
    EvenReflection,
    OddReflection,
    ExplicitBasis {
        ambient_dimension: usize,
        reduced_dimension: usize,
        basis_id: String,
    },
    Projector {
        ambient_dimension: usize,
        reduced_dimension: usize,
        projector_id: String,
    },
}

impl Subspace {
    pub fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Full | Self::EvenReflection | Self::OddReflection => Ok(()),
            Self::ExplicitBasis {
                ambient_dimension,
                reduced_dimension,
                basis_id,
            } => validate_reduction(*ambient_dimension, *reduced_dimension, basis_id, "basis_id"),
            Self::Projector {
                ambient_dimension,
                reduced_dimension,
                projector_id,
            } => validate_reduction(
                *ambient_dimension,
                *reduced_dimension,
                projector_id,
                "projector_id",
            ),
        }
    }
}

fn validate_reduction(
    ambient_dimension: usize,
    reduced_dimension: usize,
    identifier: &str,
    identifier_name: &str,
) -> Result<(), ConfigError> {
    if ambient_dimension == 0 || reduced_dimension == 0 {
        return Err(ConfigError::new("subspace dimensions must be positive"));
    }
    if reduced_dimension > ambient_dimension {
        return Err(ConfigError::new(
            "reduced_dimension may not exceed ambient_dimension",
        ));
    }
    if identifier.trim().is_empty() {
        return Err(ConfigError::new(format!(
            "{identifier_name} must be nonempty"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_reduction_must_be_well_formed() {
        let subspace = Subspace::ExplicitBasis {
            ambient_dimension: 10,
            reduced_dimension: 11,
            basis_id: "bad".to_owned(),
        };
        assert!(subspace.validate().is_err());
    }
}
