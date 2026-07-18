use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

/// Uniform numerical evidence attached to every computed eigenpair.
///
/// `T` is the active report scalar. High-precision in-memory reports use their
/// backend-native scalar, while portable reports may use lossless decimal
/// strings without changing the diagnostic schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EigenpairDiagnostics<T> {
    pub absolute_residual: T,
    pub relative_residual: T,
    pub scaled_backward_error: T,
    pub orthogonality_error: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryClassification {
    Never,
    RetryUnchanged,
    ResumeFromCheckpoint,
    AfterInputCorrection,
    AtHigherPrecision,
    AfterResourceIncrease,
    AfterAuthorityChange,
}

/// Portable failure context shared by numerical, cache, certificate, and CLI routes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureDiagnostic {
    pub schema_version: u32,
    pub stage: String,
    pub operation: String,
    pub problem_identity: Option<String>,
    pub artifact_identity: Option<String>,
    pub source_cause: String,
    pub saved_diagnostic_path: Option<String>,
    pub resource_context: BTreeMap<String, String>,
    pub authority_context: BTreeMap<String, String>,
    pub retry: RetryClassification,
    pub higher_precision_reasonable: bool,
}

impl FailureDiagnostic {
    pub fn new(
        stage: impl Into<String>,
        operation: impl Into<String>,
        source_cause: impl Into<String>,
        retry: RetryClassification,
    ) -> Self {
        Self {
            schema_version: 1,
            stage: stage.into(),
            operation: operation.into(),
            problem_identity: None,
            artifact_identity: None,
            source_cause: source_cause.into(),
            saved_diagnostic_path: None,
            resource_context: BTreeMap::new(),
            authority_context: BTreeMap::new(),
            retry,
            higher_precision_reasonable: retry == RetryClassification::AtHigherPrecision,
        }
    }

    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.schema_version != 1
            || self.stage.trim().is_empty()
            || self.operation.trim().is_empty()
            || self.source_cause.trim().is_empty()
            || self
                .problem_identity
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            || self
                .artifact_identity
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            || self
                .saved_diagnostic_path
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
        {
            return Err(crate::ConfigError::new(
                "failure diagnostic requires schema 1 and nonempty contextual fields",
            ));
        }
        Ok(())
    }

    /// Persist a diagnostic without overwriting prior evidence.
    pub fn write_new(mut self, path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "diagnostic path must be nonempty",
            ));
        }
        self.saved_diagnostic_path = Some(path.to_string_lossy().into_owned());
        self.validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let bytes = serde_json::to_vec_pretty(&self).map_err(io::Error::other)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn failure_diagnostic_persists_context_and_refuses_overwrite() {
        let path = std::env::temp_dir().join(format!(
            "xc-failure-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut diagnostic = FailureDiagnostic::new(
            "certificate_verification",
            "verify finite CCM inertia",
            "pivot enclosure does not exclude zero",
            RetryClassification::AtHigherPrecision,
        );
        diagnostic.problem_identity = Some("ccm:c=13:n=4".to_owned());
        diagnostic.artifact_identity = Some("sha256:certificate".to_owned());
        diagnostic
            .resource_context
            .insert("precision_bits".to_owned(), "512".to_owned());

        let saved = diagnostic.clone().write_new(&path).unwrap();
        assert_eq!(
            saved.saved_diagnostic_path.as_deref(),
            Some(path.to_string_lossy().as_ref())
        );
        assert!(saved.higher_precision_reasonable);
        let decoded: FailureDiagnostic =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(decoded, saved);
        assert_eq!(
            diagnostic.write_new(&path).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        std::fs::remove_file(path).unwrap();
    }
}
