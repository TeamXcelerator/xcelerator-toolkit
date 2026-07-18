// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::ConfigError;

/// Machine-readable declaration of a domain's independently reusable
/// artifacts and semantic invalidation boundaries.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactReusePlan {
    pub schema_version: u32,
    pub domain: String,
    pub semantics_version: String,
    pub artifacts: Vec<ArtifactReuseNode>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactReuseNode {
    pub kind: String,
    pub independently_cacheable: bool,
    pub dependencies: Vec<String>,
    pub invalidated_by: Vec<String>,
}

impl ArtifactReusePlan {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version == 0
            || self.domain.trim().is_empty()
            || self.semantics_version.trim().is_empty()
            || self.artifacts.is_empty()
        {
            return Err(ConfigError::new(
                "artifact reuse plan requires schema, domain, semantics, and artifacts",
            ));
        }
        let kinds = self
            .artifacts
            .iter()
            .map(|artifact| artifact.kind.as_str())
            .collect::<BTreeSet<_>>();
        if kinds.len() != self.artifacts.len() {
            return Err(ConfigError::new(
                "artifact reuse plan contains duplicate artifact kinds",
            ));
        }
        for artifact in &self.artifacts {
            if artifact.kind.trim().is_empty()
                || artifact.invalidated_by.is_empty()
                || artifact
                    .invalidated_by
                    .iter()
                    .any(|field| field.trim().is_empty())
                || artifact
                    .dependencies
                    .iter()
                    .any(|dependency| !kinds.contains(dependency.as_str()))
            {
                return Err(ConfigError::new(format!(
                    "artifact reuse node {:?} has invalid dependencies or invalidation fields",
                    artifact.kind
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_plan_is_serializable_and_rejects_unknown_dependencies() {
        let mut plan = ArtifactReusePlan {
            schema_version: 1,
            domain: "example".to_owned(),
            semantics_version: "example-v1".to_owned(),
            artifacts: vec![ArtifactReuseNode {
                kind: "result".to_owned(),
                independently_cacheable: true,
                dependencies: Vec::new(),
                invalidated_by: vec!["precision".to_owned()],
            }],
        };
        plan.validate().unwrap();
        assert!(serde_json::to_string(&plan).unwrap().contains("result"));
        plan.artifacts[0].dependencies.push("missing".to_owned());
        assert!(plan.validate().is_err());
    }
}
