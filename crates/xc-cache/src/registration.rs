//! Compare local registration metadata with the running toolkit's kind catalog.
//! This is a metadata preflight, not a payload, capacity, or permission audit.

use crate::{
    artifact_kind_is_private_only, artifact_kinds_for_family, CacheError, CacheVisibility,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactKindRegistrationAudit {
    pub family: String,
    pub visibility: CacheVisibility,
    pub toolkit_version: String,
    pub expected_kinds: Vec<String>,
    pub missing_from_registry: Vec<String>,
    pub missing_from_shard: Vec<String>,
    pub unexpected_in_registry: Vec<String>,
    pub unexpected_in_shard: Vec<String>,
}

impl ArtifactKindRegistrationAudit {
    pub fn is_ready(&self) -> bool {
        self.missing_from_registry.is_empty()
            && self.missing_from_shard.is_empty()
            && self.unexpected_in_registry.is_empty()
            && self.unexpected_in_shard.is_empty()
    }
}

/// Check both registration lists against existing production routing. Public
/// catalogs exclude private-only kinds. This never reads or publishes payloads.
/// A ready result establishes kind-list agreement only, not capture completion.
pub fn audit_artifact_kind_registration(
    family: &str,
    visibility: CacheVisibility,
    registry_kinds: &[String],
    shard_kinds: &[String],
) -> Result<ArtifactKindRegistrationAudit, CacheError> {
    if !matches!(
        visibility,
        CacheVisibility::Private | CacheVisibility::Public
    ) {
        return Err(CacheError::InvalidManifest(
            "registration requires a private or public lane".into(),
        ));
    }
    let kinds = artifact_kinds_for_family(family).ok_or_else(|| {
        CacheError::InvalidManifest(format!("unknown artifact family {family:?}"))
    })?;
    let expected = kinds
        .iter()
        .filter(|kind| {
            visibility == CacheVisibility::Private || !artifact_kind_is_private_only(kind)
        })
        .map(|kind| (*kind).to_owned())
        .collect::<BTreeSet<_>>();
    let checked = |kinds: &[String]| -> Result<BTreeSet<String>, CacheError> {
        let set = kinds.iter().cloned().collect::<BTreeSet<_>>();
        if set.len() != kinds.len() || set.iter().any(|kind| kind.trim().is_empty()) {
            return Err(CacheError::InvalidManifest(
                "registration contains duplicate or empty kinds".into(),
            ));
        }
        Ok(set)
    };
    let registry = checked(registry_kinds)?;
    let shard = checked(shard_kinds)?;
    Ok(ArtifactKindRegistrationAudit {
        family: family.into(),
        visibility,
        toolkit_version: env!("CARGO_PKG_VERSION").into(),
        expected_kinds: expected.iter().cloned().collect(),
        missing_from_registry: expected.difference(&registry).cloned().collect(),
        missing_from_shard: expected.difference(&shard).cloned().collect(),
        unexpected_in_registry: registry.difference(&expected).cloned().collect(),
        unexpected_in_shard: shard.difference(&expected).cloned().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evidence() -> Vec<String> {
        artifact_kinds_for_family("ccm-evidence")
            .unwrap()
            .iter()
            .map(|s| (*s).into())
            .collect()
    }
    #[test]
    fn detects_drift_even_when_both_remote_lists_agree() {
        let current = evidence();
        let old = current
            .iter()
            .filter(|s| *s != "ccm_prefix_analysis")
            .cloned()
            .collect::<Vec<_>>();
        let audit =
            audit_artifact_kind_registration("ccm-evidence", CacheVisibility::Private, &old, &old)
                .unwrap();
        assert!(!audit.is_ready());
        assert_eq!(audit.missing_from_registry, ["ccm_prefix_analysis"]);
        assert_eq!(audit.missing_from_shard, ["ccm_prefix_analysis"]);
        assert!(audit_artifact_kind_registration(
            "ccm-evidence",
            CacheVisibility::Private,
            &current,
            &current
        )
        .unwrap()
        .is_ready());
    }
    #[test]
    fn public_lane_cannot_advertise_private_only_kinds() {
        let all = artifact_kinds_for_family("ccm-distance")
            .unwrap()
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>();
        let public = all
            .iter()
            .filter(|s| !artifact_kind_is_private_only(s))
            .cloned()
            .collect::<Vec<_>>();
        assert!(audit_artifact_kind_registration(
            "ccm-distance",
            CacheVisibility::Public,
            &public,
            &public
        )
        .unwrap()
        .is_ready());
        let mixed = audit_artifact_kind_registration(
            "ccm-distance",
            CacheVisibility::Public,
            &public,
            &all,
        )
        .unwrap();
        assert_eq!(
            mixed.unexpected_in_shard,
            [
                "ccm_deviation_decomposition",
                "ccm_distance_resolution_evidence",
                "ccm_target_distance",
                "ccm_target_residual_analysis"
            ]
        );
        assert!(!mixed.is_ready());
    }
    #[test]
    fn unknown_families_kinds_and_duplicate_registrations_are_not_ready() {
        let all = evidence();
        assert!(audit_artifact_kind_registration(
            "future-family",
            CacheVisibility::Private,
            &all,
            &all
        )
        .is_err());
        let mut duplicate = all.clone();
        duplicate.push(all[0].clone());
        assert!(audit_artifact_kind_registration(
            "ccm-evidence",
            CacheVisibility::Private,
            &duplicate,
            &all
        )
        .is_err());
        let mut wrong = all.clone();
        wrong.push("ccm_tau_matrix".into());
        assert!(!audit_artifact_kind_registration(
            "ccm-evidence",
            CacheVisibility::Private,
            &wrong,
            &all
        )
        .unwrap()
        .is_ready());
    }
}
