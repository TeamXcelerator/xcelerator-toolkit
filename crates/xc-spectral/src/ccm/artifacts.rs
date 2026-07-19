// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Granular CCM artifact identity and cache-key construction.
//!
//! Expensive source construction is separated from root-window requests. A
//! request for additional roots may therefore reuse the same matrix
//! components, assembled form, eigenpair, and secular source.

use serde::{Deserialize, Serialize};
use xc_cache::{ArtifactKey, CacheError, ContentDigest, DependencyRef};

pub const CCM_MATHEMATICS_SEMANTICS: &str = "ccm-v0.13.0-v1";

pub fn ccm_artifact_reuse_plan() -> xc_core::ArtifactReusePlan {
    use xc_core::{ArtifactReuseNode, ArtifactReusePlan};
    let node = |kind: &str,
                independently_cacheable: bool,
                dependencies: &[&str],
                invalidated_by: &[&str]|
     -> ArtifactReuseNode {
        ArtifactReuseNode {
            kind: kind.to_owned(),
            independently_cacheable,
            dependencies: dependencies
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            invalidated_by: invalidated_by
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        }
    };
    ArtifactReusePlan {
        schema_version: 1,
        domain: "ccm".to_owned(),
        semantics_version: CCM_MATHEMATICS_SEMANTICS.to_owned(),
        artifacts: vec![
            node(
                "prime_content_metadata",
                false,
                &[],
                &["prime_cutoff", "cutoff_mode"],
            ),
            node(
                "quadrature_rule",
                true,
                &[],
                &["order", "precision_bits", "rule_semantics"],
            ),
            node(
                "archimedean_integrals",
                true,
                &["quadrature_rule"],
                &["lambda_squared", "n_modes", "precision_bits"],
            ),
            node(
                "prime_component",
                true,
                &["prime_content_metadata"],
                &["lambda_squared", "n_modes", "precision_bits"],
            ),
            node(
                "pole_descriptor",
                false,
                &[],
                &["lambda_squared", "basis", "pole_semantics"],
            ),
            node(
                "tau_matrix",
                true,
                &[
                    "archimedean_integrals",
                    "prime_component",
                    "pole_descriptor",
                ],
                &["assembly_semantics", "normalization"],
            ),
            node(
                "even_sector_matrix",
                true,
                &["tau_matrix"],
                &["parity_basis_semantics"],
            ),
            node(
                "odd_sector_matrix",
                true,
                &["tau_matrix"],
                &["parity_basis_semantics"],
            ),
            node(
                "factorization",
                true,
                &["tau_matrix", "even_sector_matrix", "odd_sector_matrix"],
                &["subspace", "factorization_semantics", "precision_bits"],
            ),
            node(
                "sector_spectrum",
                true,
                &["even_sector_matrix", "odd_sector_matrix"],
                &["subspace", "eigenpair_count", "solver_semantics"],
            ),
            node(
                "sector_gap",
                true,
                &["sector_spectrum"],
                &["gap_semantics", "precision_bits"],
            ),
            node(
                "weil_eigenpair",
                true,
                &["factorization"],
                &["state_target", "solver_semantics", "normalization"],
            ),
            node(
                "secular_source",
                true,
                &["weil_eigenpair"],
                &["source_semantics"],
            ),
            node(
                "root_count_window",
                true,
                &["secular_source"],
                &["window_boundaries", "count_semantics"],
            ),
            node(
                "root_discovery_window",
                true,
                &["secular_source", "root_count_window"],
                &["root_range", "isolation_policy", "target_digits"],
            ),
            node(
                "root_refinement",
                true,
                &["secular_source"],
                &["reference_seeds", "refinement_policy", "target_digits"],
            ),
            node(
                "post_discovery_comparison",
                true,
                &["root_discovery_window"],
                &["reference_dataset", "comparison_semantics"],
            ),
            node(
                "configuration_evidence",
                true,
                &["weil_eigenpair", "root_discovery_window"],
                &["evidence_semantics"],
            ),
            node(
                "certificate",
                true,
                &["root_discovery_window", "configuration_evidence"],
                &["certificate_policy", "assurance"],
            ),
        ],
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CcmArtifactKind {
    ArchimedeanIntegrals,
    ArchimedeanComponent,
    PrimeComponent,
    PoleComponent,
    TauMatrix,
    EvenSectorMatrix,
    OddSectorMatrix,
    Factorization,
    SectorSpectrum,
    SectorGap,
    WeilEigenpair,
    ProlateCandidate,
    SecularSource,
    RootCountWindow,
    RootDiscoveryWindow,
    RootRefinement,
    SpectralWindow,
    PostDiscoveryComparison,
    ConvergenceDiagnostics,
    CertificateBundle,
}

impl CcmArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ArchimedeanIntegrals => "ccm_archimedean_integrals",
            Self::ArchimedeanComponent => "ccm_archimedean_component",
            Self::PrimeComponent => "ccm_prime_component",
            Self::PoleComponent => "ccm_pole_component",
            Self::TauMatrix => "ccm_tau_matrix",
            Self::EvenSectorMatrix => "ccm_even_sector_matrix",
            Self::OddSectorMatrix => "ccm_odd_sector_matrix",
            Self::Factorization => "ccm_factorization",
            Self::SectorSpectrum => "ccm_sector_spectrum",
            Self::SectorGap => "ccm_sector_gap",
            Self::WeilEigenpair => "ccm_weil_eigenpair",
            Self::ProlateCandidate => "ccm_prolate_candidate",
            Self::SecularSource => "ccm_secular_source",
            Self::RootCountWindow => "ccm_root_count_window",
            Self::RootDiscoveryWindow => "ccm_root_discovery_window",
            Self::RootRefinement => "ccm_root_refinement",
            Self::SpectralWindow => "ccm_spectral_window",
            Self::PostDiscoveryComparison => "ccm_post_discovery_comparison",
            Self::ConvergenceDiagnostics => "ccm_convergence_diagnostics",
            Self::CertificateBundle => "ccm_certificate_bundle",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmSourceParameters {
    pub mathematics_semantics: String,
    pub lambda_sq: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub subspace: String,
    pub prime_cutoff: u64,
    pub cutoff_mode: String,
    pub assembly_variant: String,
}

impl CcmSourceParameters {
    pub fn validate(&self) -> Result<(), CacheError> {
        if self.mathematics_semantics.trim().is_empty()
            || self.lambda_sq.trim().is_empty()
            || self.subspace.trim().is_empty()
            || self.cutoff_mode.trim().is_empty()
            || self.assembly_variant.trim().is_empty()
            || self.n_modes == 0
            || self.precision_bits < 32
        {
            return Err(CacheError::InvalidManifest(
                "invalid CCM source parameters".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmComponentParameters {
    pub source: CcmSourceParameters,
    pub component_version: String,
    pub quadrature_or_formula: String,
    pub finite_cutoff: Option<String>,
    pub cutoff_free: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmEigenpairParameters {
    pub source: CcmSourceParameters,
    pub matrix_digest: ContentDigest,
    pub target: String,
    pub solver_plan_digest: ContentDigest,
    pub normalization: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmWindowParameters {
    pub secular_source_digest: ContentDigest,
    pub target: String,
    pub lower_height: String,
    pub upper_height: String,
    pub target_digits: u32,
    pub assurance: String,
    pub discovery_method: String,
    pub count_method: String,
    pub ordinal_assignment: String,
    pub reference_seeded: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CcmRootRefinementParameters {
    pub secular_source_digest: ContentDigest,
    pub discovery_digest: ContentDigest,
    pub root_index: Option<usize>,
    pub bracket_lower: String,
    pub bracket_upper: String,
    pub target_digits: u32,
    pub precision_bits: u32,
    pub refinement_method: String,
}

#[derive(Clone, Debug)]
pub struct CcmCacheKeyBuilder {
    pub namespace: String,
}

impl Default for CcmCacheKeyBuilder {
    fn default() -> Self {
        Self {
            namespace: "ccm".to_owned(),
        }
    }
}

impl CcmCacheKeyBuilder {
    pub fn new(namespace: impl Into<String>) -> Result<Self, CacheError> {
        let namespace = namespace.into();
        if namespace.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "CCM cache namespace must be nonempty".to_owned(),
            ));
        }
        Ok(Self { namespace })
    }

    pub fn source_logical_key(&self, source: &CcmSourceParameters) -> String {
        format!(
            "{}/lambda_sq={}/n={}/prec={}/subspace={}/cutoff={}/{}",
            self.namespace,
            source.lambda_sq,
            source.n_modes,
            source.precision_bits,
            source.subspace,
            source.prime_cutoff,
            source.assembly_variant
        )
    }

    fn key<T: Serialize>(
        &self,
        kind: CcmArtifactKind,
        logical_key: String,
        parameters: &T,
    ) -> Result<ArtifactKey, CacheError> {
        let bytes = serde_json::to_vec(parameters)?;
        ArtifactKey::new(kind.as_str(), logical_key, &bytes)
    }

    pub fn component_key(
        &self,
        kind: CcmArtifactKind,
        parameters: &CcmComponentParameters,
    ) -> Result<ArtifactKey, CacheError> {
        parameters.source.validate()?;
        if !matches!(
            kind,
            CcmArtifactKind::ArchimedeanComponent
                | CcmArtifactKind::PrimeComponent
                | CcmArtifactKind::PoleComponent
        ) {
            return Err(CacheError::InvalidManifest(
                "component_key requires an individual CCM form component kind".to_owned(),
            ));
        }
        self.key(
            kind,
            format!(
                "{}/component={}",
                self.source_logical_key(&parameters.source),
                kind.as_str()
            ),
            parameters,
        )
    }

    pub fn matrix_key(
        &self,
        source: &CcmSourceParameters,
        component_dependencies: &[DependencyRef],
        even_sector: bool,
    ) -> Result<ArtifactKey, CacheError> {
        source.validate()?;
        let parameters = (
            source,
            component_dependencies,
            if even_sector { "even" } else { "full" },
        );
        self.key(
            if even_sector {
                CcmArtifactKind::EvenSectorMatrix
            } else {
                CcmArtifactKind::TauMatrix
            },
            format!(
                "{}/matrix={}",
                self.source_logical_key(source),
                if even_sector { "even" } else { "full" }
            ),
            &parameters,
        )
    }

    pub fn parity_sector_matrix_key(
        &self,
        source: &CcmSourceParameters,
        tau_dependency: &DependencyRef,
        parity: &str,
    ) -> Result<ArtifactKey, CacheError> {
        source.validate()?;
        let kind = match parity {
            "even" => CcmArtifactKind::EvenSectorMatrix,
            "odd" => CcmArtifactKind::OddSectorMatrix,
            _ => {
                return Err(CacheError::InvalidManifest(
                    "CCM parity sector must be even or odd".to_owned(),
                ))
            }
        };
        self.key(
            kind,
            format!("{}/matrix={parity}", self.source_logical_key(source)),
            &(source, tau_dependency, parity),
        )
    }

    pub fn eigenpair_key(
        &self,
        parameters: &CcmEigenpairParameters,
    ) -> Result<ArtifactKey, CacheError> {
        parameters.source.validate()?;
        self.key(
            CcmArtifactKind::WeilEigenpair,
            format!(
                "{}/eigenpair={}",
                self.source_logical_key(&parameters.source),
                parameters.target
            ),
            parameters,
        )
    }

    pub fn secular_source_key(
        &self,
        eigenpair_digest: &ContentDigest,
        normalization: &str,
    ) -> Result<ArtifactKey, CacheError> {
        if !eigenpair_digest.validate() || normalization.trim().is_empty() {
            return Err(CacheError::InvalidManifest(
                "invalid CCM secular source parameters".to_owned(),
            ));
        }
        self.key(
            CcmArtifactKind::SecularSource,
            format!("{}/secular_source={}", self.namespace, eigenpair_digest),
            &(eigenpair_digest, normalization),
        )
    }

    pub fn discovery_window_key(
        &self,
        parameters: &CcmWindowParameters,
    ) -> Result<ArtifactKey, CacheError> {
        if !parameters.secular_source_digest.validate()
            || parameters.discovery_method.trim().is_empty()
            || parameters.count_method.trim().is_empty()
            || parameters.ordinal_assignment != "exact_cumulative_finite_source_root_count"
            || parameters.reference_seeded
        {
            return Err(CacheError::InvalidManifest(
                "CCM discovery windows must be reference-free and exact-count indexed".to_owned(),
            ));
        }
        self.key(
            CcmArtifactKind::RootDiscoveryWindow,
            format!(
                "{}/source={}/window={}-{}",
                self.namespace,
                parameters.secular_source_digest,
                parameters.lower_height,
                parameters.upper_height
            ),
            parameters,
        )
    }

    pub fn root_refinement_key(
        &self,
        parameters: &CcmRootRefinementParameters,
    ) -> Result<ArtifactKey, CacheError> {
        if !parameters.secular_source_digest.validate()
            || !parameters.discovery_digest.validate()
            || parameters.precision_bits < 32
        {
            return Err(CacheError::InvalidManifest(
                "invalid CCM root refinement parameters".to_owned(),
            ));
        }
        self.key(
            CcmArtifactKind::RootRefinement,
            format!(
                "{}/source={}/root={}",
                self.namespace,
                parameters.secular_source_digest,
                parameters
                    .root_index
                    .map_or_else(|| "unindexed".to_owned(), |index| index.to_string())
            ),
            parameters,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xc_cache::CacheQuality;

    #[test]
    fn ccm_reuse_plan_keeps_root_refinement_off_matrix_dependencies() {
        let plan = ccm_artifact_reuse_plan();
        plan.validate().unwrap();
        let root = plan
            .artifacts
            .iter()
            .find(|node| node.kind == "root_refinement")
            .unwrap();
        assert_eq!(root.dependencies, vec!["secular_source"]);
        assert!(!root.invalidated_by.iter().any(|field| field == "n_modes"));
    }

    fn source() -> CcmSourceParameters {
        CcmSourceParameters {
            mathematics_semantics: CCM_MATHEMATICS_SEMANTICS.to_owned(),
            lambda_sq: "100".to_owned(),
            n_modes: 500,
            precision_bits: 3338,
            subspace: "even".to_owned(),
            prime_cutoff: 100,
            cutoff_mode: "integer".to_owned(),
            assembly_variant: "cutoff_free".to_owned(),
        }
    }

    #[test]
    fn source_artifacts_are_independent_of_requested_root_count() {
        let builder = CcmCacheKeyBuilder::default();
        let component = CcmComponentParameters {
            source: source(),
            component_version: "arch-v2".to_owned(),
            quadrature_or_formula: "closed_form_cutoff_free".to_owned(),
            finite_cutoff: None,
            cutoff_free: true,
        };
        let first = builder
            .component_key(CcmArtifactKind::ArchimedeanComponent, &component)
            .unwrap();
        let second = builder
            .component_key(CcmArtifactKind::ArchimedeanComponent, &component)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn matrix_key_changes_when_component_digest_changes() {
        let builder = CcmCacheKeyBuilder::default();
        let dependency = |payload: &[u8]| DependencyRef {
            key: ArtifactKey::new("component", "arch", b"{}").unwrap(),
            content_digest: ContentDigest::sha256(payload),
            required_quality: CacheQuality::Validated,
        };
        let first = builder
            .matrix_key(&source(), &[dependency(b"a")], true)
            .unwrap();
        let second = builder
            .matrix_key(&source(), &[dependency(b"b")], true)
            .unwrap();
        assert_ne!(first.parameters_digest, second.parameters_digest);
    }

    #[test]
    fn parity_sector_keys_are_distinct_and_depend_on_the_exact_tau() {
        let builder = CcmCacheKeyBuilder::default();
        let dependency = DependencyRef {
            key: ArtifactKey::new("ccm_tau_matrix", "tau", b"{}").unwrap(),
            content_digest: ContentDigest::sha256(b"tau-payload"),
            required_quality: CacheQuality::Validated,
        };
        let even = builder
            .parity_sector_matrix_key(&source(), &dependency, "even")
            .unwrap();
        let odd = builder
            .parity_sector_matrix_key(&source(), &dependency, "odd")
            .unwrap();
        assert_eq!(even.kind, CcmArtifactKind::EvenSectorMatrix.as_str());
        assert_eq!(odd.kind, CcmArtifactKind::OddSectorMatrix.as_str());
        assert_ne!(even, odd);
        assert!(builder
            .parity_sector_matrix_key(&source(), &dependency, "natural")
            .is_err());
    }

    #[test]
    fn independent_discovery_key_rejects_reference_seed_provenance() {
        let builder = CcmCacheKeyBuilder::default();
        let mut parameters = CcmWindowParameters {
            secular_source_digest: ContentDigest::sha256(b"finite-source"),
            target: "index_range=2-4".to_owned(),
            lower_height: "10".to_owned(),
            upper_height: "40".to_owned(),
            target_digits: 30,
            assurance: "computed".to_owned(),
            discovery_method: "pole_aware_finite_source".to_owned(),
            count_method: "exact_finite_source_count".to_owned(),
            ordinal_assignment: "exact_cumulative_finite_source_root_count".to_owned(),
            reference_seeded: false,
        };
        builder.discovery_window_key(&parameters).unwrap();
        parameters.reference_seeded = true;
        assert!(builder.discovery_window_key(&parameters).is_err());
    }
}
