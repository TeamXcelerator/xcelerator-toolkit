//! Fail-closed capability preflight.
//!
//! A preflight request names exact capabilities. The catalog never substitutes
//! a weaker scalar backend, solver route, assurance level, cache mode, or
//! publication target. Accepted reports are safe to persist before expensive
//! computation or remote mutation begins.

use crate::provenance::CacheValidationMode;
use crate::{AssuranceLevel, ConfigDigest, ResourceEstimate, ResourcePolicy};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheAccessMode {
    Disabled,
    ReadOnly,
    LocalReadWrite,
    RemoteRead,
    RemotePublish,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PublicationTarget {
    #[default]
    None,
    Private,
    Public,
    Both,
}

impl PublicationTarget {
    pub fn includes_private(self) -> bool {
        matches!(self, Self::Private | Self::Both)
    }

    pub fn includes_public(self) -> bool {
        matches!(self, Self::Public | Self::Both)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationAuthorityMode {
    OwnerDirect,
    ContributorReviewed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PublicationAuthority {
    /// Stable, non-secret principal identifier.
    pub principal: String,
    pub mode: PublicationAuthorityMode,
    pub allowed_targets: BTreeSet<PublicationTarget>,
    pub allowed_repositories: BTreeSet<String>,
    pub policy_digest: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PublicationPreflightRequest {
    pub target: PublicationTarget,
    pub private_repository: Option<String>,
    pub public_repository: Option<String>,
    pub authority: Option<PublicationAuthority>,
}

impl Default for PublicationPreflightRequest {
    fn default() -> Self {
        Self {
            target: PublicationTarget::None,
            private_repository: None,
            public_repository: None,
            authority: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScalarCapability {
    pub id: String,
    pub supported_platforms: BTreeSet<String>,
    pub maximum_precision_bits: Option<u32>,
    pub arbitrary_precision: bool,
    pub rigorous_real_enclosures: bool,
    pub rigorous_complex_enclosures: bool,
    pub exact: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SolverCapability {
    pub id: String,
    pub algorithm_family: String,
    pub scalar_backends: BTreeSet<String>,
    pub operator_representations: BTreeSet<String>,
    pub target_kinds: BTreeSet<String>,
    pub generalized: bool,
    pub maximum_assurance: AssuranceLevel,
    pub checkpoint_supported: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CertificationCapability {
    pub id: String,
    pub scalar_backends: BTreeSet<String>,
    pub claim_kinds: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreflightRequest {
    pub effective_config_digest: ConfigDigest,
    pub platform: String,
    pub scalar_backend: String,
    pub precision_bits: u32,
    pub operator_representation: String,
    pub target_kind: String,
    pub generalized: bool,
    pub primary_solver: String,
    pub independent_solver: Option<String>,
    pub requested_assurance: AssuranceLevel,
    pub certification_route: Option<String>,
    pub certification_claim: Option<String>,
    pub cache_mode: CacheAccessMode,
    pub cache_policy_digest: Option<ConfigDigest>,
    pub cache_validation_mode: Option<CacheValidationMode>,
    /// Principal proved by the authentication adapter; never a credential.
    pub authenticated_principal: Option<String>,
    pub publication: PublicationPreflightRequest,
    pub resources: ResourcePolicy,
    pub estimate: ResourceEstimate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreflightFailureCode {
    InvalidRequest,
    UnsupportedPlatform,
    UnsupportedScalarBackend,
    UnsupportedPrecision,
    UnsupportedSolver,
    UnsupportedOperatorRepresentation,
    UnsupportedTarget,
    UnsupportedGeneralizedProblem,
    InsufficientAssuranceCapability,
    MissingIndependentRoute,
    NonIndependentRoute,
    MissingCertificationRoute,
    UnsupportedCertificationRoute,
    NonRigorousCertificationBackend,
    InfeasibleResources,
    MissingCachePolicy,
    InsufficientCacheValidation,
    PublicationNotExplicit,
    MissingAuthenticatedPrincipal,
    AuthenticatedPrincipalMismatch,
    MissingPublicationAuthority,
    PublicationAuthorityMismatch,
    PublicationRepositoryNotAuthorized,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreflightFailure {
    pub code: PreflightFailureCode,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreflightReport {
    pub accepted: bool,
    pub effective_config_digest: ConfigDigest,
    pub selected_capabilities: Vec<String>,
    pub evidence_plan: Vec<String>,
    pub failures: Vec<PreflightFailure>,
    pub warnings: Vec<String>,
}

impl PreflightReport {
    pub fn require_accepted(&self) -> Result<(), PreflightError> {
        if self.accepted {
            Ok(())
        } else {
            Err(PreflightError {
                failures: self.failures.clone(),
            })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightError {
    pub failures: Vec<PreflightFailure>,
}

impl Display for PreflightError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "capability preflight failed with {} error(s)",
            self.failures.len()
        )
    }
}

impl Error for PreflightError {}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCatalog {
    pub scalar_backends: Vec<ScalarCapability>,
    pub solvers: Vec<SolverCapability>,
    pub certification_routes: Vec<CertificationCapability>,
}

impl CapabilityCatalog {
    pub fn preflight(&self, request: &PreflightRequest) -> PreflightReport {
        let mut failures = Vec::new();
        let mut warnings = Vec::new();
        let mut selected = Vec::new();
        let mut evidence_plan = Vec::new();

        if request.platform.trim().is_empty()
            || request.scalar_backend.trim().is_empty()
            || request.operator_representation.trim().is_empty()
            || request.target_kind.trim().is_empty()
            || request.primary_solver.trim().is_empty()
        {
            fail(
                &mut failures,
                PreflightFailureCode::InvalidRequest,
                "platform, backend, operator representation, target, and primary solver must be explicit",
            );
        }
        if !request.effective_config_digest.is_sha256() {
            fail(
                &mut failures,
                PreflightFailureCode::InvalidRequest,
                "effective configuration digest must be 64 lowercase hexadecimal SHA-256 characters",
            );
        }
        if request.precision_bits == 0 {
            fail(
                &mut failures,
                PreflightFailureCode::InvalidRequest,
                "requested precision must be positive",
            );
        }

        let scalar = self
            .scalar_backends
            .iter()
            .find(|candidate| candidate.id == request.scalar_backend);
        match scalar {
            None => fail(
                &mut failures,
                PreflightFailureCode::UnsupportedScalarBackend,
                format!(
                    "scalar backend {:?} is not registered",
                    request.scalar_backend
                ),
            ),
            Some(scalar) => {
                selected.push(format!("scalar:{}", scalar.id));
                if !scalar.supported_platforms.contains(&request.platform) {
                    fail(
                        &mut failures,
                        PreflightFailureCode::UnsupportedPlatform,
                        format!(
                            "scalar backend {:?} does not support platform {:?}",
                            scalar.id, request.platform
                        ),
                    );
                }
                if scalar
                    .maximum_precision_bits
                    .is_some_and(|maximum| request.precision_bits > maximum)
                {
                    fail(
                        &mut failures,
                        PreflightFailureCode::UnsupportedPrecision,
                        format!(
                            "requested precision {} exceeds backend {:?} maximum {:?}",
                            request.precision_bits, scalar.id, scalar.maximum_precision_bits
                        ),
                    );
                }
                if request.precision_bits > 64 && !scalar.arbitrary_precision && !scalar.exact {
                    fail(
                        &mut failures,
                        PreflightFailureCode::UnsupportedPrecision,
                        format!(
                            "backend {:?} cannot satisfy {}-bit production precision",
                            scalar.id, request.precision_bits
                        ),
                    );
                }
            }
        }

        let primary = self
            .solvers
            .iter()
            .find(|solver| solver.id == request.primary_solver);
        validate_solver(primary, "primary", request, &mut failures, &mut selected);

        if request.requested_assurance == AssuranceLevel::CrossChecked
            || request.independent_solver.is_some()
        {
            match request
                .independent_solver
                .as_ref()
                .and_then(|id| self.solvers.iter().find(|solver| &solver.id == id))
            {
                None => fail(
                    &mut failures,
                    PreflightFailureCode::MissingIndependentRoute,
                    "a requested independent comparison requires an explicit second solver route",
                ),
                Some(independent) => {
                    validate_solver(
                        Some(independent),
                        "independent",
                        request,
                        &mut failures,
                        &mut selected,
                    );
                    if primary.is_some_and(|primary| {
                        primary.id == independent.id
                            || primary.algorithm_family == independent.algorithm_family
                    }) {
                        fail(
                            &mut failures,
                            PreflightFailureCode::NonIndependentRoute,
                            "the requested route pair uses the same solver or algorithm family",
                        );
                    } else {
                        evidence_plan.push(format!(
                            "compare independent solver routes {} and {}",
                            request.primary_solver, independent.id
                        ));
                    }
                }
            }
        }

        if request.requested_assurance == AssuranceLevel::Certified {
            let certification = match (
                request.certification_route.as_ref(),
                request.certification_claim.as_ref(),
            ) {
                (Some(route), Some(claim)) => self
                    .certification_routes
                    .iter()
                    .find(|candidate| &candidate.id == route)
                    .map(|capability| (capability, claim)),
                _ => None,
            };
            match certification {
                None if request.certification_route.is_none()
                    || request.certification_claim.is_none() =>
                {
                    fail(
                        &mut failures,
                        PreflightFailureCode::MissingCertificationRoute,
                        "Certified requests require an explicit certification route and claim",
                    );
                }
                None => fail(
                    &mut failures,
                    PreflightFailureCode::UnsupportedCertificationRoute,
                    format!(
                        "certification route {:?} is not registered",
                        request.certification_route
                    ),
                ),
                Some((capability, claim)) => {
                    selected.push(format!("certification:{}", capability.id));
                    if !capability.scalar_backends.contains(&request.scalar_backend)
                        || !capability.claim_kinds.contains(claim)
                    {
                        fail(
                            &mut failures,
                            PreflightFailureCode::UnsupportedCertificationRoute,
                            format!(
                                "certification route {:?} does not support backend {:?} and claim {:?}",
                                capability.id, request.scalar_backend, claim
                            ),
                        );
                    }
                    if scalar
                        .is_some_and(|scalar| !scalar.rigorous_real_enclosures && !scalar.exact)
                    {
                        fail(
                            &mut failures,
                            PreflightFailureCode::NonRigorousCertificationBackend,
                            "Certified execution requires an exact or rigorous-enclosure scalar backend",
                        );
                    }
                    evidence_plan.push(format!(
                        "verify claim {claim} through certification route {}",
                        capability.id
                    ));
                }
            }
        }

        let feasibility = request.resources.assess(request.estimate.clone());
        for violation in feasibility.violations {
            fail(
                &mut failures,
                PreflightFailureCode::InfeasibleResources,
                violation.to_string(),
            );
        }
        if !feasibility.unestimated.is_empty() {
            warnings.push(format!(
                "planner did not estimate bounded resources: {}",
                feasibility
                    .unestimated
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        validate_publication(request, &mut failures, &mut selected, &mut evidence_plan);
        validate_cache_policy(request, &mut failures, &mut selected, &mut evidence_plan);
        if request.cache_mode == CacheAccessMode::RemotePublish
            && request.publication.target == PublicationTarget::None
        {
            fail(
                &mut failures,
                PreflightFailureCode::PublicationNotExplicit,
                "remote publication mode requires an explicit private, public, or both target",
            );
        }
        if request.cache_mode != CacheAccessMode::RemotePublish
            && request.publication.target != PublicationTarget::None
        {
            warnings.push(
                "a publication target was supplied but cache mode does not permit remote publication"
                    .to_owned(),
            );
            fail(
                &mut failures,
                PreflightFailureCode::PublicationNotExplicit,
                "publication target requires remote_publish cache mode",
            );
        }

        PreflightReport {
            accepted: failures.is_empty(),
            effective_config_digest: request.effective_config_digest.clone(),
            selected_capabilities: selected,
            evidence_plan,
            failures,
            warnings,
        }
    }
}

fn validate_cache_policy(
    request: &PreflightRequest,
    failures: &mut Vec<PreflightFailure>,
    selected: &mut Vec<String>,
    evidence_plan: &mut Vec<String>,
) {
    if request.cache_mode == CacheAccessMode::Disabled {
        if request.cache_policy_digest.is_some() || request.cache_validation_mode.is_some() {
            fail(
                failures,
                PreflightFailureCode::InvalidRequest,
                "disabled cache mode cannot carry a cache policy or validation mode",
            );
        }
        return;
    }
    let Some(policy_digest) = &request.cache_policy_digest else {
        fail(
            failures,
            PreflightFailureCode::MissingCachePolicy,
            "enabled cache access requires an exact trust-policy digest",
        );
        return;
    };
    if !policy_digest.is_sha256() {
        fail(
            failures,
            PreflightFailureCode::MissingCachePolicy,
            "cache trust-policy digest must be lowercase SHA-256",
        );
    }
    let Some(validation_mode) = request.cache_validation_mode.as_ref() else {
        fail(
            failures,
            PreflightFailureCode::MissingCachePolicy,
            "enabled cache access requires an explicit validation mode",
        );
        return;
    };
    if *validation_mode == CacheValidationMode::None {
        fail(
            failures,
            PreflightFailureCode::InsufficientCacheValidation,
            "enabled cache access requires Fast or Full validation",
        );
    }
    if request.requested_assurance == AssuranceLevel::Certified
        && *validation_mode != CacheValidationMode::Full
    {
        fail(
            failures,
            PreflightFailureCode::InsufficientCacheValidation,
            "Certified execution requires full cache validation",
        );
    }
    selected.push(format!(
        "cache:{:?}:{validation_mode:?}",
        request.cache_mode
    ));
    evidence_plan.push(format!(
        "apply cache trust policy {policy_digest} with {validation_mode:?} validation"
    ));
}

fn validate_solver(
    solver: Option<&SolverCapability>,
    role: &str,
    request: &PreflightRequest,
    failures: &mut Vec<PreflightFailure>,
    selected: &mut Vec<String>,
) {
    let Some(solver) = solver else {
        fail(
            failures,
            PreflightFailureCode::UnsupportedSolver,
            format!("{role} solver route is not registered"),
        );
        return;
    };
    selected.push(format!("solver:{role}:{}", solver.id));
    if !solver.scalar_backends.contains(&request.scalar_backend) {
        fail(
            failures,
            PreflightFailureCode::UnsupportedScalarBackend,
            format!(
                "{role} solver {:?} does not support scalar backend {:?}",
                solver.id, request.scalar_backend
            ),
        );
    }
    if !solver
        .operator_representations
        .contains(&request.operator_representation)
    {
        fail(
            failures,
            PreflightFailureCode::UnsupportedOperatorRepresentation,
            format!(
                "{role} solver {:?} does not support operator representation {:?}",
                solver.id, request.operator_representation
            ),
        );
    }
    if !solver.target_kinds.contains(&request.target_kind) {
        fail(
            failures,
            PreflightFailureCode::UnsupportedTarget,
            format!(
                "{role} solver {:?} does not support target {:?}",
                solver.id, request.target_kind
            ),
        );
    }
    if request.generalized && !solver.generalized {
        fail(
            failures,
            PreflightFailureCode::UnsupportedGeneralizedProblem,
            format!("{role} solver {:?} is not generalized", solver.id),
        );
    }
    if solver.maximum_assurance < request.requested_assurance {
        fail(
            failures,
            PreflightFailureCode::InsufficientAssuranceCapability,
            format!(
                "{role} solver {:?} supports at most {:?}, requested {:?}",
                solver.id, solver.maximum_assurance, request.requested_assurance
            ),
        );
    }
}

fn validate_publication(
    request: &PreflightRequest,
    failures: &mut Vec<PreflightFailure>,
    selected: &mut Vec<String>,
    evidence_plan: &mut Vec<String>,
) {
    let publication = &request.publication;
    if publication.target == PublicationTarget::None {
        if publication.private_repository.is_some()
            || publication.public_repository.is_some()
            || publication.authority.is_some()
        {
            fail(
                failures,
                PreflightFailureCode::PublicationNotExplicit,
                "publication details were supplied while target is none",
            );
        }
        return;
    }
    let Some(authority) = &publication.authority else {
        fail(
            failures,
            PreflightFailureCode::MissingPublicationAuthority,
            "remote publication requires an authenticated principal and authority policy",
        );
        return;
    };
    let Some(authenticated_principal) = request
        .authenticated_principal
        .as_deref()
        .filter(|principal| !principal.trim().is_empty())
    else {
        fail(
            failures,
            PreflightFailureCode::MissingAuthenticatedPrincipal,
            "remote publication requires a separately authenticated principal",
        );
        return;
    };
    if authenticated_principal != authority.principal {
        fail(
            failures,
            PreflightFailureCode::AuthenticatedPrincipalMismatch,
            format!(
                "authenticated principal {authenticated_principal:?} does not match authority principal {:?}",
                authority.principal
            ),
        );
    }
    if authority.principal.trim().is_empty() || authority.policy_digest.trim().is_empty() {
        fail(
            failures,
            PreflightFailureCode::MissingPublicationAuthority,
            "publication authority principal and policy digest must be nonempty",
        );
    }
    let target_is_permitted = authority.allowed_targets.contains(&publication.target)
        || (publication.target != PublicationTarget::Both
            && authority.allowed_targets.contains(&PublicationTarget::Both));
    if !target_is_permitted {
        fail(
            failures,
            PreflightFailureCode::PublicationAuthorityMismatch,
            format!(
                "authority for principal {:?} does not permit target {:?}",
                authority.principal, publication.target
            ),
        );
    }
    for (required, repository) in [
        (
            publication.target.includes_private(),
            &publication.private_repository,
        ),
        (
            publication.target.includes_public(),
            &publication.public_repository,
        ),
    ] {
        if required {
            match repository {
                None => fail(
                    failures,
                    PreflightFailureCode::InvalidRequest,
                    "publication target is missing its repository",
                ),
                Some(repository) if !authority.allowed_repositories.contains(repository) => fail(
                    failures,
                    PreflightFailureCode::PublicationRepositoryNotAuthorized,
                    format!("repository {repository:?} is not authorized for publication"),
                ),
                Some(_) => {}
            }
        } else if repository.is_some() {
            fail(
                failures,
                PreflightFailureCode::InvalidRequest,
                "a repository was supplied for an unrequested publication target",
            );
        }
    }
    selected.push(format!(
        "publication:{:?}:{:?}",
        publication.target, authority.mode
    ));
    evidence_plan.push(format!(
        "validate publication policy {} and emit per-target receipts",
        authority.policy_digest
    ));
}

fn fail(
    failures: &mut Vec<PreflightFailure>,
    code: PreflightFailureCode,
    message: impl Into<String>,
) {
    failures.push(PreflightFailure {
        code,
        message: message.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResourceEstimate;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn catalog() -> CapabilityCatalog {
        CapabilityCatalog {
            scalar_backends: vec![
                ScalarCapability {
                    id: "f64".to_owned(),
                    supported_platforms: set(&["windows", "linux"]),
                    maximum_precision_bits: Some(53),
                    arbitrary_precision: false,
                    rigorous_real_enclosures: false,
                    rigorous_complex_enclosures: false,
                    exact: false,
                },
                ScalarCapability {
                    id: "arb".to_owned(),
                    supported_platforms: set(&["linux"]),
                    maximum_precision_bits: None,
                    arbitrary_precision: true,
                    rigorous_real_enclosures: true,
                    rigorous_complex_enclosures: true,
                    exact: false,
                },
            ],
            solvers: vec![
                SolverCapability {
                    id: "dense_qr".to_owned(),
                    algorithm_family: "householder_qr".to_owned(),
                    scalar_backends: set(&["f64", "arb"]),
                    operator_representations: set(&["dense"]),
                    target_kinds: set(&["algebraic_smallest"]),
                    generalized: false,
                    maximum_assurance: AssuranceLevel::Certified,
                    checkpoint_supported: false,
                },
                SolverCapability {
                    id: "sturm".to_owned(),
                    algorithm_family: "sturm_count".to_owned(),
                    scalar_backends: set(&["f64", "arb"]),
                    operator_representations: set(&["dense"]),
                    target_kinds: set(&["algebraic_smallest"]),
                    generalized: false,
                    maximum_assurance: AssuranceLevel::Certified,
                    checkpoint_supported: true,
                },
            ],
            certification_routes: vec![CertificationCapability {
                id: "interval_inertia".to_owned(),
                scalar_backends: set(&["arb"]),
                claim_kinds: set(&["selected_eigenvalue"]),
            }],
        }
    }

    fn base_request() -> PreflightRequest {
        PreflightRequest {
            effective_config_digest: ConfigDigest("a".repeat(64)),
            platform: "linux".to_owned(),
            scalar_backend: "f64".to_owned(),
            precision_bits: 53,
            operator_representation: "dense".to_owned(),
            target_kind: "algebraic_smallest".to_owned(),
            generalized: false,
            primary_solver: "dense_qr".to_owned(),
            independent_solver: None,
            requested_assurance: AssuranceLevel::Computed,
            certification_route: None,
            certification_claim: None,
            cache_mode: CacheAccessMode::ReadOnly,
            cache_policy_digest: Some(ConfigDigest("c".repeat(64))),
            cache_validation_mode: Some(CacheValidationMode::Full),
            authenticated_principal: None,
            publication: PublicationPreflightRequest::default(),
            resources: ResourcePolicy::default(),
            estimate: ResourceEstimate::default(),
        }
    }

    #[test]
    fn certified_request_rejects_nonrigorous_backend() {
        let mut request = base_request();
        request.requested_assurance = AssuranceLevel::Certified;
        request.independent_solver = Some("sturm".to_owned());
        request.certification_route = Some("interval_inertia".to_owned());
        request.certification_claim = Some("selected_eigenvalue".to_owned());
        let report = catalog().preflight(&request);
        assert!(!report.accepted);
        assert!(report.failures.iter().any(|failure| {
            failure.code == PreflightFailureCode::NonRigorousCertificationBackend
        }));
    }

    #[test]
    fn malformed_effective_configuration_digest_is_rejected() {
        let mut request = base_request();
        request.effective_config_digest = ConfigDigest("not-a-digest".to_owned());
        let report = catalog().preflight(&request);
        assert!(report.failures.iter().any(|failure| {
            failure.code == PreflightFailureCode::InvalidRequest
                && failure.message.contains("configuration digest")
        }));
    }

    #[test]
    fn public_publication_requires_authority() {
        let mut request = base_request();
        request.cache_mode = CacheAccessMode::RemotePublish;
        request.publication.target = PublicationTarget::Public;
        request.publication.public_repository = Some("team/public".to_owned());
        let report = catalog().preflight(&request);
        assert!(!report.accepted);
        assert!(report
            .failures
            .iter()
            .any(|failure| { failure.code == PreflightFailureCode::MissingPublicationAuthority }));
    }

    #[test]
    fn explicit_owner_publication_is_accepted() {
        let mut request = base_request();
        request.cache_mode = CacheAccessMode::RemotePublish;
        request.authenticated_principal = Some("owner".to_owned());
        request.publication = PublicationPreflightRequest {
            target: PublicationTarget::Both,
            private_repository: Some("example-org/restricted-cache".to_owned()),
            public_repository: Some("team/public".to_owned()),
            authority: Some(PublicationAuthority {
                principal: "owner".to_owned(),
                mode: PublicationAuthorityMode::OwnerDirect,
                allowed_targets: [PublicationTarget::Both].into_iter().collect(),
                allowed_repositories: set(&["example-org/restricted-cache", "team/public"]),
                policy_digest: "b".repeat(64),
            }),
        };
        let report = catalog().preflight(&request);
        assert!(report.accepted, "{:?}", report.failures);
        assert!(report
            .evidence_plan
            .iter()
            .any(|entry| entry.contains("per-target receipts")));
    }

    #[test]
    fn enabled_cache_requires_exact_policy_and_certified_full_validation() {
        let mut request = base_request();
        request.cache_policy_digest = None;
        let report = catalog().preflight(&request);
        assert!(report
            .failures
            .iter()
            .any(|failure| failure.code == PreflightFailureCode::MissingCachePolicy));

        request.cache_policy_digest = Some(ConfigDigest("c".repeat(64)));
        request.cache_validation_mode = Some(CacheValidationMode::None);
        let report = catalog().preflight(&request);
        assert!(report
            .failures
            .iter()
            .any(|failure| { failure.code == PreflightFailureCode::InsufficientCacheValidation }));

        request.cache_validation_mode = Some(CacheValidationMode::Fast);
        request.requested_assurance = AssuranceLevel::Certified;
        request.independent_solver = Some("sturm".to_owned());
        request.certification_route = Some("interval_inertia".to_owned());
        request.certification_claim = Some("selected_eigenvalue".to_owned());
        let report = catalog().preflight(&request);
        assert!(report
            .failures
            .iter()
            .any(|failure| { failure.code == PreflightFailureCode::InsufficientCacheValidation }));
    }

    #[test]
    fn publication_principal_must_match_authenticated_identity() {
        let mut request = base_request();
        request.cache_mode = CacheAccessMode::RemotePublish;
        request.authenticated_principal = Some("different-user".to_owned());
        request.publication = PublicationPreflightRequest {
            target: PublicationTarget::Public,
            private_repository: None,
            public_repository: Some("team/public".to_owned()),
            authority: Some(PublicationAuthority {
                principal: "owner".to_owned(),
                mode: PublicationAuthorityMode::OwnerDirect,
                allowed_targets: [PublicationTarget::Public].into_iter().collect(),
                allowed_repositories: set(&["team/public"]),
                policy_digest: "b".repeat(64),
            }),
        };
        let report = catalog().preflight(&request);
        assert!(report.failures.iter().any(|failure| {
            failure.code == PreflightFailureCode::AuthenticatedPrincipalMismatch
        }));
    }

    #[test]
    fn dual_target_authority_also_authorizes_one_of_its_targets() {
        let mut request = base_request();
        request.cache_mode = CacheAccessMode::RemotePublish;
        request.authenticated_principal = Some("owner".to_owned());
        request.publication.target = PublicationTarget::Private;
        request.publication.private_repository = Some("example-org/restricted-cache".to_owned());
        request.publication.authority = Some(PublicationAuthority {
            principal: "owner".to_owned(),
            mode: PublicationAuthorityMode::OwnerDirect,
            allowed_targets: [PublicationTarget::Both].into_iter().collect(),
            allowed_repositories: ["example-org/restricted-cache".to_owned()]
                .into_iter()
                .collect(),
            policy_digest: "b".repeat(64),
        });
        assert!(catalog().preflight(&request).accepted);
    }

    #[test]
    fn crosscheck_rejects_same_algorithm_family() {
        let mut catalog = catalog();
        let mut alias = catalog.solvers[0].clone();
        alias.id = "dense_qr_alias".to_owned();
        catalog.solvers.push(alias);
        let mut request = base_request();
        request.requested_assurance = AssuranceLevel::CrossChecked;
        request.independent_solver = Some("dense_qr_alias".to_owned());
        let report = catalog.preflight(&request);
        assert!(!report.accepted);
        assert!(report
            .failures
            .iter()
            .any(|failure| { failure.code == PreflightFailureCode::NonIndependentRoute }));
    }
}
