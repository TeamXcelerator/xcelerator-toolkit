// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Portable certificate records and independent structural verification.
//!
//! Numerical backends produce these records; lightweight consumers can verify
//! dimensions, hashes, enclosure ordering, exact rational inequalities, and
//! completeness metadata without rerunning the discovery solve.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use xc_cache::{sha256_hex, ArtifactKey, ContentDigest, ToolkitVersion};
use xc_core::{
    ApproximationLedger, ArtifactCitationMetadata, AssuranceLevel, DecimalLiteral, SolverProvenance,
};

pub fn certification_artifact_reuse_plan() -> xc_core::ArtifactReusePlan {
    use xc_core::{ArtifactReuseNode, ArtifactReusePlan};
    let node = |kind: &str, dependencies: &[&str], invalidated_by: &[&str]| ArtifactReuseNode {
        kind: kind.to_owned(),
        independently_cacheable: true,
        dependencies: dependencies
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        invalidated_by: invalidated_by
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    };
    ArtifactReusePlan {
        schema_version: 1,
        domain: "certification".to_owned(),
        semantics_version: "certificate-v0.13.0-v1".to_owned(),
        artifacts: vec![
            node(
                "canonical_claim_inputs",
                &[],
                &["claim_semantics", "input_digests"],
            ),
            node(
                "inertia_evidence",
                &["canonical_claim_inputs"],
                &["interval_backend", "precision_bits"],
            ),
            node(
                "eigenvalue_enclosure",
                &["canonical_claim_inputs"],
                &["selected_index", "target_width", "bisection_policy"],
            ),
            node(
                "spectral_gap",
                &["eigenvalue_enclosure"],
                &["cluster_selection"],
            ),
            node(
                "verification_report",
                &["inertia_evidence", "eigenvalue_enclosure", "spectral_gap"],
                &["verification_policy"],
            ),
            node(
                "certificate_bundle",
                &["canonical_claim_inputs", "verification_report"],
                &["bundle_schema", "assurance_policy"],
            ),
        ],
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CertificateError {
    Invalid(String),
    Unsupported(String),
    VerificationFailed(String),
}

impl Display for CertificateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid certificate: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported certificate: {message}"),
            Self::VerificationFailed(message) => {
                write!(f, "certificate verification failed: {message}")
            }
        }
    }
}

impl Error for CertificateError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecimalInterval {
    pub lower: String,
    pub upper: String,
}

impl DecimalInterval {
    fn endpoints(&self) -> Result<(DecimalLiteral, DecimalLiteral), CertificateError> {
        let lower = DecimalLiteral::new(self.lower.clone()).map_err(|error| {
            CertificateError::Invalid(format!("invalid lower decimal endpoint: {error}"))
        })?;
        let upper = DecimalLiteral::new(self.upper.clone()).map_err(|error| {
            CertificateError::Invalid(format!("invalid upper decimal endpoint: {error}"))
        })?;
        Ok((lower, upper))
    }

    pub fn validate_order(&self) -> Result<(), CertificateError> {
        let (lower, upper) = self.endpoints()?;
        if lower.cmp_numeric(&upper).map_err(|error| {
            CertificateError::Invalid(format!("invalid decimal interval: {error}"))
        })? == Ordering::Greater
        {
            return Err(CertificateError::Invalid(
                "decimal interval must satisfy lower <= upper".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn is_disjoint_from(&self, other: &Self) -> Result<bool, CertificateError> {
        self.validate_order()?;
        other.validate_order()?;
        let (self_lower, self_upper) = self.endpoints()?;
        let (other_lower, other_upper) = other.endpoints()?;
        let left_disjoint = self_upper.cmp_numeric(&other_lower).map_err(|error| {
            CertificateError::Invalid(format!("invalid decimal interval: {error}"))
        })? == Ordering::Less;
        let right_disjoint = other_upper.cmp_numeric(&self_lower).map_err(|error| {
            CertificateError::Invalid(format!("invalid decimal interval: {error}"))
        })? == Ordering::Less;
        Ok(left_disjoint || right_disjoint)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactRationalRecord {
    /// Canonical integer numerator, base 10.
    pub numerator: String,
    /// Canonical positive integer denominator, base 10.
    pub denominator: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactRationalIntervalRecord {
    pub lower: ExactRationalRecord,
    pub upper: ExactRationalRecord,
}

impl ExactRationalIntervalRecord {
    pub fn validate_syntax(&self) -> Result<(), CertificateError> {
        self.lower.validate_syntax()?;
        self.upper.validate_syntax()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationErrorSource {
    MatrixAssembly,
    Quadrature,
    Truncation,
    Banding,
    Basis,
    Transformation,
    Rounding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CertificationErrorComponent {
    pub source: CertificationErrorSource,
    pub absolute_bound: ExactRationalRecord,
    pub affects_claim: bool,
    pub evidence_digest: ContentDigest,
    pub method: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CertificationErrorBudget {
    pub required_sources: Vec<CertificationErrorSource>,
    pub components: Vec<CertificationErrorComponent>,
}

impl CertificationErrorBudget {
    pub fn validate_declared_coverage(&self) -> Result<(), CertificateError> {
        let required = self
            .required_sources
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if required.len() != self.required_sources.len() {
            return Err(CertificateError::Invalid(
                "certification error budget repeats a required source".to_owned(),
            ));
        }
        let mut observed = std::collections::BTreeSet::new();
        for component in &self.components {
            component.absolute_bound.validate_syntax()?;
            if !component.evidence_digest.validate() || component.method.trim().is_empty() {
                return Err(CertificateError::Invalid(
                    "certification error component lacks evidence or method".to_owned(),
                ));
            }
            if !observed.insert(component.source) {
                return Err(CertificateError::Invalid(
                    "certification error budget repeats a component source".to_owned(),
                ));
            }
            if required.contains(&component.source) != component.affects_claim {
                return Err(CertificateError::Invalid(format!(
                    "error source {:?} has inconsistent claim-impact declaration",
                    component.source
                )));
            }
        }
        let missing = required.difference(&observed).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(CertificateError::VerificationFailed(format!(
                "certification error budget omits required sources: {missing:?}"
            )));
        }
        Ok(())
    }
}

impl ExactRationalRecord {
    pub fn validate_syntax(&self) -> Result<(), CertificateError> {
        fn valid_integer(value: &str, allow_negative: bool) -> bool {
            let bytes = value.as_bytes();
            if bytes.is_empty() {
                return false;
            }
            let start = if allow_negative && bytes[0] == b'-' {
                1
            } else {
                0
            };
            start < bytes.len() && bytes[start..].iter().all(u8::is_ascii_digit)
        }
        if !valid_integer(&self.numerator, true) {
            return Err(CertificateError::Invalid(
                "invalid exact rational numerator".to_owned(),
            ));
        }
        if !valid_integer(&self.denominator, false)
            || self.denominator.trim_start_matches('0').is_empty()
        {
            return Err(CertificateError::Invalid(
                "exact rational denominator must be a positive integer".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InertiaCertificate {
    pub dimension: usize,
    pub positive: usize,
    pub negative: usize,
    pub zero_or_unresolved: usize,
    pub matrix_digest: ContentDigest,
    pub scalar_backend: String,
    pub precision_bits: u32,
    pub pivot_enclosures_digest: Option<ContentDigest>,
}

/// Self-contained exact-endpoint interval-inertia proof. The matrix entries
/// include assembly uncertainty; an independent HP verifier reconstructs the
/// intervals and reruns interval LDL^T without access to the producer state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortableIntervalInertiaCertificate {
    pub schema_version: u32,
    pub certificate_id: ContentDigest,
    pub toolkit_version: ToolkitVersion,
    pub configuration: BTreeMap<String, String>,
    pub dimension: usize,
    pub precision_bits: u32,
    pub scalar_backend: String,
    pub matrix_digest: ContentDigest,
    pub assembly_evidence_digest: ContentDigest,
    pub matrix_row_major: Vec<ExactRationalIntervalRecord>,
    pub positive: usize,
    pub negative: usize,
    pub zero_or_unresolved: usize,
    pub pivot_enclosures: Vec<ExactRationalIntervalRecord>,
    pub notes: Vec<String>,
}

impl PortableIntervalInertiaCertificate {
    pub fn computed_matrix_digest(&self) -> Result<ContentDigest, CertificateError> {
        let bytes =
            serde_json::to_vec(&(self.dimension, &self.matrix_row_major)).map_err(|error| {
                CertificateError::Invalid(format!("failed to serialize interval matrix: {error}"))
            })?;
        Ok(ContentDigest(sha256_hex(&bytes)))
    }

    pub fn computed_certificate_id(&self) -> Result<ContentDigest, CertificateError> {
        let mut canonical = self.clone();
        canonical.certificate_id = ContentDigest("00".repeat(32));
        let bytes = serde_json::to_vec(&canonical).map_err(|error| {
            CertificateError::Invalid(format!(
                "failed to serialize portable inertia certificate: {error}"
            ))
        })?;
        Ok(ContentDigest(sha256_hex(&bytes)))
    }

    pub fn refresh_certificate_id(&mut self) -> Result<(), CertificateError> {
        self.certificate_id = self.computed_certificate_id()?;
        Ok(())
    }

    pub fn validate_structure(&self) -> Result<(), CertificateError> {
        if self.schema_version == 0
            || self.dimension == 0
            || self.matrix_row_major.len() != self.dimension.saturating_mul(self.dimension)
        {
            return Err(CertificateError::Invalid(
                "portable inertia certificate has invalid schema or dimensions".to_owned(),
            ));
        }
        if self.precision_bits < 32
            || self.scalar_backend.trim().is_empty()
            || self.configuration.is_empty()
        {
            return Err(CertificateError::Invalid(
                "portable inertia certificate lacks a valid precision or backend".to_owned(),
            ));
        }
        if self.positive + self.negative + self.zero_or_unresolved != self.dimension {
            return Err(CertificateError::Invalid(
                "portable inertia counts do not sum to the matrix dimension".to_owned(),
            ));
        }
        if self.pivot_enclosures.len()
            != self.dimension.saturating_sub(self.zero_or_unresolved)
                + usize::from(self.zero_or_unresolved > 0)
        {
            return Err(CertificateError::Invalid(
                "portable inertia pivot count is inconsistent with its unresolved suffix"
                    .to_owned(),
            ));
        }
        for interval in self.matrix_row_major.iter().chain(&self.pivot_enclosures) {
            interval.validate_syntax()?;
        }
        if !self.matrix_digest.validate()
            || !self.assembly_evidence_digest.validate()
            || !self.certificate_id.validate()
        {
            return Err(CertificateError::Invalid(
                "portable inertia certificate contains an invalid digest".to_owned(),
            ));
        }
        if self.computed_matrix_digest()? != self.matrix_digest {
            return Err(CertificateError::VerificationFailed(
                "portable inertia matrix digest mismatch".to_owned(),
            ));
        }
        if self.computed_certificate_id()? != self.certificate_id {
            return Err(CertificateError::VerificationFailed(
                "portable inertia certificate identifier mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EigenvalueCountBoundaryCertificate {
    pub threshold: ExactRationalRecord,
    pub count_below: usize,
    pub positive_above: usize,
    pub pivot_enclosures: Vec<ExactRationalIntervalRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntervalEigenvalueCountCertificate {
    pub dimension: usize,
    pub lower_boundary: Option<EigenvalueCountBoundaryCertificate>,
    pub upper_boundary: EigenvalueCountBoundaryCertificate,
    pub eigenvalue_count: usize,
    pub interval_semantics: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum IntervalEigenvalueCountResult {
    Conclusive {
        certificate: IntervalEigenvalueCountCertificate,
    },
    Inconclusive {
        boundary: String,
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactSelectedEigenvalueEnclosure {
    pub dimension: usize,
    pub matrix_digest: ContentDigest,
    pub requested_index: usize,
    pub first_enclosed_index: usize,
    pub last_enclosed_index: usize,
    pub lower: ExactRationalRecord,
    pub upper: ExactRationalRecord,
    pub lower_boundary: EigenvalueCountBoundaryCertificate,
    pub upper_boundary: EigenvalueCountBoundaryCertificate,
    pub bisection_steps: usize,
    pub simple: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum SelectedEigenvalueEnclosureResult {
    Conclusive {
        certificate: Box<ExactSelectedEigenvalueEnclosure>,
    },
    Inconclusive {
        boundary: String,
        reason: String,
    },
}

/// Self-contained exact selected-eigenvalue proof for offline replay.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortableSelectedEigenvalueCertificate {
    pub schema_version: u32,
    pub matrix_row_major: Vec<ExactRationalIntervalRecord>,
    pub enclosure: ExactSelectedEigenvalueEnclosure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExactSpectralGapCertificate {
    pub lower_cluster: ExactSelectedEigenvalueEnclosure,
    pub upper_cluster: ExactSelectedEigenvalueEnclosure,
    pub certified_lower_bound: ExactRationalRecord,
}

impl InertiaCertificate {
    pub fn validate(&self) -> Result<(), CertificateError> {
        if self.positive + self.negative + self.zero_or_unresolved != self.dimension {
            return Err(CertificateError::Invalid(
                "inertia counts do not sum to matrix dimension".to_owned(),
            ));
        }
        if self.precision_bits < 32 {
            return Err(CertificateError::Invalid(
                "certificate precision is implausibly small".to_owned(),
            ));
        }
        if !self.matrix_digest.validate() {
            return Err(CertificateError::Invalid(
                "matrix digest is not a valid SHA-256 digest".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn certifies_positive_definite(&self) -> bool {
        self.positive == self.dimension && self.negative == 0 && self.zero_or_unresolved == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EigenvalueEnclosure {
    pub first_index: usize,
    pub last_index: usize,
    pub interval: DecimalInterval,
    pub multiplicity_lower: usize,
    pub multiplicity_upper: usize,
}

impl EigenvalueEnclosure {
    pub fn validate(&self) -> Result<(), CertificateError> {
        if self.first_index > self.last_index {
            return Err(CertificateError::Invalid(
                "eigenvalue enclosure index range is reversed".to_owned(),
            ));
        }
        if self.multiplicity_lower == 0 || self.multiplicity_lower > self.multiplicity_upper {
            return Err(CertificateError::Invalid(
                "invalid multiplicity bounds".to_owned(),
            ));
        }
        self.interval.validate_order()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SpectralGapCertificate {
    pub lower_cluster: EigenvalueEnclosure,
    pub upper_cluster: EigenvalueEnclosure,
    pub certified_lower_bound: String,
}

impl SpectralGapCertificate {
    pub fn validate(&self) -> Result<(), CertificateError> {
        self.lower_cluster.validate()?;
        self.upper_cluster.validate()?;
        if !self
            .lower_cluster
            .interval
            .is_disjoint_from(&self.upper_cluster.interval)?
        {
            return Err(CertificateError::VerificationFailed(
                "eigenvalue clusters overlap".to_owned(),
            ));
        }
        let bound = DecimalLiteral::new(self.certified_lower_bound.clone()).map_err(|error| {
            CertificateError::Invalid(format!("invalid gap lower bound: {error}"))
        })?;
        let zero = DecimalLiteral::new("0").expect("zero is a valid decimal literal");
        if bound.cmp_numeric(&zero).map_err(|error| {
            CertificateError::Invalid(format!("invalid gap lower bound: {error}"))
        })? != Ordering::Greater
        {
            return Err(CertificateError::VerificationFailed(
                "gap lower bound must be strictly positive".to_owned(),
            ));
        }
        // Portable structural verification proves separation and positivity.
        // Verifying that an arbitrary decimal bound is no larger than the exact
        // endpoint difference belongs to the active exact/ball backend and its
        // evidence digest; it must not be approximated through f64 here.
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "claim")]
pub enum CertificateClaim {
    PositiveDefiniteMatrix,
    MatrixInertia,
    EigenvalueEnclosure,
    SpectralGap,
    GeneralizedRayleighLowerBound,
    RootIsolation,
    RootCount,
    SpectralWindowCompleteness,
    Custom { name: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CertifiedArtifactRef {
    pub key: ArtifactKey,
    pub content_digest: ContentDigest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificateBundle {
    pub schema_version: u32,
    pub certificate_id: ContentDigest,
    pub claim: CertificateClaim,
    pub assurance: AssuranceLevel,
    pub toolkit_version: ToolkitVersion,
    /// Preferred citation and author attribution for this exact bundle.
    pub citation: Option<ArtifactCitationMetadata>,
    pub provenance: SolverProvenance,
    pub inputs: Vec<CertifiedArtifactRef>,
    pub inertia: Option<InertiaCertificate>,
    pub eigenvalue_enclosures: Vec<EigenvalueEnclosure>,
    pub spectral_gap: Option<SpectralGapCertificate>,
    pub exact_records: BTreeMap<String, ExactRationalRecord>,
    pub evidence_digests: BTreeMap<String, ContentDigest>,
    pub approximation_ledger: ApproximationLedger,
    pub assumptions: Vec<String>,
    pub notes: Vec<String>,
}

impl CertificateBundle {
    /// Compute the stable identifier from the complete serialized bundle with
    /// the identifier field replaced by a fixed zero digest.
    pub fn computed_certificate_id(&self) -> Result<ContentDigest, CertificateError> {
        let mut canonical = self.clone();
        canonical.certificate_id = ContentDigest("00".repeat(32));
        let bytes = serde_json::to_vec(&canonical).map_err(|error| {
            CertificateError::Invalid(format!("failed to serialize certificate: {error}"))
        })?;
        Ok(ContentDigest(sha256_hex(&bytes)))
    }

    pub fn refresh_certificate_id(&mut self) -> Result<(), CertificateError> {
        self.certificate_id = self.computed_certificate_id()?;
        Ok(())
    }

    /// Attaches validated citation metadata and refreshes the identity that
    /// binds it to this exact certificate bundle.
    pub fn attach_citation(
        &mut self,
        citation: ArtifactCitationMetadata,
    ) -> Result<(), CertificateError> {
        citation
            .validate()
            .map_err(|error| CertificateError::Invalid(error.to_string()))?;
        self.citation = Some(citation);
        self.refresh_certificate_id()
    }

    pub fn validate_structure(&self) -> Result<(), CertificateError> {
        xc_core::validate_secret_free(self, "certificate bundle")
            .map_err(|error| CertificateError::Invalid(error.to_string()))?;
        if self.schema_version == 0 {
            return Err(CertificateError::Invalid(
                "certificate schema_version must be positive".to_owned(),
            ));
        }
        if self.assurance != AssuranceLevel::Certified {
            return Err(CertificateError::Invalid(
                "certificate bundle must use Certified assurance".to_owned(),
            ));
        }
        if let Some(citation) = &self.citation {
            citation
                .validate()
                .map_err(|error| CertificateError::Invalid(error.to_string()))?;
            if citation.software_version != self.toolkit_version.to_string() {
                return Err(CertificateError::Invalid(
                    "certificate citation software version differs from toolkit version".to_owned(),
                ));
            }
        }
        self.approximation_ledger
            .validate_for_assurance(AssuranceLevel::Certified)
            .map_err(|error| CertificateError::VerificationFailed(error.to_string()))?;
        if !self.certificate_id.validate() {
            return Err(CertificateError::Invalid(
                "certificate_id is not a valid SHA-256 digest".to_owned(),
            ));
        }
        let computed = self.computed_certificate_id()?;
        if computed != self.certificate_id {
            return Err(CertificateError::VerificationFailed(format!(
                "certificate identifier mismatch: expected {}, computed {}",
                self.certificate_id, computed
            )));
        }
        for input in &self.inputs {
            if !input.content_digest.validate() {
                return Err(CertificateError::Invalid(
                    "input artifact contains an invalid digest".to_owned(),
                ));
            }
        }
        if let Some(inertia) = &self.inertia {
            inertia.validate()?;
        }
        for enclosure in &self.eigenvalue_enclosures {
            enclosure.validate()?;
        }
        if let Some(gap) = &self.spectral_gap {
            gap.validate()?;
        }
        for record in self.exact_records.values() {
            record.validate_syntax()?;
        }
        if matches!(self.claim, CertificateClaim::PositiveDefiniteMatrix)
            && !self
                .inertia
                .as_ref()
                .is_some_and(InertiaCertificate::certifies_positive_definite)
        {
            return Err(CertificateError::VerificationFailed(
                "positive-definite claim lacks a conclusive positive inertia certificate"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub valid: bool,
    pub checks: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Independently verifies the structure and claim invariants of a certificate bundle.
///
/// # Mathematical semantics
/// Recomputes the bundle identity and checks the finite claim's required
/// inertia, enclosure, gap, exact-record, evidence, and approximation-ledger
/// relationships as applicable.
///
/// # Precision
/// Verification consumes the exact portable records stored in the bundle; it
/// does not silently recompute them in binary64 or claim more precision than
/// the certificate records.
///
/// # Failure states
/// All validation failures are returned in `VerificationReport.errors` with
/// `valid = false`; malformed input is never accepted with warnings alone.
///
/// # Assurance and validity
/// A valid report establishes only the encoded finite claim under its recorded
/// assumptions. It does not turn finite spectral evidence into an infinite-
/// dimensional theorem.
///
/// # Cache effects
/// Verification is pure with respect to the cache. Publication and reuse must
/// separately bind the verified bundle digest in artifact provenance.
///
/// # Example
/// Compiled example: `crates/xc-certify/examples/finite_certificate.rs`.
pub fn verify_bundle(bundle: &CertificateBundle) -> VerificationReport {
    match bundle.validate_structure() {
        Ok(()) => VerificationReport {
            valid: true,
            checks: vec!["certificate structure and claim-specific invariants verified".to_owned()],
            warnings: bundle
                .assumptions
                .iter()
                .map(|assumption| format!("assumption: {assumption}"))
                .collect(),
            errors: Vec::new(),
        },
        Err(error) => VerificationReport {
            valid: false,
            checks: Vec::new(),
            warnings: Vec::new(),
            errors: vec![error.to_string()],
        },
    }
}

#[cfg(feature = "hp")]
pub mod exact {
    use super::{
        sha256_hex, CertificateError, CertificationErrorBudget, ContentDigest,
        EigenvalueCountBoundaryCertificate, ExactRationalIntervalRecord, ExactRationalRecord,
        ExactSelectedEigenvalueEnclosure, ExactSpectralGapCertificate,
        IntervalEigenvalueCountCertificate, IntervalEigenvalueCountResult,
        PortableIntervalInertiaCertificate, PortableSelectedEigenvalueCertificate,
        SelectedEigenvalueEnclosureResult, ToolkitVersion, VerificationReport,
    };
    use rug::Rational;
    use xc_numerics::interval::RationalInterval;

    pub fn parse(record: &ExactRationalRecord) -> Result<Rational, CertificateError> {
        record.validate_syntax()?;
        let text = format!("{}/{}", record.numerator, record.denominator);
        let incomplete = Rational::parse(&text).map_err(|e| {
            CertificateError::Invalid(format!("failed to parse exact rational: {e}"))
        })?;
        Ok(Rational::from(incomplete))
    }

    pub fn rational_record(value: &Rational) -> ExactRationalRecord {
        ExactRationalRecord {
            numerator: value.numer().to_string(),
            denominator: value.denom().to_string(),
        }
    }

    pub fn interval_record(value: &RationalInterval) -> ExactRationalIntervalRecord {
        ExactRationalIntervalRecord {
            lower: rational_record(value.lower()),
            upper: rational_record(value.upper()),
        }
    }

    pub fn parse_interval(
        record: &ExactRationalIntervalRecord,
    ) -> Result<RationalInterval, CertificateError> {
        record.validate_syntax()?;
        RationalInterval::new(parse(&record.lower)?, parse(&record.upper)?)
            .map_err(|error| CertificateError::Invalid(error.to_string()))
    }

    pub fn build_portable_interval_inertia_certificate(
        matrix: &[RationalInterval],
        dimension: usize,
        precision_bits: u32,
        scalar_backend: impl Into<String>,
        assembly_evidence_digest: ContentDigest,
        configuration: std::collections::BTreeMap<String, String>,
        notes: Vec<String>,
    ) -> Result<PortableIntervalInertiaCertificate, CertificateError> {
        if !assembly_evidence_digest.validate() {
            return Err(CertificateError::Invalid(
                "assembly evidence digest is invalid".to_owned(),
            ));
        }
        let inertia = interval_symmetric_ldlt_inertia(matrix, dimension)?;
        let (positive, negative, zero_or_unresolved, pivots) = match inertia {
            IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            } => (positive, negative, 0, pivot_enclosures),
            IntervalInertiaResult::Inconclusive {
                positive,
                negative,
                zero_or_unresolved,
                pivot_enclosures,
                ..
            } => (positive, negative, zero_or_unresolved, pivot_enclosures),
        };
        let mut certificate = PortableIntervalInertiaCertificate {
            schema_version: 1,
            certificate_id: ContentDigest("00".repeat(32)),
            toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).map_err(|error| {
                CertificateError::Invalid(format!("invalid toolkit package version: {error}"))
            })?,
            configuration,
            dimension,
            precision_bits,
            scalar_backend: scalar_backend.into(),
            matrix_digest: ContentDigest("00".repeat(32)),
            assembly_evidence_digest,
            matrix_row_major: matrix.iter().map(interval_record).collect(),
            positive,
            negative,
            zero_or_unresolved,
            pivot_enclosures: pivots.iter().map(interval_record).collect(),
            notes,
        };
        certificate.matrix_digest = certificate.computed_matrix_digest()?;
        certificate.refresh_certificate_id()?;
        certificate.validate_structure()?;
        Ok(certificate)
    }

    fn verify_portable_interval_inertia_exact(
        certificate: &PortableIntervalInertiaCertificate,
    ) -> Result<(), CertificateError> {
        certificate.validate_structure()?;
        let matrix = certificate
            .matrix_row_major
            .iter()
            .map(parse_interval)
            .collect::<Result<Vec<_>, _>>()?;
        let recomputed = interval_symmetric_ldlt_inertia(&matrix, certificate.dimension)?;
        let (positive, negative, unresolved, pivots) = match recomputed {
            IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            } => (positive, negative, 0, pivot_enclosures),
            IntervalInertiaResult::Inconclusive {
                positive,
                negative,
                zero_or_unresolved,
                pivot_enclosures,
                ..
            } => (positive, negative, zero_or_unresolved, pivot_enclosures),
        };
        let recorded_pivots = certificate
            .pivot_enclosures
            .iter()
            .map(parse_interval)
            .collect::<Result<Vec<_>, _>>()?;
        if (positive, negative, unresolved)
            != (
                certificate.positive,
                certificate.negative,
                certificate.zero_or_unresolved,
            )
            || pivots != recorded_pivots
        {
            return Err(CertificateError::VerificationFailed(
                "independent interval LDLT replay differs from the recorded inertia proof"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub fn verify_portable_interval_inertia_certificate(
        certificate: &PortableIntervalInertiaCertificate,
    ) -> VerificationReport {
        match verify_portable_interval_inertia_exact(certificate) {
            Ok(()) => VerificationReport {
                valid: true,
                checks: vec![
                    "certificate and matrix digests verified".to_owned(),
                    "exact matrix endpoint ordering and symmetry verified".to_owned(),
                    "interval LDLT inertia and every pivot enclosure independently replayed"
                        .to_owned(),
                ],
                warnings: Vec::new(),
                errors: Vec::new(),
            },
            Err(error) => VerificationReport {
                valid: false,
                checks: Vec::new(),
                warnings: Vec::new(),
                errors: vec![error.to_string()],
            },
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExactInertia {
        pub positive: usize,
        pub negative: usize,
        pub zero_or_unresolved: usize,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum IntervalInertiaResult {
        Conclusive {
            positive: usize,
            negative: usize,
            pivot_enclosures: Vec<RationalInterval>,
        },
        Inconclusive {
            pivot_index: usize,
            positive: usize,
            negative: usize,
            zero_or_unresolved: usize,
            pivot_enclosures: Vec<RationalInterval>,
            reason: String,
        },
    }

    impl IntervalInertiaResult {
        pub fn is_conclusive(&self) -> bool {
            matches!(self, Self::Conclusive { .. })
        }
    }

    struct CertifiedTwoByTwoPivot {
        lower_eigenvalue: RationalInterval,
        upper_eigenvalue: RationalInterval,
        inverse_11: RationalInterval,
        inverse_12: RationalInterval,
        inverse_22: RationalInterval,
    }

    fn certify_two_by_two_pivot(
        matrix: &[RationalInterval],
        dimension: usize,
        first: usize,
        second: usize,
    ) -> Result<Option<CertifiedTwoByTwoPivot>, CertificateError> {
        let a = &matrix[first * dimension + first];
        let b = &matrix[first * dimension + second];
        let c = &matrix[second * dimension + second];
        let determinant = a.mul(c).sub(&b.square());
        if determinant.contains_zero() {
            return Ok(None);
        }
        let four = RationalInterval::point(Rational::from((4, 1)));
        let two = RationalInterval::point(Rational::from((2, 1)));
        let discriminant = a.sub(c).square().add(&b.square().mul(&four));
        let root = discriminant.sqrt_nonnegative(256).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 pivot square-root enclosure failed: {error}"
            ))
        })?;
        let trace = a.add(c);
        let lower_eigenvalue = trace.sub(&root).div(&two).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 lower eigenvalue enclosure failed: {error}"
            ))
        })?;
        let upper_eigenvalue = trace.add(&root).div(&two).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 upper eigenvalue enclosure failed: {error}"
            ))
        })?;
        let has_strict_sign =
            |value: &RationalInterval| value.is_strictly_positive() || value.is_strictly_negative();
        if !has_strict_sign(&lower_eigenvalue) || !has_strict_sign(&upper_eigenvalue) {
            return Ok(None);
        }
        let inverse_11 = c.div(&determinant).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 inverse (1,1) failed: {error}"
            ))
        })?;
        let inverse_12 = b.neg().div(&determinant).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 inverse (1,2) failed: {error}"
            ))
        })?;
        let inverse_22 = a.div(&determinant).map_err(|error| {
            CertificateError::VerificationFailed(format!(
                "interval 2x2 inverse (2,2) failed: {error}"
            ))
        })?;
        Ok(Some(CertifiedTwoByTwoPivot {
            lower_eigenvalue,
            upper_eigenvalue,
            inverse_11,
            inverse_12,
            inverse_22,
        }))
    }

    fn symmetric_swap(
        matrix: &mut [RationalInterval],
        dimension: usize,
        left: usize,
        right: usize,
    ) {
        if left == right {
            return;
        }
        for column in 0..dimension {
            matrix.swap(left * dimension + column, right * dimension + column);
        }
        for row in 0..dimension {
            matrix.swap(row * dimension + left, row * dimension + right);
        }
    }

    /// Rigorous interval LDL^T inertia with deterministic symmetric 1x1
    /// pivoting. At each Schur-complement step the first remaining diagonal
    /// enclosure with a strict sign is selected and moved by a congruent row
    /// and column permutation. When no signed diagonal exists, the first 2x2
    /// principal block whose interval determinant and two outward eigenvalue
    /// enclosures exclude zero is used. If neither strategy is certified, the
    /// result is explicitly inconclusive; no numerical threshold is used.
    pub fn interval_symmetric_ldlt_inertia(
        matrix: &[RationalInterval],
        dimension: usize,
    ) -> Result<IntervalInertiaResult, CertificateError> {
        if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
            return Err(CertificateError::Invalid(format!(
                "interval matrix length {} does not match dimension {dimension}",
                matrix.len()
            )));
        }
        for row in 0..dimension {
            for column in 0..row {
                if matrix[row * dimension + column] != matrix[column * dimension + row] {
                    return Err(CertificateError::Invalid(format!(
                        "interval matrix is not symmetric at ({row}, {column})"
                    )));
                }
            }
        }
        let mut schur = matrix.to_vec();
        let mut pivots = Vec::with_capacity(dimension);
        let mut positive = 0usize;
        let mut negative = 0usize;
        let mut pivot_index = 0usize;
        while pivot_index < dimension {
            let selected = (pivot_index..dimension).find(|&candidate| {
                let diagonal = &schur[candidate * dimension + candidate];
                diagonal.is_strictly_positive() || diagonal.is_strictly_negative()
            });
            if let Some(selected) = selected {
                symmetric_swap(&mut schur, dimension, pivot_index, selected);
                let pivot = schur[pivot_index * dimension + pivot_index].clone();
                pivots.push(pivot.clone());
                if pivot.is_strictly_positive() {
                    positive += 1;
                } else if pivot.is_strictly_negative() {
                    negative += 1;
                } else {
                    unreachable!("the selected interval pivot has a strict sign");
                }
                for row in pivot_index + 1..dimension {
                    for column in row..dimension {
                        let numerator = schur[row * dimension + pivot_index]
                            .mul(&schur[column * dimension + pivot_index]);
                        let correction = numerator.div(&pivot).map_err(|error| {
                            CertificateError::VerificationFailed(format!(
                                "interval pivoted LDLT division failed: {error}"
                            ))
                        })?;
                        let updated = schur[row * dimension + column].sub(&correction);
                        schur[row * dimension + column] = updated.clone();
                        schur[column * dimension + row] = updated;
                    }
                }
                pivot_index += 1;
                continue;
            }

            let mut selected_block = None;
            'search: for first in pivot_index..dimension {
                for second in first + 1..dimension {
                    if certify_two_by_two_pivot(&schur, dimension, first, second)?.is_some() {
                        selected_block = Some((first, second));
                        break 'search;
                    }
                }
            }
            let Some((first, second)) = selected_block else {
                pivots.push(schur[pivot_index * dimension + pivot_index].clone());
                return Ok(IntervalInertiaResult::Inconclusive {
                    pivot_index,
                    positive,
                    negative,
                    zero_or_unresolved: dimension - pivot_index,
                    pivot_enclosures: pivots,
                    reason: "no remaining certified 1x1 or 2x2 interval pivot; higher precision or another certified strategy is required"
                        .to_owned(),
                });
            };
            symmetric_swap(&mut schur, dimension, pivot_index, first);
            let adjusted_second = if second == pivot_index { first } else { second };
            symmetric_swap(&mut schur, dimension, pivot_index + 1, adjusted_second);
            let block = certify_two_by_two_pivot(&schur, dimension, pivot_index, pivot_index + 1)?
                .expect("selected 2x2 interval pivot remains valid after symmetric permutation");
            for eigenvalue in [&block.lower_eigenvalue, &block.upper_eigenvalue] {
                pivots.push(eigenvalue.clone());
                if eigenvalue.is_strictly_positive() {
                    positive += 1;
                } else if eigenvalue.is_strictly_negative() {
                    negative += 1;
                } else {
                    unreachable!("certified block eigenvalue has a strict sign");
                }
            }
            for row in pivot_index + 2..dimension {
                let row_first = schur[row * dimension + pivot_index].clone();
                let row_second = schur[row * dimension + pivot_index + 1].clone();
                for column in row..dimension {
                    let column_first = &schur[column * dimension + pivot_index];
                    let column_second = &schur[column * dimension + pivot_index + 1];
                    let correction = row_first
                        .mul(&block.inverse_11)
                        .mul(column_first)
                        .add(&row_first.mul(&block.inverse_12).mul(column_second))
                        .add(&row_second.mul(&block.inverse_12).mul(column_first))
                        .add(&row_second.mul(&block.inverse_22).mul(column_second));
                    let updated = schur[row * dimension + column].sub(&correction);
                    schur[row * dimension + column] = updated.clone();
                    schur[column * dimension + row] = updated;
                }
            }
            pivot_index += 2;
        }
        Ok(IntervalInertiaResult::Conclusive {
            positive,
            negative,
            pivot_enclosures: pivots,
        })
    }

    fn eigenvalue_count_boundary(
        matrix: &[RationalInterval],
        dimension: usize,
        threshold: &Rational,
        boundary_name: &str,
    ) -> Result<EigenvalueCountBoundaryCertificate, (String, String)> {
        let mut shifted = matrix.to_vec();
        let threshold_interval = RationalInterval::point(threshold.clone());
        for diagonal in 0..dimension {
            let index = diagonal * dimension + diagonal;
            if let Some(entry) = shifted.get_mut(index) {
                *entry = entry.sub(&threshold_interval);
            }
        }
        match interval_symmetric_ldlt_inertia(&shifted, dimension) {
            Ok(IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            }) => Ok(EigenvalueCountBoundaryCertificate {
                threshold: rational_record(threshold),
                count_below: negative,
                positive_above: positive,
                pivot_enclosures: pivot_enclosures.iter().map(interval_record).collect(),
            }),
            Ok(IntervalInertiaResult::Inconclusive { reason, .. }) => {
                Err((boundary_name.to_owned(), reason))
            }
            Err(error) => Err((boundary_name.to_owned(), error.to_string())),
        }
    }

    /// Certify the number of eigenvalues strictly below an exact threshold.
    /// A threshold touching an unresolved eigenvalue returns `Inconclusive`.
    pub fn certify_interval_matrix_eigenvalues_below(
        matrix: &[RationalInterval],
        dimension: usize,
        threshold: Rational,
    ) -> IntervalEigenvalueCountResult {
        match eigenvalue_count_boundary(matrix, dimension, &threshold, "upper") {
            Ok(upper_boundary) => IntervalEigenvalueCountResult::Conclusive {
                certificate: IntervalEigenvalueCountCertificate {
                    dimension,
                    eigenvalue_count: upper_boundary.count_below,
                    lower_boundary: None,
                    upper_boundary,
                    interval_semantics: "(-infinity, upper)".to_owned(),
                },
            },
            Err((boundary, reason)) => {
                IntervalEigenvalueCountResult::Inconclusive { boundary, reason }
            }
        }
    }

    /// Certify the number of eigenvalues in the open interval `(lower, upper)`
    /// from two independent shifted-inertia counts.
    pub fn certify_interval_matrix_eigenvalues_in_open_interval(
        matrix: &[RationalInterval],
        dimension: usize,
        lower: Rational,
        upper: Rational,
    ) -> IntervalEigenvalueCountResult {
        if lower >= upper {
            return IntervalEigenvalueCountResult::Inconclusive {
                boundary: "configuration".to_owned(),
                reason: "eigenvalue count interval requires lower < upper".to_owned(),
            };
        }
        let lower_boundary = match eigenvalue_count_boundary(matrix, dimension, &lower, "lower") {
            Ok(boundary) => boundary,
            Err((boundary, reason)) => {
                return IntervalEigenvalueCountResult::Inconclusive { boundary, reason };
            }
        };
        let upper_boundary = match eigenvalue_count_boundary(matrix, dimension, &upper, "upper") {
            Ok(boundary) => boundary,
            Err((boundary, reason)) => {
                return IntervalEigenvalueCountResult::Inconclusive { boundary, reason };
            }
        };
        let Some(eigenvalue_count) = upper_boundary
            .count_below
            .checked_sub(lower_boundary.count_below)
        else {
            return IntervalEigenvalueCountResult::Inconclusive {
                boundary: "reconciliation".to_owned(),
                reason: "shifted inertia count decreased across ordered thresholds".to_owned(),
            };
        };
        IntervalEigenvalueCountResult::Conclusive {
            certificate: IntervalEigenvalueCountCertificate {
                dimension,
                lower_boundary: Some(lower_boundary),
                upper_boundary,
                eigenvalue_count,
                interval_semantics: "(lower, upper)".to_owned(),
            },
        }
    }

    fn exact_interval_matrix_digest(
        matrix: &[RationalInterval],
        dimension: usize,
    ) -> Result<ContentDigest, CertificateError> {
        let records = matrix.iter().map(interval_record).collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&(dimension, records)).map_err(|error| {
            CertificateError::Invalid(format!(
                "failed to serialize selected-eigenvalue matrix: {error}"
            ))
        })?;
        Ok(ContentDigest(sha256_hex(&bytes)))
    }

    fn validate_selected_eigenvalue_enclosure(
        certificate: &ExactSelectedEigenvalueEnclosure,
    ) -> Result<(Rational, Rational), CertificateError> {
        if certificate.dimension == 0
            || certificate.requested_index >= certificate.dimension
            || certificate.first_enclosed_index > certificate.requested_index
            || certificate.last_enclosed_index < certificate.requested_index
            || certificate.last_enclosed_index >= certificate.dimension
        {
            return Err(CertificateError::Invalid(
                "selected-eigenvalue certificate has inconsistent dimensions or indices".to_owned(),
            ));
        }
        if !certificate.matrix_digest.validate() {
            return Err(CertificateError::Invalid(
                "selected-eigenvalue certificate has an invalid matrix digest".to_owned(),
            ));
        }
        if certificate.lower_boundary.count_below != certificate.first_enclosed_index
            || certificate.upper_boundary.count_below
                != certificate.last_enclosed_index.saturating_add(1)
            || certificate
                .lower_boundary
                .count_below
                .checked_add(certificate.lower_boundary.positive_above)
                != Some(certificate.dimension)
            || certificate
                .upper_boundary
                .count_below
                .checked_add(certificate.upper_boundary.positive_above)
                != Some(certificate.dimension)
            || certificate.lower_boundary.pivot_enclosures.len() != certificate.dimension
            || certificate.upper_boundary.pivot_enclosures.len() != certificate.dimension
            || certificate.simple
                != (certificate.first_enclosed_index == certificate.last_enclosed_index)
        {
            return Err(CertificateError::VerificationFailed(
                "selected-eigenvalue index, count, pivot, or multiplicity evidence is inconsistent"
                    .to_owned(),
            ));
        }
        let lower = parse(&certificate.lower)?;
        let upper = parse(&certificate.upper)?;
        if lower >= upper
            || parse(&certificate.lower_boundary.threshold)? != lower
            || parse(&certificate.upper_boundary.threshold)? != upper
        {
            return Err(CertificateError::VerificationFailed(
                "selected-eigenvalue endpoints and boundary thresholds are inconsistent".to_owned(),
            ));
        }
        for pivot in certificate
            .lower_boundary
            .pivot_enclosures
            .iter()
            .chain(&certificate.upper_boundary.pivot_enclosures)
        {
            parse_interval(pivot)?;
        }
        Ok((lower, upper))
    }

    /// Independently replay both shifted-inertia boundaries against a supplied
    /// exact-endpoint interval matrix.
    pub fn verify_selected_interval_eigenvalue_enclosure(
        certificate: &ExactSelectedEigenvalueEnclosure,
        matrix: &[RationalInterval],
    ) -> VerificationReport {
        let verify = || -> Result<(), CertificateError> {
            let (lower, upper) = validate_selected_eigenvalue_enclosure(certificate)?;
            if matrix.len() != certificate.dimension.saturating_mul(certificate.dimension)
                || exact_interval_matrix_digest(matrix, certificate.dimension)?
                    != certificate.matrix_digest
            {
                return Err(CertificateError::VerificationFailed(
                    "selected-eigenvalue matrix does not match its certified digest".to_owned(),
                ));
            }
            let replay_boundary = |threshold: &Rational, name: &str| {
                eigenvalue_count_boundary(matrix, certificate.dimension, threshold, name).map_err(
                    |(boundary, reason)| {
                        CertificateError::VerificationFailed(format!(
                            "{boundary} shifted-inertia replay was inconclusive: {reason}"
                        ))
                    },
                )
            };
            if replay_boundary(&lower, "lower")? != certificate.lower_boundary
                || replay_boundary(&upper, "upper")? != certificate.upper_boundary
            {
                return Err(CertificateError::VerificationFailed(
                    "selected-eigenvalue shifted-inertia replay differs from recorded boundaries"
                        .to_owned(),
                ));
            }
            Ok(())
        };
        match verify() {
            Ok(()) => VerificationReport {
                valid: true,
                checks: vec![
                    "selected index, enclosure, and multiplicity structure verified".to_owned(),
                    "exact interval-matrix digest verified".to_owned(),
                    "both shifted-inertia boundaries independently replayed".to_owned(),
                ],
                warnings: Vec::new(),
                errors: Vec::new(),
            },
            Err(error) => VerificationReport {
                valid: false,
                checks: Vec::new(),
                warnings: Vec::new(),
                errors: vec![error.to_string()],
            },
        }
    }

    pub fn build_portable_selected_eigenvalue_certificate(
        matrix: &[RationalInterval],
        enclosure: &ExactSelectedEigenvalueEnclosure,
    ) -> Result<PortableSelectedEigenvalueCertificate, CertificateError> {
        let portable = PortableSelectedEigenvalueCertificate {
            schema_version: 1,
            matrix_row_major: matrix.iter().map(interval_record).collect(),
            enclosure: enclosure.clone(),
        };
        let report = verify_portable_selected_eigenvalue_certificate(&portable);
        if report.valid {
            Ok(portable)
        } else {
            Err(CertificateError::VerificationFailed(
                report.errors.join("; "),
            ))
        }
    }

    /// Replay a self-contained exact enclosure without running discovery or
    /// trusting any displayed decimal approximation.
    pub fn verify_portable_selected_eigenvalue_certificate(
        portable: &PortableSelectedEigenvalueCertificate,
    ) -> VerificationReport {
        if portable.schema_version != 1 {
            return VerificationReport {
                valid: false,
                checks: Vec::new(),
                warnings: Vec::new(),
                errors: vec!["unsupported portable selected-eigenvalue schema".to_owned()],
            };
        }
        let matrix = match portable
            .matrix_row_major
            .iter()
            .map(parse_interval)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(matrix) => matrix,
            Err(error) => {
                return VerificationReport {
                    valid: false,
                    checks: Vec::new(),
                    warnings: Vec::new(),
                    errors: vec![error.to_string()],
                };
            }
        };
        verify_selected_interval_eigenvalue_enclosure(&portable.enclosure, &matrix)
    }

    /// Enclose the zero-based algebraically ordered eigenvalue selected by
    /// `requested_index`. Every bisection decision is an exact-rational
    /// shifted-inertia proof. A boundary whose interval LDL^T pivot contains
    /// zero is reported as inconclusive instead of being perturbed.
    pub fn certify_selected_interval_eigenvalue(
        matrix: &[RationalInterval],
        dimension: usize,
        requested_index: usize,
        mut lower: Rational,
        mut upper: Rational,
        target_width: Rational,
        maximum_bisection_steps: usize,
    ) -> SelectedEigenvalueEnclosureResult {
        if dimension == 0
            || matrix.len() != dimension.saturating_mul(dimension)
            || requested_index >= dimension
            || lower >= upper
            || target_width <= 0
            || maximum_bisection_steps == 0
        {
            return SelectedEigenvalueEnclosureResult::Inconclusive {
                boundary: "configuration".to_owned(),
                reason: "selected-eigenvalue certification requires a square nonempty matrix, a valid zero-based index, lower < upper, positive target width, and a positive step limit".to_owned(),
            };
        }

        let mut lower_boundary = match eigenvalue_count_boundary(matrix, dimension, &lower, "lower")
        {
            Ok(boundary) => boundary,
            Err((boundary, reason)) => {
                return SelectedEigenvalueEnclosureResult::Inconclusive { boundary, reason };
            }
        };
        let mut upper_boundary = match eigenvalue_count_boundary(matrix, dimension, &upper, "upper")
        {
            Ok(boundary) => boundary,
            Err((boundary, reason)) => {
                return SelectedEigenvalueEnclosureResult::Inconclusive { boundary, reason };
            }
        };
        if lower_boundary.count_below > requested_index
            || upper_boundary.count_below <= requested_index
        {
            return SelectedEigenvalueEnclosureResult::Inconclusive {
                boundary: "bracket".to_owned(),
                reason: format!(
                    "initial shifted-inertia counts [{}, {}] do not enclose eigenvalue index {requested_index}",
                    lower_boundary.count_below, upper_boundary.count_below
                ),
            };
        }

        let mut bisection_steps = 0usize;
        while upper.clone() - lower.clone() > target_width {
            if bisection_steps == maximum_bisection_steps {
                return SelectedEigenvalueEnclosureResult::Inconclusive {
                    boundary: "bisection".to_owned(),
                    reason: format!(
                        "target width was not reached within {maximum_bisection_steps} steps"
                    ),
                };
            }
            let midpoint = (lower.clone() + upper.clone()) / 2;
            let midpoint_boundary =
                match eigenvalue_count_boundary(matrix, dimension, &midpoint, "midpoint") {
                    Ok(boundary) => boundary,
                    Err((boundary, reason)) => {
                        return SelectedEigenvalueEnclosureResult::Inconclusive {
                            boundary,
                            reason,
                        };
                    }
                };
            if midpoint_boundary.count_below <= requested_index {
                lower = midpoint;
                lower_boundary = midpoint_boundary;
            } else {
                upper = midpoint;
                upper_boundary = midpoint_boundary;
            }
            bisection_steps += 1;
        }

        let Some(enclosed_count) = upper_boundary
            .count_below
            .checked_sub(lower_boundary.count_below)
        else {
            return SelectedEigenvalueEnclosureResult::Inconclusive {
                boundary: "reconciliation".to_owned(),
                reason: "shifted-inertia count decreased across the final enclosure".to_owned(),
            };
        };
        if enclosed_count == 0 {
            return SelectedEigenvalueEnclosureResult::Inconclusive {
                boundary: "reconciliation".to_owned(),
                reason: "final enclosure contains no certified eigenvalue".to_owned(),
            };
        }
        let matrix_digest = match exact_interval_matrix_digest(matrix, dimension) {
            Ok(digest) => digest,
            Err(error) => {
                return SelectedEigenvalueEnclosureResult::Inconclusive {
                    boundary: "serialization".to_owned(),
                    reason: error.to_string(),
                };
            }
        };
        SelectedEigenvalueEnclosureResult::Conclusive {
            certificate: Box::new(ExactSelectedEigenvalueEnclosure {
                dimension,
                matrix_digest,
                requested_index,
                first_enclosed_index: lower_boundary.count_below,
                last_enclosed_index: upper_boundary.count_below - 1,
                lower: rational_record(&lower),
                upper: rational_record(&upper),
                lower_boundary,
                upper_boundary,
                bisection_steps,
                simple: enclosed_count == 1,
            }),
        }
    }

    /// Build an exact lower bound for the gap between adjacent certified
    /// clusters of the same interval matrix.
    pub fn certify_exact_spectral_gap(
        lower_cluster: &ExactSelectedEigenvalueEnclosure,
        upper_cluster: &ExactSelectedEigenvalueEnclosure,
    ) -> Result<ExactSpectralGapCertificate, CertificateError> {
        let (_, lower_upper) = validate_selected_eigenvalue_enclosure(lower_cluster)?;
        let (upper_lower, _) = validate_selected_eigenvalue_enclosure(upper_cluster)?;
        if lower_cluster.dimension != upper_cluster.dimension
            || lower_cluster.matrix_digest != upper_cluster.matrix_digest
        {
            return Err(CertificateError::Invalid(
                "spectral-gap clusters refer to different matrices".to_owned(),
            ));
        }
        if lower_cluster.last_enclosed_index.checked_add(1)
            != Some(upper_cluster.first_enclosed_index)
        {
            return Err(CertificateError::Invalid(
                "spectral-gap clusters are not adjacent in algebraic order".to_owned(),
            ));
        }
        let gap = upper_lower - lower_upper;
        if gap <= 0 {
            return Err(CertificateError::VerificationFailed(
                "certified cluster enclosures do not have positive separation".to_owned(),
            ));
        }
        Ok(ExactSpectralGapCertificate {
            lower_cluster: lower_cluster.clone(),
            upper_cluster: upper_cluster.clone(),
            certified_lower_bound: rational_record(&gap),
        })
    }

    /// Verify an exact gap record from its two already replayable cluster
    /// certificates. Matrix-boundary replay remains available through
    /// `verify_selected_interval_eigenvalue_enclosure`.
    pub fn verify_exact_spectral_gap_certificate(
        certificate: &ExactSpectralGapCertificate,
    ) -> VerificationReport {
        match certify_exact_spectral_gap(&certificate.lower_cluster, &certificate.upper_cluster) {
            Ok(recomputed) if recomputed == *certificate => VerificationReport {
                valid: true,
                checks: vec![
                    "cluster structure, adjacency, and common matrix identity verified".to_owned(),
                    "strictly positive exact gap lower bound recomputed".to_owned(),
                ],
                warnings: Vec::new(),
                errors: Vec::new(),
            },
            Ok(_) => VerificationReport {
                valid: false,
                checks: Vec::new(),
                warnings: Vec::new(),
                errors: vec!["spectral-gap lower bound differs from exact recomputation".to_owned()],
            },
            Err(error) => VerificationReport {
                valid: false,
                checks: Vec::new(),
                warnings: Vec::new(),
                errors: vec![error.to_string()],
            },
        }
    }

    pub fn combined_certification_error_bound(
        budget: &CertificationErrorBudget,
    ) -> Result<Rational, CertificateError> {
        budget.validate_declared_coverage()?;
        let mut total = Rational::from((0, 1));
        for component in &budget.components {
            let bound = parse(&component.absolute_bound)?;
            if bound < 0 {
                return Err(CertificateError::Invalid(format!(
                    "error bound for {:?} is negative",
                    component.source
                )));
            }
            if component.affects_claim {
                total += bound;
            }
        }
        Ok(total)
    }

    /// Exact unpivoted symmetric LDL^T inertia for rational matrices.
    ///
    /// The routine is rigorous when every pivot is nonzero. A zero pivot is
    /// reported as unsupported rather than perturbed, because a certified
    /// certificate may not manufacture a sign. Pivoted exact/ball LDL^T is a
    /// separate backend milestone.
    pub fn exact_symmetric_ldlt_inertia(
        matrix: &[Rational],
        dimension: usize,
    ) -> Result<ExactInertia, CertificateError> {
        if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
            return Err(CertificateError::Invalid(format!(
                "exact matrix length {} does not match dimension {dimension}",
                matrix.len()
            )));
        }
        for row in 0..dimension {
            for column in 0..row {
                if matrix[row * dimension + column] != matrix[column * dimension + row] {
                    return Err(CertificateError::Invalid(format!(
                        "exact matrix is not symmetric at ({row}, {column})"
                    )));
                }
            }
        }

        let mut lower = vec![Rational::from((0, 1)); dimension * dimension];
        let mut diagonal = vec![Rational::from((0, 1)); dimension];
        let mut positive = 0usize;
        let mut negative = 0usize;
        for pivot_index in 0..dimension {
            lower[pivot_index * dimension + pivot_index] = Rational::from((1, 1));
            let mut pivot = matrix[pivot_index * dimension + pivot_index].clone();
            for prior in 0..pivot_index {
                let mut term = lower[pivot_index * dimension + prior].clone();
                term *= &lower[pivot_index * dimension + prior];
                term *= &diagonal[prior];
                pivot -= term;
            }
            if pivot == 0 {
                return Err(CertificateError::Unsupported(format!(
                    "exact LDL^T encountered a zero pivot at index {pivot_index}; use a pivoted certified backend"
                )));
            }
            if pivot > 0 {
                positive += 1;
            } else {
                negative += 1;
            }
            diagonal[pivot_index] = pivot;

            for row in (pivot_index + 1)..dimension {
                let mut numerator = matrix[row * dimension + pivot_index].clone();
                for prior in 0..pivot_index {
                    let mut term = lower[row * dimension + prior].clone();
                    term *= &lower[pivot_index * dimension + prior];
                    term *= &diagonal[prior];
                    numerator -= term;
                }
                numerator /= &diagonal[pivot_index];
                lower[row * dimension + pivot_index] = numerator;
            }
        }
        Ok(ExactInertia {
            positive,
            negative,
            zero_or_unresolved: 0,
        })
    }

    /// Verify `numerator / denominator >= claimed_lower` exactly.
    pub fn verify_rayleigh_lower_bound(
        numerator: &ExactRationalRecord,
        denominator: &ExactRationalRecord,
        claimed_lower: &ExactRationalRecord,
    ) -> Result<(), CertificateError> {
        let numerator = parse(numerator)?;
        let denominator = parse(denominator)?;
        let claimed = parse(claimed_lower)?;
        if denominator <= 0 {
            return Err(CertificateError::Invalid(
                "Rayleigh denominator must be positive".to_owned(),
            ));
        }
        let quotient = numerator / denominator;
        if quotient < claimed {
            return Err(CertificateError::VerificationFailed(
                "exact Rayleigh quotient is below the claimed lower bound".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certification_reuse_plan_links_reports_without_embedding_them() {
        let plan = certification_artifact_reuse_plan();
        plan.validate().unwrap();
        let bundle = plan
            .artifacts
            .iter()
            .find(|node| node.kind == "certificate_bundle")
            .unwrap();
        assert!(bundle
            .dependencies
            .contains(&"verification_report".to_owned()));
    }

    fn digest() -> ContentDigest {
        ContentDigest("00".repeat(32))
    }

    #[test]
    fn positive_inertia_is_conclusive() {
        let inertia = InertiaCertificate {
            dimension: 401,
            positive: 401,
            negative: 0,
            zero_or_unresolved: 0,
            matrix_digest: digest(),
            scalar_backend: "arb".to_owned(),
            precision_bits: 9000,
            pivot_enclosures_digest: Some(digest()),
        };
        inertia.validate().unwrap();
        assert!(inertia.certifies_positive_definite());
    }

    #[test]
    fn certificate_identifier_covers_bundle_contents() {
        let mut bundle = CertificateBundle {
            schema_version: 1,
            certificate_id: digest(),
            claim: CertificateClaim::MatrixInertia,
            assurance: AssuranceLevel::Certified,
            toolkit_version: ToolkitVersion::parse("0.13.0").unwrap(),
            citation: None,
            provenance: SolverProvenance::current_package("test"),
            inputs: Vec::new(),
            inertia: Some(InertiaCertificate {
                dimension: 2,
                positive: 2,
                negative: 0,
                zero_or_unresolved: 0,
                matrix_digest: digest(),
                scalar_backend: "exact".to_owned(),
                precision_bits: 256,
                pivot_enclosures_digest: None,
            }),
            eigenvalue_enclosures: Vec::new(),
            spectral_gap: None,
            exact_records: BTreeMap::new(),
            evidence_digests: BTreeMap::new(),
            approximation_ledger: ApproximationLedger::default(),
            assumptions: Vec::new(),
            notes: Vec::new(),
        };
        bundle.refresh_certificate_id().unwrap();
        assert!(bundle.validate_structure().is_ok());

        let uncited_id = bundle.certificate_id.clone();
        bundle
            .attach_citation(ArtifactCitationMetadata {
                schema_version: 1,
                artifact_title: "Finite matrix inertia certificate".to_owned(),
                artifact_type: "certificate_bundle".to_owned(),
                authors: vec![xc_core::ArchiveAuthor {
                    given_names: "Ronnie".to_owned(),
                    family_names: "Andrews".to_owned(),
                    name_suffix: Some("Jr.".to_owned()),
                    orcid: "https://orcid.org/0009-0003-9724-3104".to_owned(),
                }],
                software_title: "Xcelerator Toolkit".to_owned(),
                software_version: "0.13.0".to_owned(),
                repository: "https://github.com/TeamXcelerator/xcelerator-toolkit".to_owned(),
                preferred_citation: "Andrews, finite matrix inertia certificate (2026).".to_owned(),
                software_doi: None,
                artifact_doi: Some("10.5281/zenodo.1234567".to_owned()),
            })
            .unwrap();
        assert_ne!(bundle.certificate_id, uncited_id);
        bundle.validate_structure().unwrap();
        let encoded = serde_json::to_vec(&bundle).unwrap();
        let decoded: CertificateBundle = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.citation, bundle.citation);

        let mut citation_tamper = bundle.clone();
        citation_tamper
            .citation
            .as_mut()
            .unwrap()
            .preferred_citation
            .push_str(" changed");
        assert!(citation_tamper.validate_structure().is_err());

        let mut wrong_version = bundle.clone();
        wrong_version.citation.as_mut().unwrap().software_version = "0.14.0".to_owned();
        wrong_version.refresh_certificate_id().unwrap();
        assert!(wrong_version.validate_structure().is_err());

        let mut credential_bearing = bundle.clone();
        credential_bearing
            .notes
            .push("github_pat_abcdefghijklmnopqrstuvwxyz123456".to_owned()); // SECRET_AUDIT_PATTERN
        credential_bearing.refresh_certificate_id().unwrap();
        let error = credential_bearing
            .validate_structure()
            .unwrap_err()
            .to_string();
        assert!(error.contains("credential-shaped material"));
        assert!(!error.contains("github_pat_"));

        let mut unbounded = bundle.clone();
        unbounded
            .approximation_ledger
            .entries
            .push(xc_core::ApproximationEvidence {
                kind: xc_core::ApproximationKind::Compression,
                purpose: "compressed decisive matrix".to_owned(),
                decisive_for_accepted_result: true,
                rigorous_bound: None,
            });
        unbounded.refresh_certificate_id().unwrap();
        assert!(unbounded
            .validate_structure()
            .unwrap_err()
            .to_string()
            .contains("rigorous bound"));

        bundle.notes.push("changed".to_owned());
        assert!(bundle.validate_structure().is_err());
    }

    #[test]
    fn overlapping_gap_enclosures_are_rejected() {
        let enclosure = |index, lower: &str, upper: &str| EigenvalueEnclosure {
            first_index: index,
            last_index: index,
            interval: DecimalInterval {
                lower: lower.to_owned(),
                upper: upper.to_owned(),
            },
            multiplicity_lower: 1,
            multiplicity_upper: 1,
        };
        let gap = SpectralGapCertificate {
            lower_cluster: enclosure(0, "1", "3"),
            upper_cluster: enclosure(1, "2", "4"),
            certified_lower_bound: "0.1".to_owned(),
        };
        assert!(gap.validate().is_err());
    }
}

#[cfg(all(test, feature = "hp"))]
mod exact_inertia_tests {
    use super::exact::{
        build_portable_interval_inertia_certificate,
        build_portable_selected_eigenvalue_certificate, certify_exact_spectral_gap,
        certify_interval_matrix_eigenvalues_below,
        certify_interval_matrix_eigenvalues_in_open_interval, certify_selected_interval_eigenvalue,
        combined_certification_error_bound, exact_symmetric_ldlt_inertia,
        interval_symmetric_ldlt_inertia, verify_exact_spectral_gap_certificate,
        verify_portable_interval_inertia_certificate,
        verify_portable_selected_eigenvalue_certificate,
        verify_selected_interval_eigenvalue_enclosure, IntervalInertiaResult,
    };
    use super::{
        CertificationErrorBudget, CertificationErrorComponent, CertificationErrorSource,
        ExactRationalRecord, IntervalEigenvalueCountResult, SelectedEigenvalueEnclosureResult,
    };
    use rug::Rational;
    use xc_cache::ContentDigest;
    use xc_numerics::interval::RationalInterval;

    #[test]
    fn exact_ldlt_certifies_positive_and_indefinite_examples() {
        let positive = vec![
            Rational::from((2, 1)),
            Rational::from((1, 1)),
            Rational::from((1, 1)),
            Rational::from((3, 1)),
        ];
        let inertia = exact_symmetric_ldlt_inertia(&positive, 2).unwrap();
        assert_eq!(inertia.positive, 2);
        assert_eq!(inertia.negative, 0);

        let indefinite = vec![
            Rational::from((1, 1)),
            Rational::from((0, 1)),
            Rational::from((0, 1)),
            Rational::from((-1, 1)),
        ];
        let inertia = exact_symmetric_ldlt_inertia(&indefinite, 2).unwrap();
        assert_eq!(inertia.positive, 1);
        assert_eq!(inertia.negative, 1);
    }

    #[test]
    fn interval_ldlt_certifies_strict_pivots_and_reports_ambiguous_pivot() {
        let interval = |lower, upper| {
            RationalInterval::new(Rational::from(lower), Rational::from(upper)).unwrap()
        };
        let zero = RationalInterval::point(Rational::from((0, 1)));
        let positive = vec![
            interval((19, 10), (21, 10)),
            zero.clone(),
            zero,
            interval((29, 10), (31, 10)),
        ];
        let result = interval_symmetric_ldlt_inertia(&positive, 2).unwrap();
        match result {
            IntervalInertiaResult::Conclusive {
                positive, negative, ..
            } => {
                assert_eq!(positive, 2);
                assert_eq!(negative, 0);
            }
            other => panic!("strictly positive diagonal fixture was not certified: {other:?}"),
        }

        let ambiguous = vec![interval((-1, 1), (1, 1))];
        let result = interval_symmetric_ldlt_inertia(&ambiguous, 1).unwrap();
        assert!(matches!(
            result,
            IntervalInertiaResult::Inconclusive { pivot_index: 0, .. }
        ));
    }

    #[test]
    fn interval_ldlt_certifies_symmetric_1x1_and_2x2_pivots() {
        let point = |numerator, denominator| {
            RationalInterval::point(Rational::from((numerator, denominator)))
        };
        // The natural leading pivot is zero. A congruent row/column swap
        // selects 2, after which the remaining Schur pivot is -1/2.
        let pivoted = vec![point(0, 1), point(1, 1), point(1, 1), point(2, 1)];
        let result = interval_symmetric_ldlt_inertia(&pivoted, 2).unwrap();
        match result {
            IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            } => {
                assert_eq!((positive, negative), (1, 1));
                assert_eq!(pivot_enclosures, vec![point(2, 1), point(-1, 2)]);
            }
            other => panic!("symmetric 1x1 pivoting did not resolve the fixture: {other:?}"),
        }

        // This nonsingular matrix has no signed diagonal candidate. The
        // certified 2x2 eigenvalue enclosures are exactly -1 and 1.
        let block_required = vec![point(0, 1), point(1, 1), point(1, 1), point(0, 1)];
        let result = interval_symmetric_ldlt_inertia(&block_required, 2).unwrap();
        match result {
            IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            } => {
                assert_eq!((positive, negative), (1, 1));
                assert_eq!(pivot_enclosures, vec![point(-1, 1), point(1, 1)]);
            }
            other => panic!("a certified 2x2 pivot did not resolve the fixture: {other:?}"),
        }

        let certificate = build_portable_interval_inertia_certificate(
            &block_required,
            2,
            256,
            "exact-rational-test",
            ContentDigest("33".repeat(32)),
            std::collections::BTreeMap::from([(
                "fixture".to_owned(),
                "indefinite-block-pivot".to_owned(),
            )]),
            vec!["2x2 pivot eigenvalue evidence".to_owned()],
        )
        .unwrap();
        assert!(verify_portable_interval_inertia_certificate(&certificate).valid);

        // A leading block must also update a nonempty trailing Schur
        // complement. The triangle adjacency matrix has spectrum (2,-1,-1).
        let triangle = vec![
            point(0, 1),
            point(1, 1),
            point(1, 1),
            point(1, 1),
            point(0, 1),
            point(1, 1),
            point(1, 1),
            point(1, 1),
            point(0, 1),
        ];
        match interval_symmetric_ldlt_inertia(&triangle, 3).unwrap() {
            IntervalInertiaResult::Conclusive {
                positive,
                negative,
                pivot_enclosures,
            } => {
                assert_eq!((positive, negative), (1, 2));
                assert_eq!(
                    pivot_enclosures,
                    vec![point(-1, 1), point(1, 1), point(-2, 1)]
                );
            }
            other => panic!("2x2 block Schur update did not resolve the fixture: {other:?}"),
        }

        let ambiguous =
            RationalInterval::new(Rational::from((-1, 1)), Rational::from((1, 1))).unwrap();
        let unresolved = vec![
            ambiguous.clone(),
            ambiguous.clone(),
            ambiguous.clone(),
            ambiguous,
        ];
        assert!(matches!(
            interval_symmetric_ldlt_inertia(&unresolved, 2).unwrap(),
            IntervalInertiaResult::Inconclusive { pivot_index: 0, .. }
        ));
    }

    #[test]
    fn portable_interval_inertia_round_trips_and_replays_independently() {
        let interval = |lower, upper| {
            RationalInterval::new(Rational::from(lower), Rational::from(upper)).unwrap()
        };
        let zero = RationalInterval::point(Rational::from((0, 1)));
        let matrix = vec![
            interval((19, 10), (21, 10)),
            zero.clone(),
            zero,
            interval((29, 10), (31, 10)),
        ];
        let certificate = build_portable_interval_inertia_certificate(
            &matrix,
            2,
            256,
            "exact-rational-test",
            ContentDigest("22".repeat(32)),
            std::collections::BTreeMap::from([("fixture".to_owned(), "positive-2x2".to_owned())]),
            vec!["matrix endpoints include assembly uncertainty".to_owned()],
        )
        .unwrap();
        let encoded = serde_json::to_vec(&certificate).unwrap();
        let decoded = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(certificate, decoded);
        let report = verify_portable_interval_inertia_certificate(&decoded);
        assert!(report.valid, "{:?}", report.errors);

        let mut forged_pivot = decoded.clone();
        forged_pivot.pivot_enclosures[0].lower.numerator = "18".to_owned();
        forged_pivot.refresh_certificate_id().unwrap();
        let report = verify_portable_interval_inertia_certificate(&forged_pivot);
        assert!(!report.valid);
        assert!(report.errors.iter().any(|error| error.contains("replay")));

        let mut corrupt_matrix = decoded;
        corrupt_matrix.matrix_row_major[0].lower.numerator = "18".to_owned();
        let report = verify_portable_interval_inertia_certificate(&corrupt_matrix);
        assert!(!report.valid);
        assert!(report.errors.iter().any(|error| error.contains("digest")));
    }

    #[test]
    fn shifted_interval_inertia_counts_below_and_between_thresholds() {
        let diagonal = |lower, upper| {
            RationalInterval::new(Rational::from(lower), Rational::from(upper)).unwrap()
        };
        let zero = RationalInterval::point(Rational::from((0, 1)));
        let matrix = vec![
            diagonal((9, 10), (11, 10)),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            diagonal((29, 10), (31, 10)),
            zero.clone(),
            zero.clone(),
            zero,
            diagonal((49, 10), (51, 10)),
        ];
        let below = certify_interval_matrix_eigenvalues_below(&matrix, 3, Rational::from((2, 1)));
        let IntervalEigenvalueCountResult::Conclusive { certificate } = below else {
            panic!("count below a separating threshold was inconclusive: {below:?}");
        };
        assert_eq!(certificate.eigenvalue_count, 1);
        assert!(certificate.lower_boundary.is_none());

        let between = certify_interval_matrix_eigenvalues_in_open_interval(
            &matrix,
            3,
            Rational::from((2, 1)),
            Rational::from((4, 1)),
        );
        let IntervalEigenvalueCountResult::Conclusive { certificate } = between else {
            panic!("count between separating thresholds was inconclusive: {between:?}");
        };
        assert_eq!(certificate.eigenvalue_count, 1);
        assert_eq!(certificate.lower_boundary.as_ref().unwrap().count_below, 1);
        assert_eq!(certificate.upper_boundary.count_below, 2);

        let touching =
            certify_interval_matrix_eigenvalues_below(&matrix, 3, Rational::from((3, 1)));
        assert!(matches!(
            touching,
            IntervalEigenvalueCountResult::Inconclusive { .. }
        ));
    }

    #[test]
    fn selected_eigenvalues_and_adjacent_gap_are_certified_exactly() {
        let point = |numerator, denominator| {
            RationalInterval::point(Rational::from((numerator, denominator)))
        };
        let zero = point(0, 1);
        let matrix = vec![
            point(6, 5),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            point(16, 5),
            zero.clone(),
            zero.clone(),
            zero,
            point(26, 5),
        ];
        let selected = |index, lower, upper| {
            certify_selected_interval_eigenvalue(
                &matrix,
                3,
                index,
                Rational::from((lower, 1)),
                Rational::from((upper, 1)),
                Rational::from((1, 1000)),
                32,
            )
        };
        let SelectedEigenvalueEnclosureResult::Conclusive { certificate: first } =
            selected(0, 0, 2)
        else {
            panic!("first selected eigenvalue was not certified");
        };
        let SelectedEigenvalueEnclosureResult::Conclusive {
            certificate: second,
        } = selected(1, 2, 4)
        else {
            panic!("second selected eigenvalue was not certified");
        };
        assert!(first.simple);
        assert!(second.simple);
        assert_eq!(
            (first.first_enclosed_index, first.last_enclosed_index),
            (0, 0)
        );
        assert_eq!(
            (second.first_enclosed_index, second.last_enclosed_index),
            (1, 1)
        );
        let report = verify_selected_interval_eigenvalue_enclosure(&first, &matrix);
        assert!(report.valid, "{:?}", report.errors);

        let mut gap = certify_exact_spectral_gap(&first, &second).unwrap();
        assert!(super::exact::parse(&gap.certified_lower_bound).unwrap() > 0);
        assert!(verify_exact_spectral_gap_certificate(&gap).valid);
        gap.certified_lower_bound.numerator = "1".to_owned();
        gap.certified_lower_bound.denominator = "1000000".to_owned();
        assert!(!verify_exact_spectral_gap_certificate(&gap).valid);

        let mut wrong_matrix = second;
        wrong_matrix.matrix_digest = ContentDigest("33".repeat(32));
        assert!(certify_exact_spectral_gap(&first, &wrong_matrix).is_err());
    }

    #[test]
    fn portable_selected_eigenvalue_replays_offline_and_rejects_matrix_tampering() {
        let point = |numerator, denominator| {
            RationalInterval::point(Rational::from((numerator, denominator)))
        };
        let zero = point(0, 1);
        let matrix = vec![point(6, 5), zero.clone(), zero, point(16, 5)];
        let SelectedEigenvalueEnclosureResult::Conclusive { certificate } =
            certify_selected_interval_eigenvalue(
                &matrix,
                2,
                0,
                Rational::from((0, 1)),
                Rational::from((2, 1)),
                Rational::from((1, 1000)),
                32,
            )
        else {
            panic!("ground-state enclosure should be conclusive");
        };
        let portable =
            build_portable_selected_eigenvalue_certificate(&matrix, &certificate).unwrap();
        let encoded = serde_json::to_string(&portable).unwrap();
        let decoded: super::PortableSelectedEigenvalueCertificate =
            serde_json::from_str(&encoded).unwrap();
        assert!(verify_portable_selected_eigenvalue_certificate(&decoded).valid);

        let mut tampered = decoded;
        tampered.matrix_row_major[0].lower.numerator = "2".to_owned();
        assert!(!verify_portable_selected_eigenvalue_certificate(&tampered).valid);
    }

    #[test]
    fn selected_eigenvalue_boundary_collision_is_inconclusive() {
        let point = |value| RationalInterval::point(Rational::from((value, 1)));
        let zero = point(0);
        let matrix = vec![
            point(1),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            point(3),
            zero.clone(),
            zero.clone(),
            zero,
            point(5),
        ];
        let result = certify_selected_interval_eigenvalue(
            &matrix,
            3,
            1,
            Rational::from((2, 1)),
            Rational::from((4, 1)),
            Rational::from((1, 100)),
            32,
        );
        assert!(matches!(
            result,
            SelectedEigenvalueEnclosureResult::Inconclusive { ref boundary, .. }
                if boundary == "midpoint"
        ));
    }

    #[test]
    fn certification_error_budget_requires_every_claim_affecting_source() {
        let component = |source, numerator: &str| CertificationErrorComponent {
            source,
            absolute_bound: ExactRationalRecord {
                numerator: numerator.to_owned(),
                denominator: "1000".to_owned(),
            },
            affects_claim: true,
            evidence_digest: ContentDigest("11".repeat(32)),
            method: "exact reference enclosure".to_owned(),
        };
        let mut budget = CertificationErrorBudget {
            required_sources: vec![
                CertificationErrorSource::MatrixAssembly,
                CertificationErrorSource::Transformation,
            ],
            components: vec![component(CertificationErrorSource::MatrixAssembly, "2")],
        };
        assert!(budget.validate_declared_coverage().is_err());
        budget
            .components
            .push(component(CertificationErrorSource::Transformation, "3"));
        assert_eq!(
            combined_certification_error_bound(&budget).unwrap(),
            Rational::from((1, 200))
        );
    }
}
