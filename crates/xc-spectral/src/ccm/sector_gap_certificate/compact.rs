//! Small analytic replay certificates: no retained matrices or pivot traces.
//!
//! Verification always reconstructs analytic assembly and independently proves
//! all recorded inertia counts. This is a distinct format, not legacy export.

use super::*;

#[cfg(feature = "arb")]
pub mod cache;
#[cfg(feature = "arb")]
pub(crate) mod retained;

const COMPACT_SCHEMA: u32 = 1;
pub const COMPACT_SEMANTICS: &str = "ccm-analytic-sector-gap-reassembly-counts-v1";

/// Producer provenance. Matrix reconstruction hashes, rather than this caller
/// supplied provenance, bind the mathematical source checked by the verifier.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionSourceIdentity {
    pub toolkit_revision: String,
    pub source_tree_digest: ContentDigest,
    pub build_identity: String,
}

impl ReconstructionSourceIdentity {
    /// Pin the compiled workspace source-byte closure and target. The revision
    /// labels provenance; uncommitted inputs are covered by the build digest.
    pub fn current(toolkit_revision: impl Into<String>) -> Self {
        let build = env!("XC_CHECKPOINT_BUILD_ID");
        Self {
            toolkit_revision: toolkit_revision.into(),
            source_tree_digest: ContentDigest(build.to_owned()),
            build_identity: format!("xc-checkpoint-source-byte-inputs-v2:{build}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactSelectedEigenvalue {
    pub dimension: usize,
    pub matrix_digest: ContentDigest,
    pub requested_index: usize,
    pub lower: ExactRationalRecord,
    pub upper: ExactRationalRecord,
    pub lower_count_below: usize,
    pub lower_positive_above: usize,
    pub upper_count_below: usize,
    pub upper_positive_above: usize,
    pub inertia_precision_bits: u32,
}

impl CompactSelectedEigenvalue {
    fn from_selected(value: &ExactSelectedEigenvalueEnclosure) -> Result<Self> {
        if value.lower_boundary.threshold != value.lower
            || value.upper_boundary.threshold != value.upper
            || value.first_enclosed_index != value.requested_index
            || value.last_enclosed_index != value.requested_index
            || !value.simple
        {
            bail!("compact selected certificate requires a single isolated index");
        }
        Ok(Self {
            dimension: value.dimension,
            matrix_digest: value.matrix_digest.clone(),
            requested_index: value.requested_index,
            lower: value.lower.clone(),
            upper: value.upper.clone(),
            lower_count_below: value.lower_boundary.count_below,
            lower_positive_above: value.lower_boundary.positive_above,
            upper_count_below: value.upper_boundary.count_below,
            upper_positive_above: value.upper_boundary.positive_above,
            inertia_precision_bits: value.inertia_precision_bits.context(
                "compact certificates require a recorded directed selected proof precision",
            )?,
        })
    }

    fn validate(&self, dimension: usize, index: usize) -> Result<()> {
        if self.dimension != dimension
            || self.requested_index != index
            || !self.matrix_digest.validate()
            || !(64..=1_000_000).contains(&self.inertia_precision_bits)
            || self.lower_count_below != index
            || self.upper_count_below != index + 1
            || self.lower_positive_above != dimension - index
            || self.upper_positive_above != dimension - index - 1
            || parse(&self.lower)? >= parse(&self.upper)?
        {
            bail!("compact selected thresholds, counts, index or precision are invalid");
        }
        Ok(())
    }

    #[cfg(feature = "arb")]
    fn replay(&self, matrix: &[RationalInterval]) -> Result<()> {
        let lower = parse(&self.lower)?;
        let upper = parse(&self.upper)?;
        let width = upper.clone() - lower.clone();
        let result = certify_selected_interval_eigenvalue_mpfr(
            matrix,
            self.dimension,
            self.requested_index,
            lower,
            upper,
            width,
            1,
            self.inertia_precision_bits,
        );
        match result {
            SelectedEigenvalueEnclosureResult::Conclusive { certificate }
                if Self::from_selected(&certificate)? == *self =>
            {
                Ok(())
            }
            SelectedEigenvalueEnclosureResult::Conclusive { .. } => {
                bail!("fresh compact selected inertia differs from recorded assertion")
            }
            SelectedEigenvalueEnclosureResult::Inconclusive { boundary, reason } => {
                bail!("compact selected replay unresolved at {boundary}: {reason}")
            }
        }
    }
}

/// Configuration and exact assertions for independent analytic reconstruction.
/// The complete serialized object has no field whose record count grows with N.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactCcmSectorGapCertificate {
    pub schema_version: u32,
    pub reconstruction_semantics: String,
    pub source_identity: ReconstructionSourceIdentity,
    pub producer_toolkit_version: String,
    pub integer_cutoff_c: u64,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub geometric_terms: usize,
    pub scalar_backend: String,
    pub assembly_semantics: String,
    pub full_inertia_semantics: String,
    pub selected_inertia_semantics: String,
    pub interval_matrix_semantics: String,
    pub parity_invariance_premise: String,
    pub claim_scope: String,
    pub relative_enclosure_bits: u32,
    pub maximum_bracket_expansions: u32,
    pub guide_even_spectrum_content_digest: ContentDigest,
    pub guide_odd_spectrum_content_digest: ContentDigest,
    pub guide_sector_gap_content_digest: ContentDigest,
    /// SHA-256 of legacy canonical JSON `(D, raw_interval_records)`.
    pub raw_matrix_content_digest: ContentDigest,
    /// SHA-256 of the assembly-semantic/config/backend/component-record tuple.
    pub component_evidence_digest: ContentDigest,
    /// SHA-256 of canonical JSON of the symmetry-intersected record array.
    pub cutoff_free_tau_content_digest: ContentDigest,
    pub full_positive: usize,
    pub full_negative: usize,
    pub even_ground: CompactSelectedEigenvalue,
    pub even_first_excited: CompactSelectedEigenvalue,
    pub odd_ground: CompactSelectedEigenvalue,
    pub even_simplicity_gap_lower: ExactRationalRecord,
    pub even_odd_ordering_gap_lower: ExactRationalRecord,
    pub even_odd_ordering_gap_upper: ExactRationalRecord,
    pub certified_finite_ground_parity: String,
    pub certifies_finite_ground_state_simple: bool,
    pub certifies_finite_matrix_positive_definite: bool,
}

impl CompactCcmSectorGapCertificate {
    /// Lossless projection of mathematical assertions, not legacy payload bytes.
    /// This checks the legacy structure/outcomes but does not replay its proof.
    /// Public compact verification always reconstructs and proves afresh.
    pub fn from_portable(
        value: &PortableCcmSectorGapCertificate,
        source_identity: ReconstructionSourceIdentity,
    ) -> Result<Self> {
        if value.schema_version != SCHEMA_VERSION
            || value.full_matrix_inertia_certificate.zero_or_unresolved != 0
        {
            bail!("compact conversion requires schema 4 and conclusive full inertia");
        }
        sanity_retained_sector_gap_certificate(value)?;
        value.full_matrix_inertia_certificate.validate_structure()?;
        let result = Self {
            schema_version: COMPACT_SCHEMA,
            reconstruction_semantics: COMPACT_SEMANTICS.to_owned(),
            source_identity,
            producer_toolkit_version: value
                .full_matrix_inertia_certificate
                .toolkit_version
                .to_string(),
            integer_cutoff_c: value.integer_cutoff_c,
            n_modes: value.n_modes,
            precision_bits: value.precision_bits,
            geometric_terms: value.geometric_terms,
            scalar_backend: value.scalar_backend.clone(),
            assembly_semantics: ASSEMBLY_SEMANTICS.to_owned(),
            full_inertia_semantics: value
                .full_matrix_inertia_certificate
                .configuration
                .get("inertia_semantics")
                .context("missing full inertia semantics")?
                .clone(),
            selected_inertia_semantics: SELECTED_INERTIA_SEMANTICS.to_owned(),
            interval_matrix_semantics: value.interval_matrix_semantics.clone(),
            parity_invariance_premise: value.parity_invariance_premise.clone(),
            claim_scope: value.claim_scope.clone(),
            relative_enclosure_bits: value.relative_enclosure_bits,
            maximum_bracket_expansions: value.maximum_bracket_expansions,
            guide_even_spectrum_content_digest: value.guide_even_spectrum_content_digest.clone(),
            guide_odd_spectrum_content_digest: value.guide_odd_spectrum_content_digest.clone(),
            guide_sector_gap_content_digest: value.guide_sector_gap_content_digest.clone(),
            raw_matrix_content_digest: value.full_matrix_inertia_certificate.matrix_digest.clone(),
            component_evidence_digest: value.component_evidence_digest.clone(),
            cutoff_free_tau_content_digest: value.cutoff_free_tau_content_digest.clone(),
            full_positive: value.full_matrix_inertia_certificate.positive,
            full_negative: value.full_matrix_inertia_certificate.negative,
            even_ground: CompactSelectedEigenvalue::from_selected(&value.even_ground)?,
            even_first_excited: CompactSelectedEigenvalue::from_selected(
                &value.even_first_excited,
            )?,
            odd_ground: CompactSelectedEigenvalue::from_selected(&value.odd_ground)?,
            even_simplicity_gap_lower: value.even_simplicity_gap.certified_lower_bound.clone(),
            even_odd_ordering_gap_lower: value.even_odd_ordering_gap_lower.clone(),
            even_odd_ordering_gap_upper: value.even_odd_ordering_gap_upper.clone(),
            certified_finite_ground_parity: value.certified_finite_ground_parity.clone(),
            certifies_finite_ground_state_simple: value.certifies_finite_ground_state_simple,
            certifies_finite_matrix_positive_definite: value
                .certifies_finite_matrix_positive_definite,
        };
        result.validate_structure()?;
        Ok(result)
    }

    /// Scalar consistency and identity checks only; not mathematical replay.
    pub fn validate_structure(&self) -> Result<()> {
        let dimension = checked_full_dimension(self.n_modes)?;
        if self.schema_version != COMPACT_SCHEMA
            || self.reconstruction_semantics != COMPACT_SEMANTICS
            || self.integer_cutoff_c <= 1
            || !(64..=1_000_000).contains(&self.precision_bits)
            || self.geometric_terms == 0
            || self.scalar_backend.trim().is_empty()
            || self.source_identity.toolkit_revision.trim().is_empty()
            || !self.source_identity.source_tree_digest.validate()
            || self.source_identity.build_identity.trim().is_empty()
            || self.producer_toolkit_version.trim().is_empty()
            || self.assembly_semantics != ASSEMBLY_SEMANTICS
            || !matches!(
                self.full_inertia_semantics.as_str(),
                INERTIA_SEMANTICS | LEGACY_INERTIA_SEMANTICS
            )
            || self.selected_inertia_semantics != SELECTED_INERTIA_SEMANTICS
            || self.interval_matrix_semantics != INTERVAL_MATRIX_SEMANTICS
            || self.parity_invariance_premise != PARITY_INVARIANCE_PREMISE
            || self.claim_scope != CLAIM_SCOPE
            || self.full_positive.checked_add(self.full_negative) != Some(dimension)
        {
            bail!("compact certificate metadata, source identity or finite scope is invalid");
        }
        CcmSectorGapCertificationOptions {
            relative_enclosure_bits: self.relative_enclosure_bits,
            maximum_bracket_expansions: self.maximum_bracket_expansions,
        }
        .validate()?;
        for digest in [
            &self.guide_even_spectrum_content_digest,
            &self.guide_odd_spectrum_content_digest,
            &self.guide_sector_gap_content_digest,
            &self.raw_matrix_content_digest,
            &self.component_evidence_digest,
            &self.cutoff_free_tau_content_digest,
        ] {
            if !digest.validate() {
                bail!("compact certificate digest is invalid");
            }
        }
        self.even_ground.validate(self.n_modes + 1, 0)?;
        self.even_first_excited.validate(self.n_modes + 1, 1)?;
        self.odd_ground.validate(self.n_modes, 0)?;
        if self.even_ground.matrix_digest != self.even_first_excited.matrix_digest {
            bail!("compact even proofs bind different matrices");
        }
        let simplicity = parse(&self.even_first_excited.lower)? - parse(&self.even_ground.upper)?;
        let lower = parse(&self.odd_ground.lower)? - parse(&self.even_ground.upper)?;
        let upper = parse(&self.odd_ground.upper)? - parse(&self.even_ground.lower)?;
        let parity = if lower > 0 {
            "even"
        } else if upper < 0 {
            "odd"
        } else {
            "unresolved"
        };
        if simplicity <= 0
            || rational_record(&simplicity) != self.even_simplicity_gap_lower
            || rational_record(&lower) != self.even_odd_ordering_gap_lower
            || rational_record(&upper) != self.even_odd_ordering_gap_upper
            || self.certified_finite_ground_parity != parity
            || self.certifies_finite_ground_state_simple != (parity != "unresolved")
            || self.certifies_finite_matrix_positive_definite != (self.full_positive == dimension)
        {
            bail!("compact finite scalar outcomes differ from exact bracket/count arithmetic");
        }
        Ok(())
    }
}

/// Immutable successful reconstruction for consumers needing a directed residual.
/// Private fields prevent constructing a verified handle without fresh replay.
#[cfg(feature = "arb")]
pub struct VerifiedCompactSectorGap {
    certificate: CompactCcmSectorGapCertificate,
    canonical_tau: Vec<ExactRationalIntervalRecord>,
    _retention: xc_core::MemoryReservation,
}

#[cfg(feature = "arb")]
impl VerifiedCompactSectorGap {
    pub fn certificate(&self) -> &CompactCcmSectorGapCertificate {
        &self.certificate
    }
    pub fn canonical_tau(&self) -> &[ExactRationalIntervalRecord] {
        &self.canonical_tau
    }
}

/// Always assemble afresh, check all exact source hashes, and independently
/// prove full and selected inertia. Does not reuse retained verification caches.
#[cfg(feature = "arb")]
pub fn reconstruct_verified_compact_sector_gap(
    certificate: &CompactCcmSectorGapCertificate,
) -> Result<VerifiedCompactSectorGap> {
    use xc_certify::exact::{
        interval_symmetric_ldlt_inertia_mpfr, interval_symmetric_ldlt_inertia_mpfr_stable,
        IntervalInertiaResult,
    };
    certificate.validate_structure()?;
    #[cfg(test)]
    retained::FRESH_REPLAYS.with(|n| n.set(n.get() + 1));
    // Standalone verification has the same explicit default budget as capture;
    // a run's owner, when present, remains authoritative.
    let resources = super::runtime_resources()?;
    let _scope = resources.enter();
    let _workspace = super::reserve_certificate_workspace(
        certificate.n_modes,
        certificate
            .precision_bits
            .max(certificate.even_ground.inertia_precision_bits)
            .max(certificate.even_first_excited.inertia_precision_bits)
            .max(certificate.odd_ground.inertia_precision_bits),
        certificate.integer_cutoff_c,
        certificate.geometric_terms,
    )?;
    let dimension = checked_full_dimension(certificate.n_modes)?;
    let assembled =
        super::super::cutoff_free::assemble(&super::super::cutoff_free::CutoffFreeConfig {
            integer_cutoff_c: certificate.integer_cutoff_c,
            modes: certificate.n_modes,
            precision_bits: certificate.precision_bits,
            geometric_terms: certificate.geometric_terms,
        })?;
    if assembled.scalar_backend != certificate.scalar_backend
        || assembled.component_evidence_digest()? != certificate.component_evidence_digest
        || ContentDigest::sha256_json(&(dimension, interval_records(&assembled.tau)))?
            != certificate.raw_matrix_content_digest
    {
        bail!("compact raw analytic assembly, backend or component hash mismatch");
    }
    let inertia = if certificate.full_inertia_semantics == INERTIA_SEMANTICS {
        interval_symmetric_ldlt_inertia_mpfr_stable(
            &assembled.tau,
            dimension,
            certificate.precision_bits,
        )?
    } else {
        interval_symmetric_ldlt_inertia_mpfr(&assembled.tau, dimension, certificate.precision_bits)?
    };
    match inertia {
        IntervalInertiaResult::Conclusive {
            positive, negative, ..
        } if positive == certificate.full_positive && negative == certificate.full_negative => {}
        IntervalInertiaResult::Conclusive { .. } => bail!("compact full inertia counts mismatch"),
        IntervalInertiaResult::Inconclusive { reason, .. } => {
            bail!("compact full inertia unresolved: {reason}")
        }
    }
    let canonical = canonicalize_full_tau(&assembled.tau, dimension)?;
    drop(assembled);
    let records = interval_records(&canonical);
    if records_digest(&records)? != certificate.cutoff_free_tau_content_digest {
        bail!("compact canonical Tau hash mismatch");
    }
    let (even, odd) =
        project_parity_sectors(&canonical, certificate.n_modes, certificate.precision_bits)?;
    drop(canonical);
    certificate.even_ground.replay(&even)?;
    certificate.even_first_excited.replay(&even)?;
    drop(even);
    certificate.odd_ground.replay(&odd)?;
    drop(odd);
    // Transfer to an exact resident allowance before the scratch reservation
    // leaves scope. If transition headroom is unavailable, fail closed rather
    // than returning an uncharged live matrix. Only the resident charge follows
    // the handle; the dense proof workspace is released on return.
    let bytes = resident_bytes(certificate, &records, records.capacity())?;
    let retention = resources.try_reserve_retained(bytes)?;
    Ok(VerifiedCompactSectorGap {
        certificate: certificate.clone(),
        canonical_tau: records,
        _retention: retention,
    })
}

#[cfg(feature = "arb")]
fn resident_bytes(
    certificate: &CompactCcmSectorGapCertificate,
    records: &[ExactRationalIntervalRecord],
    record_capacity: usize,
) -> Result<u64> {
    // Capacity-based matrix accounting avoids a fixed dimension cutoff or an
    // arbitrary per-run cache share. Double capacities include allocator slack;
    // the small scalar certificate uses the shared conservative JSON allowance.
    let mut bytes = u128::from(super::super::capture_runtime::retained_json_bytes(
        certificate,
    )?);
    bytes += (record_capacity as u128)
        * (std::mem::size_of::<ExactRationalIntervalRecord>() as u128)
        * 2;
    for interval in records {
        for scalar in [&interval.lower, &interval.upper] {
            bytes += ((scalar.numerator.capacity() as u128)
                + (scalar.denominator.capacity() as u128)
                + 64)
                * 2;
        }
    }
    u64::try_from(bytes).context("compact resident allocation size overflow")
}

/// Independently verify the finite analytic assertions from the compact recipe.
#[cfg(feature = "arb")]
pub fn verify_compact_ccm_sector_gap_certificate(
    certificate: &CompactCcmSectorGapCertificate,
) -> VerificationReport {
    match reconstruct_verified_compact_sector_gap(certificate) {
        Ok(_) => VerificationReport {
            valid: true,
            mathematical_claim_verified: true,
            checks: vec![
                "analytic matrix freshly reassembled; backend, raw/component/canonical hashes matched".into(),
                "full zero-shift and all six selected shifted-inertia counts freshly proved".into(),
                "finite parity, simplicity and positivity outcomes checked from proven counts".into(),
            ],
            warnings: vec![
                format!("sector conclusions inherit this exact analytic premise: {PARITY_INVARIANCE_PREMISE}"),
                "one finite matrix only; no continuum, tail, convergence or global theorem".into(),
                "source/build pins are producer provenance; reconstruction hashes authenticate the analytic interval data, not the producer executable".into(),
            ],
            errors: Vec::new(),
        },
        Err(error) => invalid_report(error.to_string()),
    }
}

#[cfg(all(test, feature = "arb"))]
#[path = "compact_tests.rs"]
mod tests;
