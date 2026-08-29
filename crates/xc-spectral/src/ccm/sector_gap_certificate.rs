#![cfg_attr(not(feature = "arb"), allow(dead_code))]

//! Exact finite-dimensional parity, ordering, and simplicity certificates.
//!
//! Numerical sector spectra remain the discovery guide.  The proof itself is
//! rebuilt from the cutoff-free interval enclosure of the full CCM Weil
//! matrix, projected into exact reflection-parity sectors, and replayed with
//! exact-rational shifted-inertia counts.  The resulting claim is strictly
//! about one recorded finite `(c, N)` matrix; it is not a continuum or
//! convergence theorem.

use anyhow::{bail, Context, Result};
use rug::{ops::Pow, Float, Integer, Rational};
use serde::{Deserialize, Serialize};
use xc_cache::ContentDigest;
#[cfg(feature = "arb")]
use xc_cache::SemanticKeyEnvelope;
use xc_certify::exact::{
    certify_exact_spectral_gap, certify_selected_interval_eigenvalue, interval_record, parse,
    parse_interval, rational_record, verify_exact_spectral_gap_certificate,
    verify_portable_interval_inertia_certificate, verify_selected_interval_eigenvalue_enclosure,
};
use xc_certify::{
    ExactRationalIntervalRecord, ExactRationalRecord, ExactSelectedEigenvalueEnclosure,
    ExactSpectralGapCertificate, PortableIntervalInertiaCertificate,
    SelectedEigenvalueEnclosureResult, VerificationReport,
};
use xc_numerics::interval::RationalInterval;

#[cfg(feature = "arb")]
use super::CcmParams;

const SCHEMA_VERSION: u32 = 2;
const CLAIM_SCOPE: &str =
    "cutoff_free_finite_ccm_conditional_sector_ordering_simplicity_and_unconditional_full_inertia";
const INTERVAL_MATRIX_SEMANTICS: &str =
    "raw_full_cutoff_free_tau_certified_by_interval_ldlt_then_reflection_orbits_intersected_for_conditional_parity_projection";
const PARITY_INVARIANCE_PREMISE: &str =
    "the exact closed_form_ccm_tau matrix is centrosymmetric under index reflection";

/// Cost and discovery controls for the finite sector-gap certificate.
///
/// The numerical eigenvalues determine only the initial brackets.  Each
/// accepted bracket is independently proved by exact-rational shifted
/// inertia.  Expansions are bounded so a positive guide never receives a
/// bracket crossing zero merely because of this discovery policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CcmSectorGapCertificationOptions {
    pub relative_enclosure_bits: u32,
    pub maximum_bracket_expansions: u32,
}

impl Default for CcmSectorGapCertificationOptions {
    fn default() -> Self {
        Self {
            relative_enclosure_bits: 20,
            maximum_bracket_expansions: 8,
        }
    }
}

impl CcmSectorGapCertificationOptions {
    pub fn validate(&self) -> Result<()> {
        if self.relative_enclosure_bits < 8 {
            bail!("CCM sector-gap certification requires at least 8 relative enclosure bits");
        }
        if self.maximum_bracket_expansions == 0
            || self.maximum_bracket_expansions >= self.relative_enclosure_bits
        {
            bail!(
                "CCM sector-gap certification requires 0 < maximum bracket expansions < relative enclosure bits"
            );
        }
        Ok(())
    }
}

/// Self-contained exact certificate for the bottom of both reflection-parity
/// sectors of one finite cutoff-free CCM Weil matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortableCcmSectorGapCertificate {
    pub schema_version: u32,
    pub lambda_squared: String,
    pub integer_cutoff_c: u64,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub geometric_terms: usize,
    pub scalar_backend: String,
    pub interval_matrix_semantics: String,
    pub relative_enclosure_bits: u32,
    pub maximum_bracket_expansions: u32,
    pub guide_even_spectrum_content_digest: ContentDigest,
    pub guide_odd_spectrum_content_digest: ContentDigest,
    pub guide_sector_gap_content_digest: ContentDigest,
    pub guide_even_ground: String,
    pub guide_even_first_excited: String,
    pub guide_odd_ground: String,
    pub certification_even_ground_guide: String,
    pub certification_even_first_excited_guide: String,
    pub certification_odd_ground_guide: String,
    pub component_evidence_digest: ContentDigest,
    pub full_matrix_inertia_certificate: PortableIntervalInertiaCertificate,
    pub cutoff_free_tau_content_digest: ContentDigest,
    pub cutoff_free_tau: Vec<ExactRationalIntervalRecord>,
    pub even_ground: ExactSelectedEigenvalueEnclosure,
    pub even_first_excited: ExactSelectedEigenvalueEnclosure,
    pub odd_ground: ExactSelectedEigenvalueEnclosure,
    pub even_simplicity_gap: ExactSpectralGapCertificate,
    pub even_odd_ordering_gap_lower: ExactRationalRecord,
    pub even_odd_ordering_gap_upper: ExactRationalRecord,
    pub certified_finite_ground_parity: String,
    pub certifies_finite_ground_state_simple: bool,
    pub certifies_finite_matrix_positive_definite: bool,
    pub parity_invariance_premise: String,
    pub claim_scope: String,
}

fn invalid_report(message: impl Into<String>) -> VerificationReport {
    VerificationReport {
        valid: false,
        checks: Vec::new(),
        warnings: Vec::new(),
        errors: vec![message.into()],
    }
}

fn merge_report(report: VerificationReport, checks: &mut Vec<String>) -> Result<()> {
    if !report.valid {
        bail!(report.errors.join("; "));
    }
    checks.extend(report.checks);
    Ok(())
}

fn records_digest(records: &[ExactRationalIntervalRecord]) -> Result<ContentDigest> {
    Ok(ContentDigest::sha256(
        &serde_json::to_vec(records).context("serialize exact interval records")?,
    ))
}

/// Intersect every entry with its transpose and reflected counterparts.
///
/// The exact CCM form is symmetric and centrosymmetric.  Independent interval
/// evaluations may have slightly different widths, so intersection retains
/// only values compatible with all four mathematically identical entries.
fn canonicalize_full_tau(
    matrix: &[RationalInterval],
    dimension: usize,
) -> Result<Vec<RationalInterval>> {
    if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
        bail!("CCM sector-gap certificate requires a nonempty square full Tau enclosure");
    }
    let mut canonical = matrix.to_vec();
    for row in 0..dimension {
        for column in 0..dimension {
            let reflected_row = dimension - 1 - row;
            let reflected_column = dimension - 1 - column;
            let orbit = [
                row * dimension + column,
                column * dimension + row,
                reflected_row * dimension + reflected_column,
                reflected_column * dimension + reflected_row,
            ];
            let mut common = canonical[orbit[0]].clone();
            for &index in orbit.iter().skip(1) {
                common = common.intersection(&canonical[index]).ok_or_else(|| {
                    anyhow::anyhow!(
                        "cutoff-free Tau enclosures have disjoint symmetry-orbit entries at ({row}, {column})"
                    )
                })?;
            }
            for index in orbit {
                canonical[index] = common.clone();
            }
        }
    }
    Ok(canonical)
}

fn project_parity_sectors(
    tau: &[RationalInterval],
    n_modes: usize,
    precision_bits: u32,
) -> Result<(Vec<RationalInterval>, Vec<RationalInterval>)> {
    if n_modes == 0 {
        bail!("CCM sector-gap certification requires N >= 1");
    }
    let full_dimension = 2 * n_modes + 1;
    if tau.len() != full_dimension * full_dimension {
        bail!("CCM sector-gap projection received the wrong full Tau dimension");
    }
    let center = n_modes;
    let sqrt_two = RationalInterval::point(Rational::from(2))
        .sqrt_nonnegative(precision_bits)
        .map_err(anyhow::Error::from)?;
    let two = RationalInterval::point(Rational::from(2));
    let at = |row: usize, column: usize| &tau[row * full_dimension + column];

    let even_dimension = n_modes + 1;
    let mut even = Vec::with_capacity(even_dimension * even_dimension);
    for row in 0..even_dimension {
        for column in 0..even_dimension {
            let value = match (row, column) {
                (0, 0) => at(center, center).clone(),
                (0, j) => at(center, center - j)
                    .add(at(center, center + j))
                    .div(&sqrt_two)
                    .map_err(anyhow::Error::from)?,
                (k, 0) => at(center - k, center)
                    .add(at(center + k, center))
                    .div(&sqrt_two)
                    .map_err(anyhow::Error::from)?,
                (k, j) => at(center - k, center - j)
                    .add(at(center - k, center + j))
                    .add(at(center + k, center - j))
                    .add(at(center + k, center + j))
                    .div(&two)
                    .map_err(anyhow::Error::from)?,
            };
            even.push(value);
        }
    }

    let mut odd = Vec::with_capacity(n_modes * n_modes);
    for row in 0..n_modes {
        let k = row + 1;
        for column in 0..n_modes {
            let j = column + 1;
            let value = at(center - k, center - j)
                .sub(at(center - k, center + j))
                .sub(at(center + k, center - j))
                .add(at(center + k, center + j))
                .div(&two)
                .map_err(anyhow::Error::from)?;
            odd.push(value);
        }
    }
    Ok((even, odd))
}

fn exact_guide(guide: &Float, name: &str) -> Result<Rational> {
    if !guide.is_finite() || guide.is_zero() {
        bail!("CCM sector-gap {name} discovery guide must be finite and nonzero");
    }
    guide
        .to_rational()
        .ok_or_else(|| anyhow::anyhow!("CCM sector-gap {name} guide is not exactly rationalizable"))
}

fn interval_midpoint_guides(
    matrix: &[RationalInterval],
    dimension: usize,
    requested_count: usize,
    precision_bits: u32,
) -> Result<Vec<Float>> {
    if requested_count == 0 || requested_count > dimension {
        bail!("CCM sector-gap midpoint guide requested an invalid eigenvalue prefix");
    }
    let midpoint = matrix
        .iter()
        .map(|value| Float::with_val(precision_bits, value.midpoint()))
        .collect::<Vec<_>>();
    let (diagonal, off_diagonal) =
        xc_numerics::eigen::dense_symmetric_tridiagonal_hp(&midpoint, dimension, precision_bits)?;
    let tolerance =
        Float::with_val(precision_bits, 2).pow(-((precision_bits.saturating_sub(32)) as i32));
    let selected = xc_numerics::eigen::tridiag_selected_eigenvalues_hp(
        &diagonal,
        &off_diagonal,
        0,
        requested_count - 1,
        &tolerance,
        (precision_bits as usize).saturating_mul(2),
        precision_bits,
    )?;
    Ok(selected
        .enclosures
        .into_iter()
        .map(|enclosure| {
            let mut midpoint = enclosure.lower;
            midpoint += enclosure.upper;
            midpoint /= 2;
            midpoint
        })
        .collect())
}

fn guided_selected_certificate(
    matrix: &[RationalInterval],
    dimension: usize,
    requested_index: usize,
    guide: &Float,
    name: &str,
    options: CcmSectorGapCertificationOptions,
) -> Result<ExactSelectedEigenvalueEnclosure> {
    let center = exact_guide(guide, name)?;
    let mut radius = center.clone().abs();
    radius /= Rational::from(Integer::from(1) << options.relative_enclosure_bits);
    let mut failures = Vec::new();
    for expansion in 0..=options.maximum_bracket_expansions {
        let lower = center.clone() - radius.clone();
        let upper = center.clone() + radius.clone();
        let width = upper.clone() - lower.clone();
        match certify_selected_interval_eigenvalue(
            matrix,
            dimension,
            requested_index,
            lower,
            upper,
            width,
            1,
        ) {
            SelectedEigenvalueEnclosureResult::Conclusive { certificate } => {
                if !certificate.simple {
                    bail!(
                        "CCM sector-gap {name} bracket enclosed multiple eigenvalues at expansion {expansion}"
                    );
                }
                return Ok(*certificate);
            }
            SelectedEigenvalueEnclosureResult::Inconclusive { boundary, reason } => {
                failures.push(format!("expansion {expansion}, {boundary}: {reason}"));
            }
        }
        radius *= 2;
    }
    bail!(
        "CCM sector-gap {name} certification remained inconclusive: {}",
        failures.join(" | ")
    )
}

#[allow(clippy::too_many_arguments)]
fn build_certificate_from_tau(
    lambda_squared: String,
    integer_cutoff_c: u64,
    n_modes: usize,
    precision_bits: u32,
    geometric_terms: usize,
    scalar_backend: String,
    component_evidence_digest: ContentDigest,
    full_matrix_inertia_certificate: PortableIntervalInertiaCertificate,
    even_ground_guide: &Float,
    even_first_excited_guide: &Float,
    odd_ground_guide: &Float,
    even_spectrum_digest: ContentDigest,
    odd_spectrum_digest: ContentDigest,
    sector_gap_digest: ContentDigest,
    options: CcmSectorGapCertificationOptions,
) -> Result<PortableCcmSectorGapCertificate> {
    options.validate()?;
    if n_modes == 0 || precision_bits < 64 || geometric_terms == 0 {
        bail!("CCM sector-gap certificate metadata is outside the supported finite model");
    }
    for digest in [
        &component_evidence_digest,
        &even_spectrum_digest,
        &odd_spectrum_digest,
        &sector_gap_digest,
    ] {
        if !digest.validate() {
            bail!("CCM sector-gap certificate received an invalid source digest");
        }
    }

    let full_dimension = 2 * n_modes + 1;
    let inertia_report =
        verify_portable_interval_inertia_certificate(&full_matrix_inertia_certificate);
    if !inertia_report.valid {
        bail!(
            "CCM full-matrix inertia certificate failed exact replay: {}",
            inertia_report.errors.join("; ")
        );
    }
    let expected_cutoff = integer_cutoff_c.to_string();
    let expected_modes = n_modes.to_string();
    let expected_geometric_terms = geometric_terms.to_string();
    if full_matrix_inertia_certificate.dimension != full_dimension
        || full_matrix_inertia_certificate.precision_bits != precision_bits
        || full_matrix_inertia_certificate.scalar_backend != scalar_backend
        || full_matrix_inertia_certificate.assembly_evidence_digest != component_evidence_digest
        || full_matrix_inertia_certificate
            .configuration
            .get("integer_cutoff_c")
            != Some(&expected_cutoff)
        || full_matrix_inertia_certificate.configuration.get("modes") != Some(&expected_modes)
        || full_matrix_inertia_certificate
            .configuration
            .get("geometric_terms")
            != Some(&expected_geometric_terms)
    {
        bail!("CCM full-matrix inertia certificate metadata does not match the sector certificate");
    }
    let full_tau = full_matrix_inertia_certificate
        .matrix_row_major
        .iter()
        .map(parse_interval)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(anyhow::Error::from)?;
    let canonical_tau = canonicalize_full_tau(&full_tau, full_dimension)?;
    let cutoff_free_tau = canonical_tau
        .iter()
        .map(interval_record)
        .collect::<Vec<_>>();
    let cutoff_free_tau_content_digest = records_digest(&cutoff_free_tau)?;
    let (even, odd) = project_parity_sectors(&canonical_tau, n_modes, precision_bits)?;
    // Retained sector spectra remain provenance-bound research guides. Near
    // a deeply cancelled ground state, however, a tiny entrywise difference
    // between that point matrix and the cutoff-free interval midpoint can be
    // many relative orders at the eigenvalue. Use native midpoint guides for
    // proof search; exact shifted inertia remains the verifier.
    let even_certification_guides =
        interval_midpoint_guides(&even, n_modes + 1, 2, precision_bits)?;
    let odd_certification_guides = interval_midpoint_guides(&odd, n_modes, 1, precision_bits)?;
    let even_ground = guided_selected_certificate(
        &even,
        n_modes + 1,
        0,
        &even_certification_guides[0],
        "even ground",
        options,
    )?;
    let even_first_excited = guided_selected_certificate(
        &even,
        n_modes + 1,
        1,
        &even_certification_guides[1],
        "even first-excited",
        options,
    )?;
    let odd_ground = guided_selected_certificate(
        &odd,
        n_modes,
        0,
        &odd_certification_guides[0],
        "odd ground",
        options,
    )?;
    let even_simplicity_gap = certify_exact_spectral_gap(&even_ground, &even_first_excited)
        .map_err(anyhow::Error::from)?;
    let ordering_gap_lower = parse(&odd_ground.lower).map_err(anyhow::Error::from)?
        - parse(&even_ground.upper).map_err(anyhow::Error::from)?;
    let ordering_gap_upper = parse(&odd_ground.upper).map_err(anyhow::Error::from)?
        - parse(&even_ground.lower).map_err(anyhow::Error::from)?;
    let certified_finite_ground_parity = if ordering_gap_lower > 0 {
        "even"
    } else if ordering_gap_upper < 0 {
        "odd"
    } else {
        "unresolved"
    };
    let full_matrix_positive = full_matrix_inertia_certificate.positive == full_dimension
        && full_matrix_inertia_certificate.negative == 0
        && full_matrix_inertia_certificate.zero_or_unresolved == 0;
    Ok(PortableCcmSectorGapCertificate {
        schema_version: SCHEMA_VERSION,
        lambda_squared,
        integer_cutoff_c,
        n_modes,
        precision_bits,
        geometric_terms,
        scalar_backend,
        interval_matrix_semantics: INTERVAL_MATRIX_SEMANTICS.to_owned(),
        relative_enclosure_bits: options.relative_enclosure_bits,
        maximum_bracket_expansions: options.maximum_bracket_expansions,
        guide_even_spectrum_content_digest: even_spectrum_digest,
        guide_odd_spectrum_content_digest: odd_spectrum_digest,
        guide_sector_gap_content_digest: sector_gap_digest,
        guide_even_ground: even_ground_guide.to_string(),
        guide_even_first_excited: even_first_excited_guide.to_string(),
        guide_odd_ground: odd_ground_guide.to_string(),
        certification_even_ground_guide: even_certification_guides[0].to_string(),
        certification_even_first_excited_guide: even_certification_guides[1].to_string(),
        certification_odd_ground_guide: odd_certification_guides[0].to_string(),
        component_evidence_digest,
        full_matrix_inertia_certificate,
        cutoff_free_tau_content_digest,
        cutoff_free_tau,
        even_ground,
        even_first_excited,
        odd_ground,
        even_simplicity_gap,
        even_odd_ordering_gap_lower: rational_record(&ordering_gap_lower),
        even_odd_ordering_gap_upper: rational_record(&ordering_gap_upper),
        certified_finite_ground_parity: certified_finite_ground_parity.to_owned(),
        certifies_finite_ground_state_simple: certified_finite_ground_parity != "unresolved",
        certifies_finite_matrix_positive_definite: full_matrix_positive,
        parity_invariance_premise: PARITY_INVARIANCE_PREMISE.to_owned(),
        claim_scope: CLAIM_SCOPE.to_owned(),
    })
}

/// Independently replay the exact finite-sector claims from the stored Tau
/// intervals.  No numerical guide value participates in the proof replay.
pub fn verify_portable_ccm_sector_gap_certificate(
    certificate: &PortableCcmSectorGapCertificate,
) -> VerificationReport {
    let verify = || -> Result<Vec<String>> {
        if certificate.schema_version != SCHEMA_VERSION
            || certificate.lambda_squared != certificate.integer_cutoff_c.to_string()
            || certificate.n_modes == 0
            || certificate.precision_bits < 64
            || certificate.geometric_terms == 0
            || certificate.scalar_backend.trim().is_empty()
            || certificate.interval_matrix_semantics != INTERVAL_MATRIX_SEMANTICS
            || certificate.parity_invariance_premise != PARITY_INVARIANCE_PREMISE
            || certificate.claim_scope != CLAIM_SCOPE
        {
            bail!("CCM sector-gap certificate metadata or finite claim scope is invalid");
        }
        CcmSectorGapCertificationOptions {
            relative_enclosure_bits: certificate.relative_enclosure_bits,
            maximum_bracket_expansions: certificate.maximum_bracket_expansions,
        }
        .validate()?;
        for digest in [
            &certificate.guide_even_spectrum_content_digest,
            &certificate.guide_odd_spectrum_content_digest,
            &certificate.guide_sector_gap_content_digest,
            &certificate.component_evidence_digest,
            &certificate.cutoff_free_tau_content_digest,
        ] {
            if !digest.validate() {
                bail!("CCM sector-gap certificate contains an invalid SHA-256 digest");
            }
        }
        for guide in [
            &certificate.guide_even_ground,
            &certificate.guide_even_first_excited,
            &certificate.guide_odd_ground,
            &certificate.certification_even_ground_guide,
            &certificate.certification_even_first_excited_guide,
            &certificate.certification_odd_ground_guide,
        ] {
            let value = Float::with_val(
                certificate.precision_bits,
                Float::parse(guide).map_err(|error| anyhow::anyhow!(error.to_string()))?,
            );
            if !value.is_finite() || value.is_zero() {
                bail!("CCM sector-gap certificate has an invalid numerical discovery guide");
            }
        }
        let full_dimension = 2 * certificate.n_modes + 1;
        let expected_cutoff = certificate.integer_cutoff_c.to_string();
        let expected_modes = certificate.n_modes.to_string();
        let expected_geometric_terms = certificate.geometric_terms.to_string();
        if certificate.full_matrix_inertia_certificate.dimension != full_dimension
            || certificate.full_matrix_inertia_certificate.precision_bits
                != certificate.precision_bits
            || certificate.full_matrix_inertia_certificate.scalar_backend
                != certificate.scalar_backend
            || certificate
                .full_matrix_inertia_certificate
                .assembly_evidence_digest
                != certificate.component_evidence_digest
            || certificate
                .full_matrix_inertia_certificate
                .configuration
                .get("integer_cutoff_c")
                != Some(&expected_cutoff)
            || certificate
                .full_matrix_inertia_certificate
                .configuration
                .get("modes")
                != Some(&expected_modes)
            || certificate
                .full_matrix_inertia_certificate
                .configuration
                .get("geometric_terms")
                != Some(&expected_geometric_terms)
        {
            bail!("CCM full-matrix inertia metadata does not match the sector certificate");
        }
        let mut checks = Vec::new();
        merge_report(
            verify_portable_interval_inertia_certificate(
                &certificate.full_matrix_inertia_certificate,
            ),
            &mut checks,
        )?;
        if records_digest(&certificate.cutoff_free_tau)?
            != certificate.cutoff_free_tau_content_digest
        {
            bail!("CCM sector-gap cutoff-free Tau digest does not match its exact records");
        }
        let raw_tau = certificate
            .full_matrix_inertia_certificate
            .matrix_row_major
            .iter()
            .map(parse_interval)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::from)?;
        let canonical = canonicalize_full_tau(&raw_tau, full_dimension)?;
        let expected_canonical_records = canonical.iter().map(interval_record).collect::<Vec<_>>();
        if expected_canonical_records != certificate.cutoff_free_tau {
            bail!(
                "CCM parity-projection Tau records do not derive from the certified raw full matrix"
            );
        }
        let tau = certificate
            .cutoff_free_tau
            .iter()
            .map(parse_interval)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::from)?;
        if canonicalize_full_tau(&tau, full_dimension)? != tau {
            bail!("CCM sector-gap Tau records are not in canonical symmetry-orbit form");
        }
        let (even, odd) =
            project_parity_sectors(&tau, certificate.n_modes, certificate.precision_bits)?;
        let expected_even_guides = interval_midpoint_guides(
            &even,
            certificate.n_modes + 1,
            2,
            certificate.precision_bits,
        )?;
        let expected_odd_guides =
            interval_midpoint_guides(&odd, certificate.n_modes, 1, certificate.precision_bits)?;
        if certificate.certification_even_ground_guide != expected_even_guides[0].to_string()
            || certificate.certification_even_first_excited_guide
                != expected_even_guides[1].to_string()
            || certificate.certification_odd_ground_guide != expected_odd_guides[0].to_string()
        {
            bail!("CCM cutoff-free midpoint discovery guides differ from exact Tau replay");
        }
        if certificate.even_ground.requested_index != 0
            || certificate.even_first_excited.requested_index != 1
            || certificate.odd_ground.requested_index != 0
            || !certificate.even_ground.simple
            || !certificate.even_first_excited.simple
            || !certificate.odd_ground.simple
        {
            bail!("CCM sector-gap selected-eigenvalue indices or simplicity records are invalid");
        }
        merge_report(
            verify_selected_interval_eigenvalue_enclosure(&certificate.even_ground, &even),
            &mut checks,
        )?;
        merge_report(
            verify_selected_interval_eigenvalue_enclosure(&certificate.even_first_excited, &even),
            &mut checks,
        )?;
        merge_report(
            verify_selected_interval_eigenvalue_enclosure(&certificate.odd_ground, &odd),
            &mut checks,
        )?;
        merge_report(
            verify_exact_spectral_gap_certificate(&certificate.even_simplicity_gap),
            &mut checks,
        )?;
        let recomputed_gap =
            certify_exact_spectral_gap(&certificate.even_ground, &certificate.even_first_excited)
                .map_err(anyhow::Error::from)?;
        if recomputed_gap != certificate.even_simplicity_gap {
            bail!("CCM even-sector simplicity gap differs from exact recomputation");
        }
        let ordering_gap_lower = parse(&certificate.odd_ground.lower)
            .map_err(anyhow::Error::from)?
            - parse(&certificate.even_ground.upper).map_err(anyhow::Error::from)?;
        let ordering_gap_upper = parse(&certificate.odd_ground.upper)
            .map_err(anyhow::Error::from)?
            - parse(&certificate.even_ground.lower).map_err(anyhow::Error::from)?;
        let parity = if ordering_gap_lower > 0 {
            "even"
        } else if ordering_gap_upper < 0 {
            "odd"
        } else {
            "unresolved"
        };
        if rational_record(&ordering_gap_lower) != certificate.even_odd_ordering_gap_lower
            || rational_record(&ordering_gap_upper) != certificate.even_odd_ordering_gap_upper
            || certificate.certified_finite_ground_parity != parity
            || certificate.certifies_finite_ground_state_simple != (parity != "unresolved")
        {
            bail!("CCM even-versus-odd ordering outcome differs from exact separation replay");
        }
        let positive = certificate.full_matrix_inertia_certificate.positive == full_dimension
            && certificate.full_matrix_inertia_certificate.negative == 0
            && certificate
                .full_matrix_inertia_certificate
                .zero_or_unresolved
                == 0;
        if certificate.certifies_finite_matrix_positive_definite != positive {
            bail!("CCM finite positive-definite flag differs from full-matrix inertia replay");
        }
        checks.push(
            "raw full cutoff-free Tau inertia replays without a centrosymmetry premise".to_owned(),
        );
        checks.push(
            "conditional parity projection derives from the recorded reflection-orbit intersections"
                .to_owned(),
        );
        checks.push(format!(
            "finite ground-state parity outcome ({parity}) and simplicity replayed"
        ));
        Ok(checks)
    };

    match verify() {
        Ok(checks) => VerificationReport {
            valid: true,
            checks,
            warnings: vec![
                format!(
                    "parity, sector ordering, and sector simplicity are conditional on this premise: {PARITY_INVARIANCE_PREMISE}; full-matrix positivity is not"
                ),
                "certificate scope is one finite cutoff-free CCM matrix; it does not establish continuum parity, asymptotic convergence, or a global theorem"
                    .to_owned(),
            ],
            errors: Vec::new(),
        },
        Err(error) => invalid_report(error.to_string()),
    }
}

#[cfg(feature = "arb")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_sector_gap_certificate_via_cache(
    params: &CcmParams,
    precision_bits: u32,
    even_ground_guide: &Float,
    even_first_excited_guide: &Float,
    odd_ground_guide: &Float,
    even_manifest: &xc_cache::ArtifactManifest,
    odd_manifest: &xc_cache::ArtifactManifest,
    gap_manifest: &xc_cache::ArtifactManifest,
    options: CcmSectorGapCertificationOptions,
    cache: &xc_cache::ArtifactCacheContext<'_>,
) -> Result<PortableCcmSectorGapCertificate> {
    use std::collections::BTreeMap;
    use xc_cache::{
        resolve_or_compute_json_artifact_with_assessment, ArtifactAssuranceState,
        ArtifactExecutionCacheRequest, ArtifactProductionAssessment, CacheError, CacheQuality,
        DependencyRef, ToolkitVersion,
    };

    options.validate()?;
    if !params.lambda_sq.is_integer || params.lambda_sq_int() <= 1 || params.n_modes == 0 {
        bail!("CCM sector-gap certification requires integer lambda_squared > 1 and N >= 1");
    }
    if precision_bits < 64 {
        bail!("CCM sector-gap certification requires at least 64 bits");
    }
    if even_manifest.key.kind != "ccm_sector_spectrum"
        || odd_manifest.key.kind != "ccm_sector_spectrum"
        || gap_manifest.key.kind != "ccm_sector_gap"
    {
        bail!("CCM sector-gap certification requires retained even, odd, and gap guide manifests");
    }
    let lambda_squared = params.lambda_sq_int().to_string();
    let scalar_backend = format!("system-flint-arb-{}", super::arb_bridge::backend_version());
    let semantic_key = SemanticKeyEnvelope {
        schema_version: 1,
        artifact_kind: "ccm_sector_gap_certificate".to_owned(),
        mathematical_semantics_version:
            "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2".to_owned(),
        resolved_mathematical_parameters: serde_json::json!({
            "lambda_squared": lambda_squared,
            "n_modes": params.n_modes,
            "precision_bits": precision_bits,
            "geometric_terms": 32,
            "relative_enclosure_bits": options.relative_enclosure_bits,
            "maximum_bracket_expansions": options.maximum_bracket_expansions,
            "even_spectrum_content_digest": even_manifest.content_digest.0,
            "odd_spectrum_content_digest": odd_manifest.content_digest.0,
            "sector_gap_content_digest": gap_manifest.content_digest.0,
            "interval_matrix_semantics": INTERVAL_MATRIX_SEMANTICS,
            "parity_invariance_premise": PARITY_INVARIANCE_PREMISE,
            "scalar_backend": scalar_backend,
        }),
        normalization: Some("orthonormal_reflection_parity_sectors".to_owned()),
        target: Some(
            "finite_ccm_conditional_parity_simplicity_and_unconditional_full_inertia"
                .to_owned(),
        ),
        subspace: Some("even_vs_odd".to_owned()),
        source_data_identities: BTreeMap::from([
            (
                "ccm_even_sector_spectrum".to_owned(),
                even_manifest.content_digest.clone(),
            ),
            (
                "ccm_odd_sector_spectrum".to_owned(),
                odd_manifest.content_digest.clone(),
            ),
            (
                "ccm_sector_gap".to_owned(),
                gap_manifest.content_digest.clone(),
            ),
        ]),
        algorithm_semantics: Some(
            "cutoff_free_interval_assembly_full_matrix_ldlt_then_conditional_symmetry_intersection_parity_projection_exact_shifted_inertia_v2".to_owned(),
        ),
    };
    let semantic_digest = semantic_key.digest()?;
    let logical_key = format!(
        "ccm/sector-gap-certificate/{}/{}/{}/{}",
        params.lambda_sq_int(),
        params.n_modes,
        precision_bits,
        semantic_digest.0
    );
    let request = ArtifactExecutionCacheRequest {
        operation: "ccm.sector_gap_certificate.resolve_or_compute",
        semantic_key: &semantic_key,
        logical_key: &logical_key,
        resolver: cache.resolver,
        reference_resolver: cache.reference_resolver,
        acceptance: cache.acceptance,
        ordered_overlays: cache.ordered_overlays.clone(),
        mode: cache.mode,
        write_on_miss: cache.write_on_miss,
        write_visibility: cache.write_visibility,
        produced_quality: CacheQuality::Certified,
        producer_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION"))?,
        minimum_reader_version: ToolkitVersion::parse("0.14.1")?,
        maximum_reader_version: None,
        tags: BTreeMap::from([
            ("domain".to_owned(), "ccm".to_owned()),
            ("artifact".to_owned(), "sector_gap_certificate".to_owned()),
            (
                "certification_scope".to_owned(),
                "finite_cutoff_free_matrix".to_owned(),
            ),
        ]),
        provenance_digest: Some(gap_manifest.content_digest.clone()),
        production_sink: cache.production_sink,
    };
    let resolved = resolve_or_compute_json_artifact_with_assessment(
        &request,
        || {
            let config = super::cutoff_free::CutoffFreeConfig::new(
                params.lambda_sq_int(),
                params.n_modes,
                precision_bits,
            );
            let matrix = super::cutoff_free::assemble(&config).map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "cutoff-free CCM sector-gap assembly failed: {error:#}"
                ))
            })?;
            let component_evidence_digest =
                matrix.component_evidence_digest().map_err(|error| {
                    CacheError::InvalidManifest(format!(
                        "cutoff-free CCM component evidence failed: {error:#}"
                    ))
                })?;
            let full_matrix_inertia_certificate =
                matrix.portable_inertia_certificate().map_err(|error| {
                    CacheError::InvalidManifest(format!(
                        "cutoff-free CCM full-matrix inertia certification failed: {error:#}"
                    ))
                })?;
            let artifact = build_certificate_from_tau(
                params.lambda_sq_int().to_string(),
                params.lambda_sq_int(),
                params.n_modes,
                precision_bits,
                matrix.config.geometric_terms,
                matrix.scalar_backend.clone(),
                component_evidence_digest,
                full_matrix_inertia_certificate,
                even_ground_guide,
                even_first_excited_guide,
                odd_ground_guide,
                even_manifest.content_digest.clone(),
                odd_manifest.content_digest.clone(),
                gap_manifest.content_digest.clone(),
                options,
            )
            .map_err(|error| {
                CacheError::InvalidManifest(format!(
                    "CCM sector-gap certification failed: {error:#}"
                ))
            })?;
            let evidence_bytes = serde_json::to_vec(&artifact)?;
            let evidence_digest = if let Some(sink) = cache.production_sink {
                sink.record_evidence("ccm-sector-gap-certificate", &evidence_bytes)?
            } else {
                ContentDigest::sha256(&evidence_bytes)
            };
            let mut dependencies = [even_manifest, odd_manifest, gap_manifest]
                .into_iter()
                .map(|manifest| DependencyRef {
                    key: manifest.key.clone(),
                    content_digest: manifest.content_digest.clone(),
                    required_quality: CacheQuality::Validated,
                })
                .collect::<Vec<_>>();
            dependencies.sort_by(|left, right| {
                (
                    left.key.kind.as_str(),
                    left.key.logical_key.as_str(),
                    left.key.parameters_digest.0.as_str(),
                    left.content_digest.0.as_str(),
                )
                    .cmp(&(
                        right.key.kind.as_str(),
                        right.key.logical_key.as_str(),
                        right.key.parameters_digest.0.as_str(),
                        right.content_digest.0.as_str(),
                    ))
            });
            Ok((
                artifact,
                dependencies,
                ArtifactProductionAssessment {
                    achieved_assurance: ArtifactAssuranceState::Certified,
                    evidence_digests: vec![evidence_digest],
                },
            ))
        },
        |artifact| {
            if artifact.lambda_squared != params.lambda_sq_int().to_string()
                || artifact.integer_cutoff_c != params.lambda_sq_int()
                || artifact.n_modes != params.n_modes
                || artifact.precision_bits != precision_bits
                || artifact.geometric_terms != 32
                || artifact.scalar_backend != scalar_backend
                || artifact.relative_enclosure_bits != options.relative_enclosure_bits
                || artifact.maximum_bracket_expansions != options.maximum_bracket_expansions
                || artifact.guide_even_spectrum_content_digest != even_manifest.content_digest
                || artifact.guide_odd_spectrum_content_digest != odd_manifest.content_digest
                || artifact.guide_sector_gap_content_digest != gap_manifest.content_digest
                || artifact.guide_even_ground != even_ground_guide.to_string()
                || artifact.guide_even_first_excited != even_first_excited_guide.to_string()
                || artifact.guide_odd_ground != odd_ground_guide.to_string()
            {
                return Err(CacheError::InvalidManifest(
                    "CCM sector-gap certificate does not match its semantic identity".to_owned(),
                ));
            }
            let report = verify_portable_ccm_sector_gap_certificate(artifact);
            if !report.valid {
                return Err(CacheError::InvalidManifest(format!(
                    "CCM sector-gap certificate failed exact replay: {}",
                    report.errors.join("; ")
                )));
            }
            Ok(())
        },
    )?;
    Ok(resolved.value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> ContentDigest {
        ContentDigest::sha256(label.as_bytes())
    }

    fn synthetic_certificate() -> PortableCcmSectorGapCertificate {
        let precision_bits = 128;
        let diagonal = [5, 2, 1, 2, 5];
        let mut tau = vec![RationalInterval::point(Rational::from(0)); 25];
        for (index, value) in diagonal.into_iter().enumerate() {
            tau[index * 5 + index] = RationalInterval::point(Rational::from(value));
        }
        let component_digest = digest("components");
        let full_matrix_inertia_certificate =
            xc_certify::exact::build_portable_interval_inertia_certificate(
                &tau,
                5,
                precision_bits,
                "synthetic-exact",
                component_digest.clone(),
                std::collections::BTreeMap::from([
                    ("integer_cutoff_c".to_owned(), "5".to_owned()),
                    ("modes".to_owned(), "2".to_owned()),
                    ("geometric_terms".to_owned(), "32".to_owned()),
                ]),
                vec!["synthetic exact fixture".to_owned()],
            )
            .unwrap();
        build_certificate_from_tau(
            "5".to_owned(),
            5,
            2,
            precision_bits,
            32,
            "synthetic-exact".to_owned(),
            component_digest,
            full_matrix_inertia_certificate,
            &Float::with_val(precision_bits, 1),
            &Float::with_val(precision_bits, 2),
            &Float::with_val(precision_bits, 2),
            digest("even"),
            digest("odd"),
            digest("gap"),
            CcmSectorGapCertificationOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn exact_sector_certificate_replays_and_proves_the_finite_claims() {
        let certificate = synthetic_certificate();
        let report = verify_portable_ccm_sector_gap_certificate(&certificate);
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(certificate.certified_finite_ground_parity, "even");
        assert!(certificate.certifies_finite_ground_state_simple);
        assert!(certificate.certifies_finite_matrix_positive_definite);
        assert!(parse(&certificate.even_odd_ordering_gap_lower).unwrap() > 0);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("conditional") && warning.contains("centrosymmetric")));
    }

    #[test]
    fn exact_sector_certificate_rejects_tau_or_claim_tampering() {
        let mut tau_tampered = synthetic_certificate();
        tau_tampered.cutoff_free_tau[0].lower.numerator = "6".to_owned();
        assert!(!verify_portable_ccm_sector_gap_certificate(&tau_tampered).valid);

        let mut claim_tampered = synthetic_certificate();
        claim_tampered.certifies_finite_matrix_positive_definite = false;
        assert!(!verify_portable_ccm_sector_gap_certificate(&claim_tampered).valid);

        let mut parity_tampered = synthetic_certificate();
        parity_tampered.certified_finite_ground_parity = "odd".to_owned();
        assert!(!verify_portable_ccm_sector_gap_certificate(&parity_tampered).valid);

        let mut premise_tampered = synthetic_certificate();
        premise_tampered.parity_invariance_premise = "none".to_owned();
        assert!(!verify_portable_ccm_sector_gap_certificate(&premise_tampered).valid);
    }

    #[cfg(feature = "arb")]
    #[test]
    fn cutoff_free_small_matrix_produces_a_replayable_sector_certificate() {
        let precision_bits = 256;
        let matrix = super::super::cutoff_free::assemble(
            &super::super::cutoff_free::CutoffFreeConfig::new(5, 2, precision_bits),
        )
        .unwrap();
        let certificate = build_certificate_from_tau(
            matrix.config.integer_cutoff_c.to_string(),
            matrix.config.integer_cutoff_c,
            matrix.config.modes,
            precision_bits,
            matrix.config.geometric_terms,
            matrix.scalar_backend.clone(),
            matrix.component_evidence_digest().unwrap(),
            matrix.portable_inertia_certificate().unwrap(),
            &Float::with_val(precision_bits, 1),
            &Float::with_val(precision_bits, 2),
            &Float::with_val(precision_bits, 3),
            digest("real-even"),
            digest("real-odd"),
            digest("real-gap"),
            CcmSectorGapCertificationOptions::default(),
        )
        .unwrap();
        let report = verify_portable_ccm_sector_gap_certificate(&certificate);
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(certificate.certified_finite_ground_parity, "odd");
    }

    #[test]
    fn certification_options_reject_brackets_that_can_cross_zero_by_policy() {
        assert!(CcmSectorGapCertificationOptions {
            relative_enclosure_bits: 8,
            maximum_bracket_expansions: 8,
        }
        .validate()
        .is_err());
        assert!(CcmSectorGapCertificationOptions::default()
            .validate()
            .is_ok());
    }
}
