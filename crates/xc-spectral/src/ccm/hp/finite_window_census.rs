//! Replayable exact finite-window counts, including scientifically meaningful zero counts.
//! This does not count carrier zeros, roots at stored poles, or roots outside the window.
use super::*;
use rug::Rational;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedFiniteWindowCensus {
    pub schema_version: u32,
    pub lambda_squared: String,
    pub n_modes: usize,
    pub precision_bits: u32,
    pub eigenpair_source: DependencyRef,
    pub secular_source: DependencyRef,
    pub source_weights_digest: ContentDigest,
    pub source_poles_digest: ContentDigest,
    pub numerator_digest: ContentDigest,
    pub numerator_degree: usize,
    pub interior_polynomial_degree: usize,
    pub lower_exact_rational: String,
    pub upper_exact_rational: String,
    pub excluded_lower_multiplicity: usize,
    pub excluded_upper_multiplicity: usize,
    pub excluded_interior_pole_factors: Vec<ExcludedPoleFactor>,
    pub distinct_movable_roots: usize,
    pub square_free_interior: bool,
    pub arithmetic: String,
    pub scope: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedPoleFactor {
    pub pole_exact_rational: String,
    pub multiplicity: usize,
}

fn polynomial_digest(coefficients: &[Rational]) -> Result<ContentDigest> {
    Ok(ContentDigest::sha256(&serde_json::to_vec(
        &coefficients
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    )?))
}

fn deflate_endpoint(coefficients: &mut Vec<Rational>, point: &Rational) -> Result<usize> {
    let mut multiplicity = 0;
    while coefficients.len() > 1
        && xc_numerics::interval::evaluate_real_polynomial_exact(coefficients, point)? == 0
    {
        let mut quotient = vec![Rational::from(0); coefficients.len() - 1];
        *quotient.last_mut().expect("nonconstant polynomial") =
            coefficients.last().unwrap().clone();
        for index in (1..quotient.len()).rev() {
            quotient[index - 1] = coefficients[index].clone() + quotient[index].clone() * point;
        }
        if coefficients[0].clone() + quotient[0].clone() * point != 0 {
            bail!("exact endpoint deflation left a remainder");
        }
        *coefficients = quotient;
        multiplicity += 1;
    }
    Ok(multiplicity)
}

fn exact_open_count(
    original: &[Rational],
    lower: &Rational,
    upper: &Rational,
    excluded_poles: &[Rational],
    work: u32,
) -> Result<(usize, usize, usize, usize, Vec<ExcludedPoleFactor>)> {
    if original.is_empty() || original.iter().all(|coefficient| coefficient == &0) || lower >= upper
    {
        bail!("finite-window count requires a nonzero polynomial and ordered endpoints");
    }
    let mut coefficients = original.to_vec();
    while coefficients.len() > 1 && coefficients.last().unwrap() == &0 {
        coefficients.pop();
    }
    let lower_multiplicity = deflate_endpoint(&mut coefficients, lower)?;
    let upper_multiplicity = deflate_endpoint(&mut coefficients, upper)?;
    let mut pole_factors = Vec::new();
    for pole in excluded_poles
        .iter()
        .filter(|pole| *pole > lower && *pole < upper)
    {
        let multiplicity = deflate_endpoint(&mut coefficients, pole)?;
        if multiplicity > 0 {
            pole_factors.push(ExcludedPoleFactor {
                pole_exact_rational: pole.to_string(),
                multiplicity,
            });
        }
    }
    let degree = coefficients.len() - 1;
    if degree == 0 {
        return Ok((
            0,
            degree,
            lower_multiplicity,
            upper_multiplicity,
            pole_factors,
        ));
    }
    #[cfg(feature = "arb")]
    let (count, square_free) = {
        let (roots, square_free) = crate::ccm::arb_bridge::rational_polynomial_real_roots(
            &coefficients,
            lower,
            upper,
            work,
        )?;
        (roots.len(), square_free)
    };
    #[cfg(not(feature = "arb"))]
    let (count, square_free) = {
        let _ = work;
        // This portable rational replay is deliberately limited. Larger counts
        // require FLINT, rather than an unannounced approximate fallback.
        if degree > 128 {
            bail!("large finite-window count requires the Arb feature");
        }
        let count = xc_numerics::interval::exact_sturm_root_count(
            &coefficients,
            lower.clone(),
            upper.clone(),
        )?;
        (count.distinct_real_roots, count.square_free)
    };
    if !square_free {
        bail!("finite-window count has unresolved repeated interior factors");
    }
    Ok((
        count,
        degree,
        lower_multiplicity,
        upper_multiplicity,
        pole_factors,
    ))
}

/// Count an authenticated natural-parity CCM source in its open stored pole span.
/// The returned count uses exact rational polynomial arithmetic. Source assembly
/// uncertainty and every outside-window or carrier-zero question remain separate.
pub fn count_signed_finite_window_via_cache(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    primary: &HighPrecResult,
    eigenpair: &ArtifactManifest,
    secular: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<SignedFiniteWindowCensus> {
    let weights = &primary.xi;
    if cfg.effective_parity_policy() != CcmParityPolicy::Natural
        || !params.lambda_sq.is_integer
        || params.lambda_sq.value_u64 <= 1
        || params.n_modes == 0
        || params.matrix_size() != weights.len()
        || primary.precision_bits != cfg.precision_bits
    {
        bail!("signed finite-window census requires an integer natural CCM source");
    }
    for manifest in [eigenpair, secular] {
        manifest.validate()?;
        if !manifest.immutable || !manifest.quality.satisfies(CacheQuality::Validated) {
            bail!("finite-window census requires immutable validated sources");
        }
    }
    if eigenpair.key.kind != "ccm_weil_eigenpair"
        || secular.key.kind != "ccm_secular_source"
        || !xc_cache::manifest_depends_on(secular, eigenpair)?
    {
        bail!("finite-window census source identities do not join");
    }
    let resolver = cache
        .resolver
        .context("finite-window count requires its retained source resolver")?;
    let policy = cache
        .acceptance
        .context("finite-window count requires its retained source policy")?;
    let eigen_bytes =
        resolver.read_exact_payload(&eigenpair.key, &eigenpair.content_digest, policy)?;
    let eigen: PortableWeilEigenpair = serde_json::from_slice(&eigen_bytes)?;
    let secular_bytes =
        resolver.read_exact_payload(&secular.key, &secular.content_digest, policy)?;
    let source: PortableSecularSource = serde_json::from_slice(&secular_bytes)?;
    let p = cfg.precision_bits;
    let cutoff = lambda_squared_cache_identity(params);
    if eigen.lambda_squared != cutoff
        || eigen.n_modes != params.n_modes
        || eigen.precision_bits != p
        || !payload_parity_matches(
            eigen.force_even,
            eigen.parity_policy,
            CcmParityPolicy::Natural,
        )
        || source.schema_version != 2
        || source.lambda_squared != cutoff
        || source.n_modes != params.n_modes
        || source.precision_bits != p
        || !payload_parity_matches(
            source.force_even,
            source.parity_policy,
            CcmParityPolicy::Natural,
        )
        || source.eigenpair_content_digest != eigenpair.content_digest.0
        || source.eigenpair_semantic_digest != eigenpair.key.parameters_digest.0
        || source.normalization != "sum_xi_equals_sqrt_log_lambda_squared"
    {
        bail!("finite-window count payload/configuration binding differs from retained source");
    }
    let retained_weights = eigen
        .eigenvector
        .iter()
        .map(|value| parse_hp_scalar(value, p))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if retained_weights.len() != weights.len()
        || retained_weights
            .iter()
            .zip(weights)
            .any(|(stored, supplied)| stored != supplied || supplied.prec() != p)
    {
        bail!("finite-window count weights differ from authenticated retained eigenstate");
    }
    if parse_hp_scalar(&eigen.eigenvalue, p)? != primary.weil_min_eigenvalue
        || eigen.stored_state_resolution != primary.stored_state_resolution
        || eigen.ground_index != primary.ground_index_admission
    {
        bail!("finite-window count ground/eigenvalue evidence differs from authenticated retained eigenstate");
    }
    let length = log_lambda_sq_hp(params, p)?;
    let spacing = secular_spacing(&length, p)?;
    let poles = secular_poles(&spacing, params.n_modes, p);
    let upper = (spacing.clone() * params.n_modes)
        .to_rational()
        .context("nonfinite pole window")?;
    let lower = -upper.clone();
    let workspace = super::verified_discovery::DiscoveryWorkspace::estimate(
        &poles, weights, &lower, &upper, p,
    )?
    .admit()?;
    let coefficients = super::verified_discovery::numerator(&poles, weights, &workspace)?;
    let _stage = crate::ccm::capture_runtime::Stage::new(format!(
        "exact signed finite-window count replay (degree {}, {p} bits)",
        coefficients.len() - 1
    ));
    let rational_poles = poles
        .iter()
        .map(|pole| pole.to_rational().expect("finite stored pole"))
        .collect::<Vec<_>>();
    let (count, degree, lower_multiplicity, upper_multiplicity, pole_factors) = exact_open_count(
        &coefficients,
        &lower,
        &upper,
        &rational_poles,
        p.saturating_add(32),
    )?;
    // This API is deliberately an absence witness. Nonempty inventories still
    // require the ordinary discovery/refinement and retained-window path.
    if count != 0 {
        bail!("empty-window witness refused: exact numerator contains {count} interior roots");
    }
    let dependency = |manifest: &ArtifactManifest| DependencyRef {
        key: manifest.key.clone(),
        content_digest: manifest.content_digest.clone(),
        required_quality: CacheQuality::Validated,
    };
    Ok(SignedFiniteWindowCensus {
        schema_version: 1, lambda_squared: cutoff, n_modes: params.n_modes, precision_bits: p,
        eigenpair_source: dependency(eigenpair), secular_source: dependency(secular),
        source_weights_digest: crate::ccm::certified_roots::production_ccm_source_weights_digest(weights, p)?,
        source_poles_digest: crate::ccm::certified_roots::production_ccm_source_weights_digest(&poles, p)?,
        numerator_digest: polynomial_digest(&coefficients)?, numerator_degree: coefficients.len() - 1,
        interior_polynomial_degree: degree, lower_exact_rational: lower.to_string(), upper_exact_rational: upper.to_string(),
        excluded_lower_multiplicity: lower_multiplicity, excluded_upper_multiplicity: upper_multiplicity,
        excluded_interior_pole_factors: pole_factors,
        distinct_movable_roots: count, square_free_interior: true,
        arithmetic: "exact_dyadic_polynomial_certified_open_window_census_v1".into(),
        scope: "Zero distinct non-pole movable roots in the open stored pole span only. Endpoint and stored-pole factors are excluded exactly. No carrier, outside-window, analytic assembly, continuum, or RH conclusion.".into(),
    })
}

/// Reauthenticate both retained payloads and repeat the exact count. Every
/// witness field, including endpoint exclusion and source identity, must agree.
pub fn verify_signed_finite_window_census_via_cache(
    expected: &SignedFiniteWindowCensus,
    params: &CcmParams,
    cfg: &HighPrecConfig,
    primary: &HighPrecResult,
    eigenpair: &ArtifactManifest,
    secular: &ArtifactManifest,
    cache: &ArtifactCacheContext<'_>,
) -> Result<()> {
    let replay =
        count_signed_finite_window_via_cache(params, cfg, primary, eigenpair, secular, cache)?;
    if expected != &replay {
        bail!("finite-window census differs from authenticated exact count replay");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_count_requires_exact_nonzero_polynomial_and_open_boundaries() {
        let q = |values: &[i32]| {
            values
                .iter()
                .map(|v| Rational::from(*v))
                .collect::<Vec<_>>()
        };
        let lo = Rational::from(-1);
        let hi = Rational::from(1);
        assert_eq!(
            exact_open_count(&q(&[1, 0, 1]), &lo, &hi, &[], 256).unwrap(),
            (0, 2, 0, 0, vec![])
        );
        assert_eq!(
            exact_open_count(&q(&[-1, 0, 1]), &lo, &hi, &[], 256).unwrap(),
            (0, 0, 1, 1, vec![])
        );
        assert_eq!(exact_open_count(&q(&[7]), &lo, &hi, &[], 256).unwrap().0, 0);
        assert_eq!(
            exact_open_count(&q(&[0, 1]), &lo, &hi, &[], 256).unwrap().0,
            1
        );
        let at_pole = exact_open_count(&q(&[0, 1]), &lo, &hi, &[Rational::from(0)], 256).unwrap();
        assert_eq!(at_pole.0, 0);
        assert_eq!(
            at_pole.4,
            vec![ExcludedPoleFactor {
                pole_exact_rational: "0".into(),
                multiplicity: 1
            }]
        );
        assert!(exact_open_count(&q(&[0, 0]), &lo, &hi, &[], 256).is_err());
        assert!(exact_open_count(&q(&[0, 0, 1]), &lo, &hi, &[], 256).is_err());
        assert!(exact_open_count(&q(&[1, 0, 1]), &hi, &lo, &[], 256).is_err());
        assert_ne!(
            polynomial_digest(&q(&[1, 0, 1])).unwrap(),
            polynomial_digest(&q(&[-1, 0, 1])).unwrap()
        );
    }
}
