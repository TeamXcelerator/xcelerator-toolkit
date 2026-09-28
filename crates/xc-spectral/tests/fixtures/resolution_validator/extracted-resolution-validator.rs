// Mechanically copied current production bodies; source equality is tested.
use rug::Float;
use xc_spectral::distance::WeightedIntegrationRule;
use xc_spectral::distance::hp::PortableDistanceResolutionEvidence;
const GUARD_BITS:u32=64;
const RESOLUTION_EVIDENCE_THRESHOLD_DECADES:[u32;3]=[15,30,45];
const RESOLUTION_EVIDENCE_REFINEMENT_FACTOR:usize=2;
const RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER:usize=4;
const RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE:&str="1e-8";
    fn alpha_identity(alpha: &Float, prec: u32) -> String {
        Float::with_val(prec.saturating_add(GUARD_BITS), alpha).to_string()
    }

    fn invalid_retained_payload(detail: impl Into<String>) -> xc_cache::CacheError {
        xc_cache::CacheError::InvalidManifest(detail.into())
    }

    fn parse_retained_float(
        text: &str,
        precision_bits: u32,
        field: &str,
    ) -> std::result::Result<Float, xc_cache::CacheError> {
        if !(32..=1_000_000).contains(&precision_bits) {
            return Err(invalid_retained_payload(
                "retained scalar precision must be in 32..=1000000 bits",
            ));
        }
        let parsed = Float::parse(text).map_err(|error| {
            invalid_retained_payload(format!("invalid retained {field} decimal: {error}"))
        })?;
        let value = Float::with_val(precision_bits, parsed);
        if !value.is_finite() {
            return Err(invalid_retained_payload(format!(
                "retained {field} must be finite"
            )));
        }
        if value.is_zero() {
            let exact = xc_core::DecimalLiteral::new(text)
                .and_then(|literal| literal.canonical())
                .map_err(|_| {
                    invalid_retained_payload(format!("invalid retained {field} decimal"))
                })?;
            if exact.as_str() != "0" {
                return Err(invalid_retained_payload(format!(
                    "retained {field} underflows the working exponent range"
                )));
            }
        }
        Ok(value)
    }

    fn absolute_and_relative_difference(
        coarser: &Float,
        finer: &Float,
        precision_bits: u32,
    ) -> (Float, Float) {
        let working = precision_bits.saturating_add(GUARD_BITS);
        let absolute = Float::with_val(working, coarser - finer).abs();
        let denominator = Float::with_val(working, finer).abs();
        let relative = if denominator == 0u32 {
            absolute.clone()
        } else {
            Float::with_val(working, &absolute / denominator)
        };
        (
            Float::with_val(precision_bits, absolute),
            Float::with_val(precision_bits, relative),
        )
    }

    fn resolution_tolerance_met(coarser: &Float, finer: &Float, p: u32) -> bool {
        use rug::float::Round;
        let working = p.saturating_add(GUARD_BITS);
        let coarser = &Float::with_val(p, coarser);
        let finer = &Float::with_val(p, finer);
        let (larger, smaller) = if coarser >= finer {
            (coarser, finer)
        } else {
            (finer, coarser)
        };
        let absolute_upper = Float::with_val_round(working, larger - smaller, Round::Up).0;
        let scaled_upper =
            Float::with_val_round(working, &absolute_upper * 100_000_000u32, Round::Up).0;
        if !scaled_upper.is_finite() {
            return false;
        }
        if finer.is_zero() {
            scaled_upper <= 1
        } else {
            scaled_upper <= *finer
        }
    }

    pub(crate) fn validate_portable_distance_resolution_evidence(
        artifact: &PortableDistanceResolutionEvidence,
        target_definition_digest: &str,
        lambda_squared: &str,
        n_modes: usize,
        precision_bits: u32,
        alpha: &Float,
        rules: &[WeightedIntegrationRule],
    ) -> std::result::Result<(), xc_cache::CacheError> {
        let expected_coefficients = n_modes.checked_add(1).ok_or_else(|| {
            invalid_retained_payload("CCM resolution-evidence coefficient count overflows usize")
        })?;
        let uniform_rules = rules
            .iter()
            .filter(|rule| matches!(rule, WeightedIntegrationRule::UniformGrid { .. }))
            .collect::<Vec<_>>();
        if uniform_rules.is_empty()
            || artifact.schema_version != 2
            || artifact.target_definition_digest != target_definition_digest
            || artifact.lambda_squared != lambda_squared
            || artifact.n_modes != n_modes
            || artifact.precision_bits != precision_bits
            || artifact.alpha != alpha_identity(alpha, precision_bits)
            || artifact.normalization != "f(1)=1"
            || artifact.coefficient_count != expected_coefficients
            || artifact.coefficient_tail.len() != RESOLUTION_EVIDENCE_THRESHOLD_DECADES.len()
            || artifact.refinement_factor != RESOLUTION_EVIDENCE_REFINEMENT_FACTOR
            || artifact.maximum_refinement_multiplier != RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER
            || artifact.relative_tolerance != RESOLUTION_EVIDENCE_RELATIVE_TOLERANCE
            || artifact.relative_difference_denominator != "absolute_finer_distance"
            || artifact.zero_denominator_fallback != "absolute_difference"
            || artifact.refinements.len() != uniform_rules.len()
        {
            return Err(invalid_retained_payload(
                "CCM distance resolution evidence does not match its request",
            ));
        }

        for (tail, threshold_decades) in artifact
            .coefficient_tail
            .iter()
            .zip(RESOLUTION_EVIDENCE_THRESHOLD_DECADES)
        {
            if tail.threshold != format!("1e-{threshold_decades}")
                || tail
                    .effective_bandwidth
                    .is_some_and(|bandwidth| bandwidth > n_modes)
            {
                return Err(invalid_retained_payload(
                    "CCM coefficient-tail evidence does not match its policy",
                ));
            }
            for (value, field) in [
                (&tail.discarded_one_sided_l1, "discarded one-sided L1"),
                (
                    &tail.discarded_cosine_pointwise_bound,
                    "discarded cosine pointwise bound",
                ),
                (&tail.discarded_cosine_l2, "discarded cosine L2"),
            ] {
                if parse_retained_float(value, precision_bits, field)? < 0u32 {
                    return Err(invalid_retained_payload(format!(
                        "retained {field} must be nonnegative"
                    )));
                }
            }
        }

        for (entry, rule) in artifact.refinements.iter().zip(uniform_rules) {
            let twice_resolution = rule
                .resolution()
                .checked_mul(RESOLUTION_EVIDENCE_REFINEMENT_FACTOR)
                .ok_or_else(|| invalid_retained_payload("2Q resolution overflows usize"))?;
            let four_resolution = rule
                .resolution()
                .checked_mul(RESOLUTION_EVIDENCE_MAXIMUM_MULTIPLIER)
                .ok_or_else(|| invalid_retained_payload("4Q resolution overflows usize"))?;
            if entry.rule_family != rule.family()
                || entry.quadrature_rule != rule.rule()
                || entry.grid_variable != rule.variable().as_str()
                || entry.base_resolution != rule.resolution()
                || entry.twice_resolution != twice_resolution
            {
                return Err(invalid_retained_payload(
                    "CCM resolution-evidence rule does not match its request",
                ));
            }
            let mut values = vec![
                (&entry.base_distance, "base distance"),
                (&entry.twice_distance, "twice-refined distance"),
                (
                    &entry.q_to_2q_absolute_difference,
                    "Q-to-2Q absolute difference",
                ),
                (
                    &entry.q_to_2q_relative_difference,
                    "Q-to-2Q relative difference",
                ),
                (
                    &entry.final_absolute_difference,
                    "final absolute difference",
                ),
                (
                    &entry.final_relative_difference,
                    "final relative difference",
                ),
            ];
            if let Some(value) = &entry.four_times_distance {
                values.push((value, "four-times-refined distance"));
            }
            for (value, field) in values {
                if parse_retained_float(value, precision_bits, field)? < 0u32 {
                    return Err(invalid_retained_payload(format!(
                        "retained {field} must be nonnegative"
                    )));
                }
            }

            // Re-derive discrepancies from retained measurements. A self-consistent
            // verdict over forged difference fields is not numerical evidence.
            let check_difference = |left: &str,
                                    right: &str,
                                    absolute: &str,
                                    relative: &str|
             -> std::result::Result<bool, xc_cache::CacheError> {
                let left = parse_retained_float(left, precision_bits, "refinement distance")?;
                let right = parse_retained_float(right, precision_bits, "refinement distance")?;
                let (expected_absolute, expected_relative) =
                    absolute_and_relative_difference(&left, &right, precision_bits);
                let observed_absolute =
                    parse_retained_float(absolute, precision_bits, "absolute difference")?;
                let observed_relative =
                    parse_retained_float(relative, precision_bits, "relative difference")?;
                let mut epsilon = Float::with_val(precision_bits, 1);
                epsilon >>= precision_bits.saturating_sub(16);
                let mut scale = left.clone().abs().max(&right.clone().abs());
                if scale.is_zero() {
                    scale = Float::with_val(precision_bits, 1);
                }
                let absolute_tolerance = Float::with_val(precision_bits, &epsilon * scale);
                let mut relative_tolerance = expected_relative.clone().abs();
                relative_tolerance += 1;
                relative_tolerance *= epsilon;
                if Float::with_val(precision_bits, observed_absolute - expected_absolute).abs()
                    > absolute_tolerance
                    || Float::with_val(precision_bits, observed_relative - expected_relative).abs()
                        > relative_tolerance
                {
                    return Err(invalid_retained_payload(
                        "refinement discrepancies do not follow from retained distances",
                    ));
                }
                Ok(resolution_tolerance_met(&left, &right, precision_bits))
            };
            let q_to_2q_met = check_difference(
                &entry.base_distance,
                &entry.twice_distance,
                &entry.q_to_2q_absolute_difference,
                &entry.q_to_2q_relative_difference,
            )?;
            let final_met = if let Some(four) = &entry.four_times_distance {
                check_difference(
                    &entry.twice_distance,
                    four,
                    &entry.final_absolute_difference,
                    &entry.final_relative_difference,
                )?
            } else {
                q_to_2q_met
            };

            let continued = !q_to_2q_met;
            match (
                continued,
                entry.four_times_resolution,
                entry.four_times_distance.as_ref(),
            ) {
                (false, None, None) => {
                    if entry.final_resolution != twice_resolution
                        || entry.final_absolute_difference != entry.q_to_2q_absolute_difference
                        || entry.final_relative_difference != entry.q_to_2q_relative_difference
                    {
                        return Err(invalid_retained_payload(
                            "CCM Q/2Q resolution evidence has inconsistent final fields",
                        ));
                    }
                }
                (true, Some(resolution), Some(_)) if resolution == four_resolution => {
                    if entry.final_resolution != four_resolution {
                        return Err(invalid_retained_payload(
                            "CCM Q/2Q/4Q resolution evidence has the wrong final resolution",
                        ));
                    }
                }
                _ => {
                    return Err(invalid_retained_payload(
                        "CCM resolution evidence did not follow the deterministic Q/2Q/4Q policy",
                    ));
                }
            }
            if entry.tolerance_met != final_met {
                return Err(invalid_retained_payload(
                    "CCM resolution-evidence tolerance verdict is inconsistent",
                ));
            }
        }
        Ok(())
    }
