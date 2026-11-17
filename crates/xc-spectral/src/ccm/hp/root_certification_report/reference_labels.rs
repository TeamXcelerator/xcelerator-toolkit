//! Reconcile reference-seeded points with independently counted movable roots.
use super::*;

pub(super) fn certify(
    params: &CcmParams,
    cfg: &HighPrecConfig,
    primary: &HighPrecResult,
    weights: &[Float],
    options: &CcmRootCertificationOptions,
    first: usize,
    last: usize,
) -> Result<PartialIndependentCcmRootCertification> {
    let count = (last - first + 1).min(params.matrix_size());
    #[cfg(feature = "arb")]
    {
        use crate::ccm::certified_roots::{certify_partial_scope, PartialRootScope};
        let max_point = (first..=last)
            .filter_map(|label| match point(primary, label)? {
                EigenvalueResult::Converged(root) if root.value.is_finite() && root.value > 0 => {
                    Some(&root.value)
                }
                _ => None,
            })
            .max_by(|a, b| a.total_cmp(b));
        match max_point {
            Some(highest) => certify_partial_scope(
                weights,
                &lambda_squared_cache_identity(params),
                params.n_modes,
                PartialRootScope::ThroughPoint(highest),
                cfg.precision_bits,
                options.isolation_bits,
                &options.interval_newton,
            ),
            None => Ok(unavailable(
                1,
                count,
                "no converged finite positive reference-labelled point".into(),
            )),
        }
    }
    #[cfg(not(feature = "arb"))]
    {
        let _ = (cfg, primary, weights, options);
        Ok(unavailable(
            1,
            count,
            "root certification requires the xc-spectral arb feature".into(),
        ))
    }
}

fn point(primary: &HighPrecResult, label: usize) -> Option<&EigenvalueResult> {
    primary
        .eigenvalues_pos
        .get(label.checked_sub(primary.first_positive_root_index)?)
}

pub(super) fn rows(
    primary: &HighPrecResult,
    first: usize,
    last: usize,
    proof: &PartialIndependentCcmRootCertification,
) -> Result<Vec<CcmRootCertificationRow>> {
    let mut boxes = BTreeMap::new();
    for cert in &proof.certificates {
        let start = cert
            .first_selected_positive_index
            .context("missing independently counted rank")?;
        for (offset, witness) in cert.try_selected_roots()?.iter().enumerate() {
            anyhow::ensure!(
                boxes.insert(start + offset, witness).is_none(),
                "duplicate certified finite rank"
            );
        }
    }
    let mut previous_rank = 0;
    let mut rows = Vec::new();
    for label in first..=last {
        let computed = point(primary, label);
        let value = computed.and_then(EigenvalueResult::value);
        let mut matched = None;
        if let Some(EigenvalueResult::Converged(root)) = computed {
            for (&rank, &witness) in &boxes {
                if stored_root_in_decimal_interval(
                    &root.value,
                    &witness.lower,
                    &witness.upper,
                    witness.precision_bits,
                )? {
                    anyhow::ensure!(
                        matched.is_none(),
                        "point belongs to overlapping certified enclosures"
                    );
                    matched = Some((rank, witness));
                }
            }
        }
        // A duplicate or reversed assignment must never complete the request.
        let matched = matched.filter(|(rank, _)| *rank > previous_rank);
        if let Some((rank, _)) = matched {
            previous_rank = rank;
        }
        rows.push(CcmRootCertificationRow {
            ordinal: label,
            computed_status: computed.map_or("absent", computed_status).into(),
            computed_value: value.map(lossless_hp_decimal),
            outcome: if matched.is_some() {
                "certified_finite_enclosure"
            } else if value.is_some() {
                "computed_not_certified"
            } else {
                "not_computed_not_certified"
            }.into(),
            certified_lower: matched.map(|(_, witness)| witness.lower.clone()),
            certified_upper: matched.map(|(_, witness)| witness.upper.clone()),
            certified_precision_bits: matched.map(|(_, witness)| witness.precision_bits),
            computed_agreement: if matched.is_some() { "inside" } else { "not_certified" }.into(),
            computed_matches_certified_ordinal: matched.map(|(rank, _)| rank),
            reason: matched.is_none().then(|| "no unique ordered certified finite rank for this converged reference-labelled point".into()),
        });
    }
    Ok(rows)
}

pub(super) fn validate(
    report: &CcmRootCertificationReport,
    primary: &HighPrecResult,
    weights: &ContentDigest,
) -> Result<()> {
    let proof = &report.certification;
    let (first, last) = requested_ordinals(&report.request.target, primary);
    anyhow::ensure!(
        proof.first_index == 1
            && proof.last_index > 0
            && proof.last_index <= report.n_modes.saturating_mul(2).saturating_add(1)
            && (report.request.first_index, report.request.last_index) == (first, last)
            && report.reason == proof.source_limitation
            && (proof.source_limitation.is_none() || proof.certificates.is_empty()),
        "invalid reference-label coverage"
    );
    let mut ranks = std::collections::BTreeSet::new();
    for cert in &proof.certificates {
        let low = cert
            .first_selected_positive_index
            .context("missing finite rank")?;
        let high = cert
            .last_selected_positive_index
            .context("missing last finite rank")?;
        let requested_ranks = match cert.target {
            IndependentCcmRootTarget::Prefix { count } => (1, count),
            IndependentCcmRootTarget::IndexRange { first, last } => (first, last),
            IndependentCcmRootTarget::PositiveHeightWindow { .. } => {
                anyhow::bail!("expected an independently indexed certificate")
            }
        };
        anyhow::ensure!(
            low > 0 && low <= high && high <= proof.last_index && requested_ranks == (low, high),
            "wrong independent finite-rank request"
        );
        anyhow::ensure!(
            cert.source_weights_digest == *weights
                && cert.isolation_bits == report.request.isolation_bits
                && cert.interval_newton == report.request.interval_newton,
            "wrong source or certificate policy"
        );
        anyhow::ensure!(
            (low..=high).all(|r| ranks.insert(r)),
            "duplicate certified rank"
        );
    }
    anyhow::ensure!(
        proof
            .uncertified
            .keys()
            .copied()
            .eq((1..=proof.last_index).filter(|r| !ranks.contains(r))),
        "incomplete independent census inventory"
    );
    let expected = rows(primary, first, last, proof)?;
    let certified = expected
        .iter()
        .filter(|r| r.computed_agreement == "inside")
        .count();
    let missing = expected
        .iter()
        .filter(|r| r.computed_value.is_none())
        .count();
    anyhow::ensure!(
        report.rows == expected
            && report.certified_rows == certified
            && report.unresolved_rows == missing
            && report.computed_not_certified_rows == expected.len() - certified - missing
            && report.computed_outside_certified_rows == 0
            && report.outcome
                == if certified == expected.len() {
                    "certified_finite_enclosure"
                } else {
                    "partial_certification"
                },
        "reference labels, finite ranks or report verdict disagree with retained evidence"
    );
    Ok(())
}
