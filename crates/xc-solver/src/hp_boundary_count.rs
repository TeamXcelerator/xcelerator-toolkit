use super::SolverError;
use rug::Float;
use xc_core::EigenTarget;
use xc_operator::{SpectralInertia, SymmetricOperator};

pub const HP_KRYLOV_COUNT_SEMANTICS: &str = "hp_krylov_source_bound_boundary_inertia_v4";
const DENSE_COUNT_MAXIMUM_BYTES: u64 = 64 * 1024 * 1024;
const DENSE_COUNT_MAXIMUM_DIMENSION: usize = 256;

/// Source-bound completeness evidence for the requested side of a Ritz boundary.
#[derive(Clone, Debug)]
pub enum BoundaryCountEvidenceHp {
    Verified {
        lower_shift: Option<Float>,
        upper_shift: Float,
        count: usize,
    },
    ObservedCluster,
    CountMismatch {
        expected: usize,
        observed: usize,
    },
    Unavailable {
        reason: String,
    },
}
impl BoundaryCountEvidenceHp {
    pub fn establishes_requested_count(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

fn checked_count(
    count: SpectralInertia,
    dimension: usize,
) -> Result<Option<SpectralInertia>, SolverError> {
    if count
        .below
        .checked_add(count.equal)
        .and_then(|n| n.checked_add(count.above))
        != Some(dimension)
    {
        return Err(SolverError::NumericalBreakdown(
            "source-bound spectral inertia does not exhaust the operator dimension".into(),
        ));
    }
    Ok(Some(count))
}

fn inertia(
    operator: &dyn SymmetricOperator<Float>,
    shift: &Float,
    p: u32,
) -> Result<Option<SpectralInertia>, SolverError> {
    if let Some(count) = operator.spectral_inertia_at(shift)? {
        return checked_count(count, operator.dimension());
    }
    let Some(matrix) = operator.stored_symmetric_entries() else {
        return Ok(None);
    };
    if operator.dimension() > DENSE_COUNT_MAXIMUM_DIMENSION {
        return Ok(None);
    }
    let source_precision = matrix
        .iter()
        .map(Float::prec)
        .max()
        .unwrap_or(p)
        .max(shift.prec())
        .max(p);
    let Some(working) = source_precision.checked_add(32).filter(|p| *p <= 1_000_000) else {
        return Ok(None);
    };
    use xc_numerics::{
        interval::IntervalError,
        symmetric_inertia::{point_matrix_inertia_at, MpfrInertiaResult},
    };
    match point_matrix_inertia_at(
        matrix,
        operator.dimension(),
        shift,
        working,
        DENSE_COUNT_MAXIMUM_BYTES,
    ) {
        Ok(MpfrInertiaResult::Conclusive {
            positive, negative, ..
        }) => checked_count(
            SpectralInertia {
                below: negative,
                equal: 0,
                above: positive,
            },
            operator.dimension(),
        ),
        Ok(MpfrInertiaResult::Inconclusive { .. }) | Err(IntervalError::Inconclusive(_)) => {
            Ok(None)
        }
        Err(error) => Err(SolverError::NumericalBreakdown(error.to_string())),
    }
}

pub(super) fn boundary_count(
    operator: &dyn SymmetricOperator<Float>,
    target: &EigenTarget,
    requested: usize,
    values: &[Float],
    cluster_tolerance: &Float,
    p: u32,
) -> Result<BoundaryCountEvidenceHp, SolverError> {
    let unavailable = || BoundaryCountEvidenceHp::Unavailable {
        reason: "complete source-bound count unavailable at the requested separating boundary"
            .into(),
    };
    if requested == 0 || values.len() <= requested {
        return Ok(unavailable());
    }
    let (left, right) = (&values[requested - 1], &values[requested]);
    let shift = match target {
        EigenTarget::SmallestMagnitude => Some(Float::with_val(p, 0)),
        EigenTarget::ClosestTo { shift: s } => Some(Float::with_val(
            p,
            Float::parse(s.as_str())
                .map_err(|e| SolverError::InvalidConfiguration(e.to_string()))?,
        )),
        EigenTarget::AlgebraicSmallest | EigenTarget::AlgebraicLargest => None,
        _ => {
            return Err(SolverError::UnsupportedTarget(
                "boundary count requires an algebraic or shifted extreme target".into(),
            ))
        }
    };
    let distance = |x: &Float| match &shift {
        Some(s) => Float::with_val(p, x - s).abs(),
        None => x.clone(),
    };
    let dl = distance(left);
    let dr = distance(right);
    let cluster_tolerance = super::hp_effective_cluster_tolerance(
        cluster_tolerance,
        values.iter(),
        operator.dimension(),
        p,
    );
    let gap = Float::with_val(p, &dr - &dl).abs();
    if gap <= cluster_tolerance {
        return Ok(BoundaryCountEvidenceHp::ObservedCluster);
    }
    let midpoint = Float::with_val(p, Float::with_val(p, &dl + &dr) / 2u32);
    let (lower, upper, count) = if let Some(s) = shift {
        let lower = Float::with_val(p, &s - &midpoint);
        let upper = Float::with_val(p, &s + &midpoint);
        if lower.partial_cmp(&upper) != Some(std::cmp::Ordering::Less) {
            return Ok(unavailable());
        }
        let (Some(lo), Some(hi)) = (inertia(operator, &lower, p)?, inertia(operator, &upper, p)?)
        else {
            return Ok(unavailable());
        };
        if lo.equal != 0 || hi.equal != 0 {
            return Ok(unavailable());
        }
        let Some(count) = hi.below.checked_sub(lo.below) else {
            return Err(SolverError::NumericalBreakdown(
                "source inertia counts are not monotone".into(),
            ));
        };
        (Some(lower), upper, count)
    } else {
        let Some(count) = inertia(operator, &midpoint, p)? else {
            return Ok(unavailable());
        };
        if count.equal != 0 {
            return Ok(unavailable());
        }
        let count = if *target == EigenTarget::AlgebraicLargest {
            count.above
        } else {
            count.below
        };
        (None, midpoint, count)
    };
    if count == requested {
        Ok(BoundaryCountEvidenceHp::Verified {
            lower_shift: lower,
            upper_shift: upper,
            count,
        })
    } else {
        Ok(BoundaryCountEvidenceHp::CountMismatch {
            expected: requested,
            observed: count,
        })
    }
}

/// Count the closed exact-decimal interval. Nonrepresentable decimal endpoints
/// are admitted only when counts prove that their rounding bands contain no roots.
pub(super) fn interval_count(
    operator: &dyn SymmetricOperator<Float>,
    lower: &xc_core::DecimalLiteral,
    upper: &xc_core::DecimalLiteral,
    expected: usize,
    p: u32,
) -> Result<BoundaryCountEvidenceHp, SolverError> {
    use rug::float::Round;
    let unavailable = || BoundaryCountEvidenceHp::Unavailable {
        reason:
            "source-bound closed interval count or endpoint rounding-band exclusion unavailable"
                .into(),
    };
    let ld = super::hp_parse_literal_round(lower, p, Round::Down)?;
    let lu = super::hp_parse_literal_round(lower, p, Round::Up)?;
    let ud = super::hp_parse_literal_round(upper, p, Round::Down)?;
    let uu = super::hp_parse_literal_round(upper, p, Round::Up)?;
    let (Some(lod), Some(lou), Some(hid), Some(hiu)) = (
        inertia(operator, &ld, p)?,
        inertia(operator, &lu, p)?,
        inertia(operator, &ud, p)?,
        inertia(operator, &uu, p)?,
    ) else {
        return Ok(unavailable());
    };
    if (ld != lu && (lod.equal != 0 || lou.equal != 0 || lod.below != lou.below))
        || (ud != uu && (hid.equal != 0 || hiu.equal != 0 || hid.below != hiu.below))
    {
        return Ok(unavailable());
    }
    let upper_inclusive = hiu
        .below
        .checked_add(hiu.equal)
        .ok_or_else(|| SolverError::NumericalBreakdown("source interval count overflow".into()))?;
    let Some(count) = upper_inclusive.checked_sub(lod.below) else {
        return Err(SolverError::NumericalBreakdown(
            "source interval counts are not monotone".into(),
        ));
    };
    if count == expected {
        Ok(BoundaryCountEvidenceHp::Verified {
            lower_shift: Some(ld),
            upper_shift: uu,
            count,
        })
    } else {
        Ok(BoundaryCountEvidenceHp::CountMismatch {
            expected,
            observed: count,
        })
    }
}
