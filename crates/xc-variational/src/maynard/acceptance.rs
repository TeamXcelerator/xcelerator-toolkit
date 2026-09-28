//! Replay finite M_k numerical evidence against exact source forms.
use super::*;
use rug::Assign;

fn invalid(message: &str) -> MkError {
    MkError::InvalidProblem(message.into())
}

pub(super) fn validate_precision(p: u32) -> Result<(), MkError> {
    if !(65..=999_936).contains(&p) {
        return Err(invalid("M_k acceptance precision must be in 65..=999936"));
    }
    Ok(())
}

fn point(text: &str, p: u32) -> Result<Rational, MkError> {
    let decimal =
        xc_core::DecimalLiteral::new(text.to_owned()).map_err(|e| invalid(&e.to_string()))?;
    let parsed = Float::parse(decimal.as_str()).map_err(|e| invalid(&e.to_string()))?;
    let value = Float::with_val(p, parsed);
    if !value.is_finite()
        || (value.is_zero()
            && decimal
                .cmp_numeric(&xc_core::DecimalLiteral::new("0").unwrap())
                .map_err(|e| invalid(&e.to_string()))?
                != std::cmp::Ordering::Equal)
    {
        return Err(invalid(
            "M_k acceptance value is not representable at its owning precision",
        ));
    }
    value
        .to_rational()
        .ok_or_else(|| invalid("nonfinite M_k acceptance value"))
}

fn exact_decimal(text: &str) -> Result<Rational, MkError> {
    use rug::ops::Pow;
    let canonical = xc_core::DecimalLiteral::new(text)
        .and_then(|d| d.canonical())
        .map_err(|e| invalid(&e.to_string()))?;
    let (significand, exponent) = canonical
        .as_str()
        .split_once('e')
        .map_or((canonical.as_str(), "0"), |(s, e)| (s, e));
    let exponent = exponent
        .parse::<i64>()
        .map_err(|e| invalid(&e.to_string()))?;
    if exponent.unsigned_abs() > 1_000_000 || significand.len() > 1_000_000 {
        return Err(invalid("acceptance decimal exceeds rational budget"));
    }
    let numerator =
        Integer::from_str_radix(significand, 10).map_err(|e| invalid(&e.to_string()))?;
    let power = Integer::from(10).pow(exponent.unsigned_abs() as u32);
    Ok(if exponent >= 0 {
        Rational::from(numerator * power)
    } else {
        Rational::from((numerator, power))
    })
}

fn tolerance(value: &xc_core::DecimalLiteral, p: u32, positive: bool) -> Result<Rational, MkError> {
    value.validate().map_err(|e| invalid(&e.to_string()))?;
    let parsed = Float::parse(value.as_str()).map_err(|e| invalid(&e.to_string()))?;
    let lower = Float::with_val_round(p + 64, parsed, Round::Down).0;
    if !lower.is_finite() || lower < 0 || (positive && lower.is_zero()) {
        return Err(invalid(
            "M_k acceptance tolerance must be nonnegative, finite and representable",
        ));
    }
    lower
        .to_rational()
        .ok_or_else(|| invalid("invalid tolerance"))
}

pub(super) fn compact_candidate(values: &mut [Float], p: u32) -> Result<(), MkError> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err(invalid("M_k candidate must be finite and nonempty"));
    }
    let maximum = values
        .iter()
        .map(|x| x.clone().abs())
        .max_by(Float::total_cmp)
        .ok_or_else(|| invalid("empty M_k candidate"))?;
    if maximum.is_zero() {
        return Err(invalid("zero M_k candidate"));
    }
    let threshold = maximum
        >> p.checked_add(64)
            .ok_or_else(|| invalid("candidate precision overflow"))?;
    for value in values {
        if value.clone().abs() < threshold {
            value.assign(0);
        }
    }
    Ok(())
}

fn upper_text(value: &Rational, p: u32, square_root: bool) -> Result<String, MkError> {
    let mut upper = Float::with_val_round(p, value, Round::Up).0;
    if square_root {
        upper.sqrt_round(Round::Up);
    }
    if !upper.is_finite() || (upper.is_zero() && value != &0) {
        return Err(invalid("M_k diagnostic bound is not representable"));
    }
    // An exact declared upper bound may be larger than the measured error.
    // Bound the serialized exponent without changing exact-source replay.
    if !upper.is_zero() {
        let floor = Float::with_val_round(p, Float::parse("1e-500000").unwrap(), Round::Down).0;
        if upper < floor {
            return Ok("1e-500000".into());
        }
    }
    Ok(upper.to_string_radix_round(10, None, Round::Up))
}

fn check_bound(
    actual: &Rational,
    encoded: &str,
    limit: Option<&Rational>,
    p: u32,
    squared: bool,
) -> Result<(), MkError> {
    let declared = exact_decimal(encoded)?;
    let _ = p; // Bounds are decimal premises, not rounded source points.
    if declared < 0 {
        return Err(invalid("M_k diagnostic bound must be nonnegative"));
    }
    let coverage = if squared {
        declared.clone() * &declared
    } else {
        declared.clone()
    };
    if &coverage < actual || limit.is_some_and(|limit| &declared > limit) {
        return Err(invalid(
            "M_k diagnostic fails exact replay or exceeds its acceptance bound",
        ));
    }
    Ok(())
}

pub(super) fn validate_three_options(
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    validate_precision(options.precision_bits)?;
    if options.initial_precision_bits <= 32
        || options.initial_precision_bits > options.precision_bits
        || options.maximum_iterations < 2
    {
        return Err(invalid(
            "M_k acceptance requires valid initial precision and at least two iterations",
        ));
    }
    for bound in [
        &options.absolute_residual_tolerance,
        &options.scaled_backward_error_tolerance,
        &options.ritz_value_stability_tolerance,
    ] {
        tolerance(bound, options.precision_bits, true)?;
    }
    for bound in [
        &options.eigenvalue_agreement_tolerance,
        &options.overlap_tolerance,
        &options.candidate_quotient_agreement_tolerance,
    ] {
        tolerance(bound, options.precision_bits, false)?;
    }
    Ok(())
}

fn dot(a: &[Rational], b: &[Rational]) -> Rational {
    a.iter().zip(b).fold(Rational::new(), |mut sum, (a, b)| {
        sum += a.clone() * b;
        sum
    })
}

struct State {
    metric_image: Vec<Rational>,
    metric_norm: Rational,
    residual_squared: Rational,
    operator_squared: Rational,
    metric_squared: Rational,
}

fn state(
    reference: &MkSymmetricReference,
    x: &[Rational],
    lambda: &Rational,
) -> Result<State, MkError> {
    if x.len() != reference.dimension() {
        return Err(invalid("M_k replay eigenvector dimension mismatch"));
    }
    let mut metric_image = Vec::with_capacity(x.len());
    let mut operator_image = Vec::with_capacity(x.len());
    for row in 0..x.len() {
        let mut i = Rational::new();
        let mut j = Rational::new();
        for (column, coefficient) in x.iter().enumerate() {
            i += reference.i_entry(row, column)? * coefficient;
            j += reference.j_total_entry(row, column)? * coefficient;
        }
        metric_image.push(i);
        operator_image.push(j);
    }
    let metric_norm = dot(x, &metric_image);
    if metric_norm <= 0 {
        return Err(MkError::NonPositiveDenominator);
    }
    orthogonal::budget(x.iter().chain(&metric_image).chain(&operator_image))?;
    let residual: Vec<_> = operator_image
        .iter()
        .zip(&metric_image)
        .map(|(j, i)| j.clone() - lambda.clone() * i)
        .collect();
    Ok(State {
        metric_norm,
        residual_squared: dot(&residual, &residual),
        operator_squared: dot(&operator_image, &operator_image),
        metric_squared: dot(&metric_image, &metric_image),
        metric_image,
    })
}

fn converged(
    state: &State,
    lambda: &Rational,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    let p = options.precision_bits;
    let unit_tolerance = Rational::from((Integer::from(1), Integer::from(1) << (p / 2)));
    if (state.metric_norm.clone() - 1i32).abs() > unit_tolerance {
        return Err(invalid(
            "M_k accepted state must have unit exact-source I norm within 2^(-p/2)",
        ));
    }
    let absolute = tolerance(&options.absolute_residual_tolerance, p, true)?;
    if state.residual_squared <= absolute.clone() * absolute * &state.metric_norm {
        return Ok(());
    }
    let backward = tolerance(&options.scaled_backward_error_tolerance, p, true)?;
    // Lower-bound ||Jx|| + |lambda| ||Ix||, so the comparison is conservative.
    let sqrt_down = |value: &Rational| -> Result<Rational, MkError> {
        let mut lower = Float::with_val_round(p + 64, value, Round::Down).0;
        lower.sqrt_round(Round::Down);
        lower
            .to_rational()
            .ok_or_else(|| invalid("M_k replay norm is not representable"))
    };
    let denominator = sqrt_down(&state.operator_squared)?
        + lambda.clone().abs() * sqrt_down(&state.metric_squared)?;
    let permitted = backward * denominator;
    if state.residual_squared > permitted.clone() * permitted {
        return Err(invalid(
            "M_k eigenpair residual fails exact source-form replay",
        ));
    }
    Ok(())
}

// Positive definiteness of U I-J proves lambda_max < U for the exact
// finite source forms. This binds target ordering independently of residuals.
fn largest_target(
    reference: &MkSymmetricReference,
    lambda: &Rational,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    let n = reference.dimension();
    if n == 0 || n > 128 {
        return Err(invalid(
            "largest-target exact replay is limited to 128 directions",
        ));
    }
    let upper = lambda.clone()
        + tolerance(
            &options.eigenvalue_agreement_tolerance,
            options.precision_bits,
            false,
        )?;
    let mut a = Vec::with_capacity(n * n);
    for row in 0..n {
        for column in 0..n {
            a.push(
                reference.i_entry(row, column)? * &upper - reference.j_total_entry(row, column)?,
            );
        }
    }
    orthogonal::budget(a.iter())?;
    let mut lower = vec![Rational::from(0); n * n];
    let mut diagonal = vec![Rational::from(0); n];
    for row in 0..n {
        lower[row * n + row] = Rational::from(1);
        for column in 0..=row {
            let mut value = a[row * n + column].clone();
            for k in 0..column {
                value -= lower[row * n + k].clone() * &lower[column * n + k] * &diagonal[k];
                orthogonal::budget(std::iter::once(&value))?;
            }
            if row == column {
                if value <= 0 {
                    return Err(invalid(
                        "M_k largest-eigenvalue upper bound is not positive definite",
                    ));
                }
                diagonal[row] = value;
            } else {
                lower[row * n + column] = value / &diagonal[column];
            }
        }
        orthogonal::budget(a.iter().chain(&lower).chain(&diagonal))?;
    }
    Ok(())
}

struct Metrics {
    difference: Rational,
    candidate_difference: Rational,
    overlap: Rational,
    left: State,
    right: State,
}

fn metrics(
    record: &MkThreeRouteAcceptanceRecord,
    reference: &MkSymmetricReference,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<Metrics, MkError> {
    validate_three_options(options)?;
    let attempts = &record.adaptive_attempt_precisions;
    if attempts.first() != Some(&options.initial_precision_bits)
        || attempts.len() > 2
        || attempts
            .windows(2)
            .any(|v| v[0] >= v[1] || v[1] != options.precision_bits)
    {
        return Err(invalid(
            "M_k adaptive precision sequence does not match its policy",
        ));
    }
    let left_value = point(
        &record.matrix_free_eigenvalue,
        *attempts
            .last()
            .ok_or_else(|| invalid("missing M_k precision attempts"))?,
    )?;
    let right_value = point(&record.dense_eigenvalue, options.precision_bits)?;
    let left_vector = record
        .candidate_coefficients
        .iter()
        .map(rational_from_record)
        .collect::<Result<Vec<_>, _>>()?;
    let right_vector = record
        .dense_coefficients
        .iter()
        .map(rational_from_record)
        .collect::<Result<Vec<_>, _>>()?;
    let left = state(reference, &left_vector, &left_value)?;
    let right = state(reference, &right_vector, &right_value)?;
    converged(&left, &left_value, options)?;
    converged(&right, &right_value, options)?;
    largest_target(reference, &left_value, options)?;
    largest_target(reference, &right_value, options)?;
    let cross = dot(&left_vector, &right.metric_image);
    let mut overlap = cross.clone() * cross;
    overlap /= left.metric_norm.clone() * &right.metric_norm;
    overlap = Rational::from(1) - overlap;
    if overlap < 0 {
        return Err(invalid("exact M_k metric overlap violates Cauchy-Schwarz"));
    }
    let candidate = rational_from_record(&record.candidate_certificate.quotient)?;
    Ok(Metrics {
        difference: (left_value.clone() - right_value).abs(),
        candidate_difference: (left_value - candidate).abs(),
        overlap,
        left,
        right,
    })
}

pub(super) fn refresh_three(
    record: &mut MkThreeRouteAcceptanceRecord,
    reference: &MkSymmetricReference,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    let m = metrics(record, reference, options)?;
    let p = options.precision_bits;
    record.eigenvalue_absolute_difference = upper_text(&m.difference, p, false)?;
    record.candidate_absolute_difference = upper_text(&m.candidate_difference, p, false)?;
    record.one_minus_metric_overlap_squared = upper_text(&m.overlap, p, false)?;
    record.matrix_free_residual_norm = upper_text(&m.left.residual_squared, p, true)?;
    record.dense_residual_norm = upper_text(&m.right.residual_squared, p, true)?;
    Ok(())
}

pub(super) fn verify_three(
    record: &MkThreeRouteAcceptanceRecord,
    reference: &MkSymmetricReference,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    let m = metrics(record, reference, options)?;
    let p = options.precision_bits;
    for (actual, encoded, limit) in [
        (
            &m.difference,
            &record.eigenvalue_absolute_difference,
            &options.eigenvalue_agreement_tolerance,
        ),
        (
            &m.candidate_difference,
            &record.candidate_absolute_difference,
            &options.candidate_quotient_agreement_tolerance,
        ),
        (
            &m.overlap,
            &record.one_minus_metric_overlap_squared,
            &options.overlap_tolerance,
        ),
    ] {
        check_bound(
            actual,
            encoded,
            Some(&tolerance(limit, p, false)?),
            p,
            false,
        )?;
    }
    check_bound(
        &m.left.residual_squared,
        &record.matrix_free_residual_norm,
        None,
        p,
        true,
    )?;
    check_bound(
        &m.right.residual_squared,
        &record.dense_residual_norm,
        None,
        p,
        true,
    )
}

pub(super) fn scale_difference(
    record: &MkScaleAcceptanceRecord,
    quotient: &Rational,
    options: &MkScaleAcceptanceOptions,
) -> Result<Rational, MkError> {
    validate_precision(options.precision_bits)?;
    if options.historical_dense_degree_limit < 3
        || options.target_degree <= options.historical_dense_degree_limit
    {
        return Err(invalid("invalid M_k scale degree policy"));
    }
    tolerance(
        &options.quotient_agreement_tolerance,
        options.precision_bits,
        false,
    )?;
    Ok((point(&record.matrix_free_quotient, options.precision_bits)? - quotient).abs())
}

pub(super) fn refresh_scale(
    record: &mut MkScaleAcceptanceRecord,
    options: &MkScaleAcceptanceOptions,
) -> Result<(), MkError> {
    let quotient = rational_from_record(&record.exact_certificate.quotient)?;
    record.quotient_absolute_difference = upper_text(
        &scale_difference(record, &quotient, options)?,
        options.precision_bits,
        false,
    )?;
    Ok(())
}

pub(super) fn verify_scale(
    record: &MkScaleAcceptanceRecord,
    quotient: &Rational,
    options: &MkScaleAcceptanceOptions,
) -> Result<(), MkError> {
    let actual = scale_difference(record, quotient, options)?;
    check_bound(
        &actual,
        &record.quotient_absolute_difference,
        Some(&tolerance(
            &options.quotient_agreement_tolerance,
            options.precision_bits,
            false,
        )?),
        options.precision_bits,
        false,
    )
}

#[cfg(test)]
mod acceptance_tests {
    use super::*;
    fn options() -> MkThreeRouteAcceptanceOptions {
        let d = |s| xc_core::DecimalLiteral::new(s).unwrap();
        MkThreeRouteAcceptanceOptions {
            k: 5,
            degree: 3,
            precision_bits: 192,
            initial_precision_bits: 192,
            absolute_residual_tolerance: d("1e-25"),
            scaled_backward_error_tolerance: d("1e-25"),
            ritz_value_stability_tolerance: d("1e-25"),
            eigenvalue_agreement_tolerance: d("1e-20"),
            overlap_tolerance: d("1e-20"),
            candidate_quotient_agreement_tolerance: d("1e-20"),
            maximum_iterations: 3000,
        }
    }
    #[test]
    fn tiny_vector_cannot_hide_a_non_eigenstate_residual() {
        let o = options();
        let reference = MkSymmetricReference::new(o.k, o.degree).unwrap();
        let tiny = Rational::from((Integer::from(1), Integer::from(1) << 600u32));
        let v = vec![tiny; reference.dimension()];
        let actual = state(&reference, &v, &Rational::from(1)).unwrap();
        assert!(
            actual.residual_squared
                < tolerance(&o.absolute_residual_tolerance, 192, true)
                    .unwrap()
                    .square()
        );
        assert!(converged(&actual, &Rational::from(1), &o).is_err());
    }
    #[test]
    fn largest_target_requires_an_independent_upper_bound() {
        let o = options();
        let reference = MkSymmetricReference::new(o.k, o.degree).unwrap();
        // Constant trial already has quotient 2k/(k+1)>1. Every lambda<1 is
        // therefore excluded as a largest eigenvalue, independent of residual.
        assert!(largest_target(&reference, &Rational::from(1), &o).is_err());
        largest_target(&reference, &Rational::from(5), &o).unwrap();
    }
    #[test]
    fn decimal_bound_just_above_limit_is_not_rounded_down() {
        let one = Rational::from(1);
        assert!(check_bound(
            &Rational::from(0),
            "1.000000000000000000000000000000000000000001",
            Some(&one),
            65,
            false
        )
        .is_err());
        let third = Rational::from((1, 3));
        let encoded = upper_text(&third, 65, false).unwrap();
        assert!(exact_decimal(&encoded).unwrap() >= third);
        check_bound(&third, &encoded, None, 65, false).unwrap();
    }
}
