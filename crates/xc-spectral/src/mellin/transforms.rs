pub(super) fn validate_transform_f64(
    s_re: f64,
    s_im: f64,
    lambda: f64,
    n: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        s_re.is_finite() && s_im.is_finite() && lambda.is_finite() && lambda >= 1.0 && n > 0,
        "Mellin requires finite s, lambda >= 1, and positive quadrature order"
    );
    Ok(())
}

// u^(s_re-1) omega(u) = exp(s_re log(u)-u-2 log(1+exp(-u))).
// Forming the factors separately loses finite products to 0*infinity.
pub(super) fn mellin_amplitude_f64(s_re: f64, u: f64) -> f64 {
    (s_re * u.ln() - u - 2.0 * (-u).exp().ln_1p()).exp()
}

#[cfg(feature = "hp")]
pub(super) fn validate_transform_hp(
    s_re: &rug::Float,
    s_im: &rug::Float,
    lambda: &rug::Float,
    nodes: &[rug::Float],
    weights: &[rug::Float],
) -> anyhow::Result<()> {
    let p = lambda.prec();
    anyhow::ensure!(
        (32..=1_000_000).contains(&p) && lambda >= &rug::Float::with_val(p, 1),
        "Mellin precision/lambda domain"
    );
    anyhow::ensure!(
        [s_re, s_im, lambda]
            .into_iter()
            .chain(nodes)
            .chain(weights)
            .all(|v| v.is_finite() && v.prec() == p),
        "Mellin inputs must be finite at the working precision"
    );
    let report =
        xc_numerics::quadrature::check_gauss_legendre_rule_hp(nodes, weights, p, nodes.len())?;
    anyhow::ensure!(
        report.checks_passed,
        "Mellin requires a validated Gauss-Legendre rule"
    );
    Ok(())
}

#[cfg(feature = "hp")]
pub(super) fn mellin_amplitude_hp(s_re: &rug::Float, u: &rug::Float) -> rug::Float {
    use rug::Float;
    let p = u.prec();
    let guard = p + 64;
    let x = Float::with_val(guard, u);
    let mut exponent = x.clone().ln();
    exponent *= s_re;
    exponent -= &x;
    let mut denominator_log = (-x).exp().ln_1p();
    denominator_log *= 2u32;
    exponent -= denominator_log;
    Float::with_val(p, exponent.exp())
}
