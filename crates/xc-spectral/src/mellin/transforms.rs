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
thread_local! {
    static CHECKED_RULES: std::cell::RefCell<Vec<[u8; 32]>> = const { std::cell::RefCell::new(Vec::new()) };
    #[cfg(test)]
    static FULL_RULE_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    anyhow::ensure!(
        !nodes.is_empty() && nodes.len() == weights.len(),
        "Mellin rule shape"
    );
    // An arbitrary caller-supplied rule is fully checked once. Subsequent
    // transform arguments do not change that proof; all rule bits and the
    // arithmetic environment remain part of its process-local identity.
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"mellin-full-rule-check-v1\0");
    hash.update(p.to_le_bytes());
    hash.update(rug::float::exp_min().to_le_bytes());
    hash.update(rug::float::exp_max().to_le_bytes());
    hash.update((nodes.len() as u64).to_le_bytes());
    for value in nodes.iter().chain(weights) {
        let text = value.to_string_radix(16, None);
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    let digest: [u8; 32] = hash.finalize().into();
    if CHECKED_RULES.with(|rules| rules.borrow().contains(&digest)) {
        return Ok(());
    }
    #[cfg(test)]
    FULL_RULE_CHECKS.with(|count| count.set(count.get() + 1));
    let report =
        xc_numerics::quadrature::check_gauss_legendre_rule_hp(nodes, weights, p, nodes.len())?;
    anyhow::ensure!(
        report.checks_passed,
        "Mellin requires a validated Gauss-Legendre rule"
    );
    CHECKED_RULES.with(|rules| {
        let mut rules = rules.borrow_mut();
        if rules.len() == 8 {
            rules.remove(0);
        }
        rules.push(digest);
    });
    Ok(())
}

#[cfg(all(test, feature = "hp"))]
mod rule_reuse_tests {
    use super::*;
    #[test]
    fn changing_transform_arguments_does_not_replay_an_identical_rule() {
        use rug::Float;
        CHECKED_RULES.with(|rules| rules.borrow_mut().clear());
        FULL_RULE_CHECKS.with(|count| count.set(0));
        let p = 128;
        let (nodes, weights) = xc_numerics::quadrature::gauss_legendre_nodes(
            16,
            p,
            xc_numerics::quadrature::CacheMode::Off,
        );
        for t in 0..10 {
            validate_transform_hp(
                &Float::with_val(p, 1),
                &Float::with_val(p, t),
                &Float::with_val(p, 2 + t),
                &nodes,
                &weights,
            )
            .unwrap();
        }
        assert_eq!(FULL_RULE_CHECKS.with(|count| count.get()), 1);
        let mut changed = weights.clone();
        changed[1] = Float::with_val(p, -1);
        for expected in [2, 3] {
            assert!(validate_transform_hp(
                &Float::with_val(p, 1),
                &Float::with_val(p, 0),
                &Float::with_val(p, 2),
                &nodes,
                &changed
            )
            .is_err());
            assert_eq!(FULL_RULE_CHECKS.with(|count| count.get()), expected);
        }
    }
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
