//! Scaled point evaluation of the meromorphic regularized matrix element.
use anyhow::{ensure, Result};

pub(super) fn entry_f64(sr: f64, si: f64, tr: f64, ti: f64, epsilon: f64) -> Result<(f64, f64)> {
    ensure!(
        [sr, si, tr, ti, epsilon].iter().all(|x| x.is_finite()) && epsilon > 0.0,
        "matrix element requires finite parameters and epsilon > 0"
    );
    // Compensate the sum before subtracting one, so a residual smaller than
    // one ulp at 1 is not silently replaced by the critical-line value.
    let sum = sr + tr;
    let ar = if sum.is_finite() {
        let v = sum - sr;
        let error = (sr - (sum - v)) + (tr - v);
        (sum - 1.0) + error
    } else {
        sum
    };
    let ai = ti - si;
    if ar == 0.0 && ai == 0.0 {
        return Ok((1.0, 0.0));
    }
    let scale = if ar.is_finite() && ai.is_finite() {
        epsilon.max(ar.abs()).max(ai.abs())
    } else {
        [sr.abs(), si.abs(), tr.abs(), ti.abs(), epsilon, 1.0]
            .into_iter()
            .fold(0.0, f64::max)
    };
    let r = if ar.is_finite() {
        ar / scale
    } else {
        sr / scale + tr / scale - 1.0 / scale
    };
    let i = if ai.is_finite() {
        ai / scale
    } else {
        ti / scale - si / scale
    };
    let e = epsilon / scale;
    ensure!(
        e > 0.0
            && !(ar.is_finite() && ar != 0.0 && r == 0.0)
            && !(ai.is_finite() && ai != 0.0 && i == 0.0),
        "matrix-element scaling underflowed"
    );
    let numerator = e * e;
    ensure!(numerator > 0.0, "matrix element is outside binary64 range");
    let di = -2.0 * r * i;
    let difference = (e - r.abs()) * (e + r.abs());
    if i.abs() < e && di != 0.0 && !(i * i).is_normal() && difference.abs() < di.abs() {
        // i^2 underflows while e^2 - r^2 does not dominate Im(D), so i^2 still
        // decides the real part: form rho = Re(D)/Im(D) =
        // (e^2 - r^2)/Im(D) + i/(-2r) without squaring i.
        let rho = difference / di + i / (-2.0 * r);
        let base = (numerator / di) / rho.mul_add(rho, 1.0);
        let real = base * rho;
        let imaginary = -base;
        ensure!(
            real.is_finite() && imaginary.is_finite() && imaginary != 0.0,
            "matrix element is outside binary64 range"
        );
        return Ok((real, imaginary));
    }
    // Subtract the closest competing squares before adding the small square.
    // In particular |i|=|r| must preserve e^2, however small it is.
    let dr = if i.abs() >= e {
        e.mul_add(e, (i.abs() - r.abs()) * (i.abs() + r.abs()))
    } else {
        i.mul_add(i, (e - r.abs()) * (e + r.abs()))
    };
    let di = -2.0 * r * i;
    let denominator_scale = dr.abs().max(di.abs());
    ensure!(
        denominator_scale.is_finite() && denominator_scale > 0.0,
        "matrix-element pole or unrepresentable denominator"
    );
    let dr = dr / denominator_scale;
    let di = di / denominator_scale;
    let norm = dr.mul_add(dr, di * di);
    let real = (numerator * (dr / norm)) / denominator_scale;
    let imaginary = -(numerator * (di / norm)) / denominator_scale;
    ensure!(
        real.is_finite() && imaginary.is_finite() && (real != 0.0 || imaginary != 0.0),
        "matrix element is outside binary64 range"
    );
    Ok((real, imaginary))
}

pub(super) fn spectrum_f64(input: &[f64], n: usize) -> Result<Vec<f64>> {
    ensure!(
        n > 0 && n.checked_mul(n) == Some(input.len()),
        "matrix must be nonempty and square"
    );
    ensure!(
        input.iter().all(|x| x.is_finite())
            && (0..n).all(|i| (0..i).all(|j| input[i * n + j] == input[j * n + i])),
        "matrix must be finite and exactly symmetric"
    );
    let scale = input.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(vec![0.0; n]);
    }
    let entries: Vec<f64> = input.iter().map(|x| x / scale).collect();
    ensure!(
        entries
            .iter()
            .zip(input)
            .all(|(v, x)| *v != 0.0 || *x == 0.0),
        "matrix scaling underflowed"
    );
    let matrix = nalgebra::DMatrix::from_row_slice(n, n, &entries);
    let decomposition =
        nalgebra::SymmetricEigen::try_new(matrix, f64::EPSILON, n.saturating_mul(128))
            .ok_or_else(|| anyhow::anyhow!("symmetric eigenvalue iteration did not converge"))?;
    let mut values: Vec<f64> = decomposition
        .eigenvalues
        .iter()
        .map(|v| v * scale)
        .collect();
    ensure!(
        values
            .iter()
            .zip(decomposition.eigenvalues.iter())
            .all(|(v, x)| v.is_finite() && (*v != 0.0 || *x == 0.0)),
        "eigenvalue rescaling is outside binary64 range"
    );
    values.sort_by(f64::total_cmp);
    Ok(values)
}

#[cfg(feature = "hp")]
pub(super) fn entry_hp(
    sr: &rug::Float,
    si: &rug::Float,
    tr: &rug::Float,
    ti: &rug::Float,
    epsilon: &rug::Float,
    p: u32,
) -> Result<(rug::Float, rug::Float)> {
    use rug::Float;
    ensure!(
        (32..=1_000_000).contains(&p),
        "matrix-element precision must be in 32..=1000000 bits"
    );
    let inputs = [sr, si, tr, ti, epsilon];
    ensure!(
        inputs
            .iter()
            .all(|x| x.is_finite() && x.prec() <= 1_000_000)
            && epsilon > &0,
        "matrix element requires finite parameters, supported source precision, and epsilon > 0"
    );
    let work = inputs.iter().map(|x| x.prec()).fold(p, u32::max) + 32;
    let mut sum = Float::with_val(work, sr);
    sum += tr;
    let mut ar = sum.clone();
    if sum.is_finite() {
        let v = Float::with_val(work, &sum - sr);
        let mut error = Float::with_val(work, sr - Float::with_val(work, &sum - &v));
        error += Float::with_val(work, tr - &v);
        ar -= 1u32;
        ar += error;
    }
    let mut ai = Float::with_val(work, ti);
    ai -= si;
    if ar.is_zero() && ti == si {
        return Ok((Float::with_val(p, 1), Float::with_val(p, 0)));
    }
    let scale = if ar.is_finite() && ai.is_finite() {
        [
            Float::with_val(work, epsilon),
            ar.clone().abs(),
            ai.clone().abs(),
        ]
        .into_iter()
        .max_by(Float::total_cmp)
        .unwrap()
    } else {
        inputs
            .iter()
            .map(|x| Float::with_val(work, *x).abs())
            .chain([Float::with_val(work, 1)])
            .max_by(Float::total_cmp)
            .unwrap()
    };
    let r = if ar.is_finite() {
        Float::with_val(work, &ar / &scale)
    } else {
        let mut v = Float::with_val(work, sr / &scale);
        v += Float::with_val(work, tr / &scale);
        v -= Float::with_val(work, 1) / &scale;
        v
    };
    let i = if ai.is_finite() && !ai.is_zero() && ai.get_exp() != Some(rug::float::exp_min()) {
        Float::with_val(work, &ai / &scale)
    } else {
        let mut v = Float::with_val(work, ti / &scale);
        v -= Float::with_val(work, si / &scale);
        v
    };
    let e = Float::with_val(work, epsilon / &scale);
    ensure!(
        !e.is_zero()
            && !(ar.is_finite() && !ar.is_zero() && r.is_zero())
            && !(ti != si && i.is_zero()),
        "matrix-element scaling underflowed"
    );
    let numerator = e.clone().square();
    ensure!(!numerator.is_zero(), "matrix element is outside MPFR range");
    let abs_r = r.clone().abs();
    let abs_i = i.clone().abs();
    let mut ratio_di = Float::with_val(work, &r * &i);
    ratio_di *= -2i32;
    let mut difference = Float::with_val(work, &e - &abs_r);
    difference *= Float::with_val(work, &e + &abs_r);
    // i^2 can leave the exponent range (MPFR then rounds it to zero or to the
    // least positive value) exactly when 2*exp(i) <= emin; decide by exponent,
    // not by inspecting the rounded square.
    let square_underflows = i
        .get_exp()
        .is_some_and(|exponent| i64::from(exponent) * 2 <= i64::from(rug::float::exp_min()) + 2);
    if abs_i < e
        && !ratio_di.is_zero()
        && square_underflows
        && difference.clone().abs() < ratio_di.clone().abs()
    {
        // As in the binary64 path: i^2 underflows the exponent range while
        // e^2 - r^2 does not dominate Im(D), so form rho = Re(D)/Im(D) =
        // (e^2 - r^2)/Im(D) + i/(-2r) without squaring i.
        let di = ratio_di;
        let mut rho = Float::with_val(work, &difference / &di);
        let mut tail = Float::with_val(work, &i / &r);
        tail /= -2i32;
        rho += tail;
        let mut damping = rho.clone().square();
        damping += 1u32;
        let mut base = Float::with_val(work, &numerator / &di);
        base /= &damping;
        let real = Float::with_val(p, &base * &rho);
        let imaginary = Float::with_val(p, -base);
        ensure!(
            real.is_finite()
                && imaginary.is_finite()
                && !imaginary.is_zero()
                && !(!rho.is_zero() && real.is_zero()),
            "matrix element is outside MPFR range"
        );
        return Ok((real, imaginary));
    }
    let (large, small) = if abs_i >= e {
        (&abs_i, &e)
    } else {
        (&e, &abs_i)
    };
    let mut dr = Float::with_val(work, large - &abs_r);
    dr *= Float::with_val(work, large + &abs_r);
    dr += small.clone().square();
    let mut di = Float::with_val(work, &r * &i);
    di *= -2i32;
    let denominator_scale = dr.clone().abs().max(&di.clone().abs());
    ensure!(
        denominator_scale.is_finite() && !denominator_scale.is_zero(),
        "matrix-element pole or unrepresentable denominator"
    );
    dr /= &denominator_scale;
    di /= &denominator_scale;
    let mut norm = dr.clone().square();
    norm += di.clone().square();
    let mut real = Float::with_val(work, &dr / &norm);
    real *= &numerator;
    real /= &denominator_scale;
    let mut imaginary = Float::with_val(work, &di / &norm);
    imaginary *= &numerator;
    imaginary /= &denominator_scale;
    imaginary = -imaginary;
    let real = Float::with_val(p, real);
    let imaginary = Float::with_val(p, imaginary);
    ensure!(
        real.is_finite()
            && imaginary.is_finite()
            && !(!dr.is_zero() && real.is_zero())
            && !(!di.is_zero() && imaginary.is_zero()),
        "matrix element is outside MPFR range"
    );
    Ok((real, imaginary))
}
