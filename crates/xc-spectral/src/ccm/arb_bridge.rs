//! Minimal safe ownership boundary for certified FLINT/Arb special functions.

use anyhow::{bail, Result};
use rug::Float;
use std::ffi::{c_char, c_int, c_long, c_void, CStr, CString};
use xc_numerics::mpfr_interval::MpfrInterval;

unsafe extern "C" {
    fn xc_arb_flint_version() -> *const c_char;
    fn xc_arb_complex_digamma_interval(
        out_re_lower: *mut c_void,
        out_re_upper: *mut c_void,
        out_im_lower: *mut c_void,
        out_im_upper: *mut c_void,
        in_re_lower: *const c_void,
        in_re_upper: *const c_void,
        in_im_lower: *const c_void,
        in_im_upper: *const c_void,
        precision: c_long,
    );
    fn xc_arb_complex_trigamma_interval(
        out_re_lower: *mut c_void,
        out_re_upper: *mut c_void,
        out_im_lower: *mut c_void,
        out_im_upper: *mut c_void,
        in_re_lower: *const c_void,
        in_re_upper: *const c_void,
        in_im_lower: *const c_void,
        in_im_upper: *const c_void,
        precision: c_long,
    );
    fn xc_flint_rational_polynomial_root_count(
        out_count: *mut c_long,
        out_square_free: *mut c_int,
        out_lowers: *mut *mut c_void,
        out_uppers: *mut *mut c_void,
        output_capacity: c_long,
        output_precision: c_long,
        coefficients: *const *const c_char,
        coefficient_count: c_long,
        lower: *const c_char,
        upper: *const c_char,
    ) -> c_int;
}

pub fn rational_polynomial_root_count(
    coefficients_ascending: &[rug::Rational],
    lower: &rug::Rational,
    upper: &rug::Rational,
) -> Result<(usize, bool)> {
    if coefficients_ascending.len() < 2 || lower >= upper {
        bail!("FLINT root count requires a nonconstant polynomial and lower < upper");
    }
    let encoded = coefficients_ascending
        .iter()
        .map(|value| CString::new(value.to_string()).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let pointers = encoded
        .iter()
        .map(|value| value.as_ptr())
        .collect::<Vec<_>>();
    let lower = CString::new(lower.to_string())?;
    let upper = CString::new(upper.to_string())?;
    let mut count: c_long = 0;
    let mut square_free: c_int = 0;
    let status = unsafe {
        xc_flint_rational_polynomial_root_count(
            &mut count,
            &mut square_free,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            128,
            pointers.as_ptr(),
            pointers.len() as c_long,
            lower.as_ptr(),
            upper.as_ptr(),
        )
    };
    match status {
        0 if count >= 0 => Ok((count as usize, square_free != 0)),
        2 => bail!("FLINT root-count window has a polynomial root on its boundary"),
        3 => bail!("FLINT rejected an exact rational polynomial or boundary"),
        5 => bail!("FLINT root isolation requires a square-free polynomial"),
        6 => bail!("Arb could not separate a nonreal root ball from the real axis"),
        7 => bail!("Arb isolated more real roots than the supplied output capacity"),
        _ => bail!("FLINT exact root count failed with status {status}"),
    }
}

pub fn rational_polynomial_real_roots(
    coefficients_ascending: &[rug::Rational],
    lower: &rug::Rational,
    upper: &rug::Rational,
    precision_bits: u32,
) -> Result<(Vec<MpfrInterval>, bool)> {
    if coefficients_ascending.len() < 2 || lower >= upper || precision_bits <= 64 {
        bail!("FLINT root isolation requires a nonconstant polynomial, lower < upper, and HP precision");
    }
    let encoded = coefficients_ascending
        .iter()
        .map(|value| CString::new(value.to_string()).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let pointers = encoded
        .iter()
        .map(|value| value.as_ptr())
        .collect::<Vec<_>>();
    let lower_text = CString::new(lower.to_string())?;
    let upper_text = CString::new(upper.to_string())?;
    let capacity = coefficients_ascending.len() - 1;
    let mut lowers = (0..capacity)
        .map(|_| Float::with_val(precision_bits, 0))
        .collect::<Vec<_>>();
    let mut uppers = (0..capacity)
        .map(|_| Float::with_val(precision_bits, 0))
        .collect::<Vec<_>>();
    let mut lower_pointers = lowers
        .iter_mut()
        .map(|value| value.as_raw_mut().cast())
        .collect::<Vec<*mut c_void>>();
    let mut upper_pointers = uppers
        .iter_mut()
        .map(|value| value.as_raw_mut().cast())
        .collect::<Vec<*mut c_void>>();
    let mut count: c_long = 0;
    let mut square_free: c_int = 0;
    let status = unsafe {
        xc_flint_rational_polynomial_root_count(
            &mut count,
            &mut square_free,
            lower_pointers.as_mut_ptr(),
            upper_pointers.as_mut_ptr(),
            capacity as c_long,
            precision_bits as c_long,
            pointers.as_ptr(),
            pointers.len() as c_long,
            lower_text.as_ptr(),
            upper_text.as_ptr(),
        )
    };
    if status != 0 || count < 0 || count as usize > capacity {
        bail!("FLINT/Arb exact root isolation failed with status {status}");
    }
    lowers.truncate(count as usize);
    uppers.truncate(count as usize);
    let roots = lowers
        .into_iter()
        .zip(uppers)
        .map(|(lower, upper)| MpfrInterval::new(lower, upper).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    Ok((roots, square_free != 0))
}

fn evaluate(
    real: &MpfrInterval,
    imaginary: &MpfrInterval,
    trigamma: bool,
) -> Result<(MpfrInterval, MpfrInterval)> {
    if real.precision() != imaginary.precision() {
        bail!("Arb complex input intervals have different precision");
    }
    let precision = real.precision();
    let mut re_lower = Float::with_val(precision, 0);
    let mut re_upper = Float::with_val(precision, 0);
    let mut im_lower = Float::with_val(precision, 0);
    let mut im_upper = Float::with_val(precision, 0);
    unsafe {
        let function = if trigamma {
            xc_arb_complex_trigamma_interval
        } else {
            xc_arb_complex_digamma_interval
        };
        function(
            re_lower.as_raw_mut().cast(),
            re_upper.as_raw_mut().cast(),
            im_lower.as_raw_mut().cast(),
            im_upper.as_raw_mut().cast(),
            real.lower().as_raw().cast(),
            real.upper().as_raw().cast(),
            imaginary.lower().as_raw().cast(),
            imaginary.upper().as_raw().cast(),
            precision as c_long,
        );
    }
    Ok((
        MpfrInterval::new(re_lower, re_upper)?,
        MpfrInterval::new(im_lower, im_upper)?,
    ))
}

pub fn backend_version() -> &'static str {
    unsafe {
        CStr::from_ptr(xc_arb_flint_version())
            .to_str()
            .expect("FLINT_VERSION is static ASCII")
    }
}

pub fn complex_digamma(
    real: &MpfrInterval,
    imaginary: &MpfrInterval,
) -> Result<(MpfrInterval, MpfrInterval)> {
    evaluate(real, imaginary, false)
}

pub fn complex_trigamma(
    real: &MpfrInterval,
    imaginary: &MpfrInterval,
) -> Result<(MpfrInterval, MpfrInterval)> {
    evaluate(real, imaginary, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_digamma_and_trigamma_have_expected_signs() {
        let p = 128;
        let quarter = MpfrInterval::from_rational(&rug::Rational::from((1, 4)), p);
        let zero = MpfrInterval::from_i64(0, p);
        let (digamma, digamma_im) = complex_digamma(&quarter, &zero).unwrap();
        let (trigamma, trigamma_im) = complex_trigamma(&quarter, &zero).unwrap();
        assert!(digamma.upper() < &Float::with_val(p, 0));
        assert!(trigamma.is_strictly_positive());
        assert!(digamma_im.lower() <= &Float::with_val(p, 0));
        assert!(digamma_im.upper() >= &Float::with_val(p, 0));
        assert!(trigamma_im.lower() <= &Float::with_val(p, 0));
        assert!(trigamma_im.upper() >= &Float::with_val(p, 0));
        assert!(!backend_version().is_empty());
    }

    #[test]
    fn exact_flint_root_count_uses_open_rational_window() {
        // x^3 - x has roots -1, 0, 1; only 0 and 1 lie in (-1/2, 3/2).
        let coefficients = [0, -1, 0, 1]
            .into_iter()
            .map(rug::Rational::from)
            .collect::<Vec<_>>();
        let result = rational_polynomial_root_count(
            &coefficients,
            &rug::Rational::from((-1, 2)),
            &rug::Rational::from((3, 2)),
        )
        .unwrap();
        assert_eq!(result, (2, true));
    }
}

unsafe extern "C" {
    fn xc_arb_finite_transform(
        re_lo: *mut c_void,
        re_hi: *mut c_void,
        im_lo: *mut c_void,
        im_hi: *mut c_void,
        dr_lo: *mut c_void,
        dr_hi: *mut c_void,
        di_lo: *mut c_void,
        di_hi: *mut c_void,
        zrl: *const c_void,
        zrh: *const c_void,
        zil: *const c_void,
        zih: *const c_void,
        cutoff: *const c_char,
        coefficients: *const *const c_void,
        count: c_long,
        precision: c_long,
    ) -> std::ffi::c_int;
    fn xc_arb_argument(
        lo: *mut c_void,
        hi: *mut c_void,
        rl: *const c_void,
        rh: *const c_void,
        il: *const c_void,
        ih: *const c_void,
        precision: c_long,
    );
}
pub(crate) fn finite_transform(
    cutoff: &str,
    coefficients: &[Float],
    re: &MpfrInterval,
    im: &MpfrInterval,
) -> Result<(MpfrInterval, MpfrInterval, MpfrInterval, MpfrInterval)> {
    if re.precision() != im.precision()
        || coefficients.is_empty()
        || coefficients.len().is_multiple_of(2)
    {
        bail!("invalid finite transform dimensions");
    }
    let p = re.precision();
    let c = CString::new(cutoff)?;
    let pointers = coefficients
        .iter()
        .map(|v| v.as_raw().cast())
        .collect::<Vec<*const c_void>>();
    let mut values = (0..8).map(|_| Float::with_val(p, 0)).collect::<Vec<_>>();
    let out = values
        .iter_mut()
        .map(|v| v.as_raw_mut().cast())
        .collect::<Vec<*mut c_void>>();
    let status = unsafe {
        xc_arb_finite_transform(
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
            out[5],
            out[6],
            out[7],
            re.lower().as_raw().cast(),
            re.upper().as_raw().cast(),
            im.lower().as_raw().cast(),
            im.upper().as_raw().cast(),
            c.as_ptr(),
            pointers.as_ptr(),
            pointers.len() as c_long,
            p as c_long,
        )
    };
    if status != 1 {
        bail!("finite Arb transform input or enclosure unresolved");
    }
    Ok((
        MpfrInterval::new(values[0].clone(), values[1].clone())?,
        MpfrInterval::new(values[2].clone(), values[3].clone())?,
        MpfrInterval::new(values[4].clone(), values[5].clone())?,
        MpfrInterval::new(values[6].clone(), values[7].clone())?,
    ))
}
pub(crate) fn argument(re: &MpfrInterval, im: &MpfrInterval) -> Result<MpfrInterval> {
    if re.precision() != im.precision() {
        bail!("argument precision mismatch");
    }
    let p = re.precision();
    let mut lo = Float::with_val(p, 0);
    let mut hi = Float::with_val(p, 0);
    unsafe {
        xc_arb_argument(
            lo.as_raw_mut().cast(),
            hi.as_raw_mut().cast(),
            re.lower().as_raw().cast(),
            re.upper().as_raw().cast(),
            im.lower().as_raw().cast(),
            im.upper().as_raw().cast(),
            p as c_long,
        );
    }
    Ok(MpfrInterval::new(lo, hi)?)
}

#[cfg(test)]
mod finite_transform_failure_tests {
    use super::*;

    #[test]
    fn near_carrier_rectangles_stay_narrow_and_contain_point_evaluations() {
        let p = 256;
        let coefficients = [2, -3, 1].map(|x| Float::with_val(p, x));
        let half = Float::with_val(p, 13).ln() / 2;
        for mode in -1..=1 {
            // Include exact zero, nonzero centers much closer to the carrier
            // than the rectangle radius, tiny rectangles, and both sides of
            // the |q|=1 branch boundary. Sample all corners, edges and center.
            for (offset, radius_bits) in [
                (Float::with_val(p, 0), 20),
                (Float::with_val(p, 1) >> 40, 20),
                (Float::with_val(p, 1) >> 100, 20),
                (Float::with_val(p, 1) >> 80, 180),
                (Float::with_val(p, 0.99) / &half, 20),
                (Float::with_val(p, 1.01) / &half, 20),
            ] {
                let center = -Float::with_val(p, rug::float::Constant::Pi) * mode / &half + offset;
                let radius = Float::with_val(p, 1) >> radius_bits;
                let imaginary_center = Float::with_val(p, 1) >> 100;
                let re = MpfrInterval::new(
                    Float::with_val(p, &center) - &radius,
                    Float::with_val(p, &center) + &radius,
                )
                .unwrap();
                let im = MpfrInterval::new(
                    Float::with_val(p, &imaginary_center) - &radius,
                    Float::with_val(p, &imaginary_center) + &radius,
                )
                .unwrap();
                let enclosure = finite_transform("13", &coefficients, &re, &im).unwrap();
                for interval in [&enclosure.0, &enclosure.1, &enclosure.2, &enclosure.3] {
                    assert!(Float::with_val(p, interval.upper()) - interval.lower()
                        < Float::with_val(p, &radius) * 100,
                        "near-carrier enclosure lost useful precision: mode={mode}, radius=2^-{radius_bits}");
                }
                for x in [re.lower(), &center, re.upper()] {
                    for y in [im.lower(), &imaginary_center, im.upper()] {
                        let point = finite_transform(
                            "13",
                            &coefficients,
                            &MpfrInterval::point(Float::with_val(2 * p, x)),
                            &MpfrInterval::point(Float::with_val(2 * p, y)),
                        )
                        .unwrap();
                        for (outer, inner) in [
                            (&enclosure.0, &point.0),
                            (&enclosure.1, &point.1),
                            (&enclosure.2, &point.2),
                            (&enclosure.3, &point.3),
                        ] {
                            assert!(outer.lower() <= inner.lower() && outer.upper() >= inner.upper(),
                                "near-carrier rectangle does not contain independent scalar enclosure");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tiny_nonzero_scalar_carrier_preserves_derivative_relative_accuracy() {
        let p = 256;
        let x = Float::with_val(p, 1) >> 1000;
        let value = finite_transform(
            "13",
            &[Float::with_val(p, 1)],
            &MpfrInterval::point(x),
            &MpfrInterval::from_i64(0, p),
        )
        .unwrap();
        assert!(value.2.upper() < &0);
        let width = Float::with_val(p, value.2.upper()) - value.2.lower();
        assert!(width < Float::with_val(p, value.2.upper()).abs() >> 200);
    }

    #[test]
    fn centered_taylor_retains_precision_on_narrow_rectangles() {
        let p = 256;
        let left = Float::with_val(p, 70);
        let right = Float::with_val(p, &left) + (Float::with_val(p, 1) >> 120);
        let re = MpfrInterval::new(left, right).unwrap();
        let im = MpfrInterval::point(Float::with_val(p, -1));
        let value = finite_transform("13", &[Float::with_val(p, 1)], &re, &im).unwrap();
        for interval in [value.0, value.1, value.2, value.3] {
            let width = Float::with_val(p, interval.upper()) - interval.lower();
            assert!(
                width < Float::with_val(p, 1) >> 100,
                "Taylor enclosure lost arithmetic precision"
            );
        }
    }

    #[test]
    fn centered_taylor_rectangle_contains_point_values_and_derivatives() {
        let p = 256;
        let coefficients = [-1, 2, 7, 2, -1].map(|x| Float::with_val(p, x));
        for (left, right, bottom, top) in [
            (-1., 1., -0.2, 0.2),
            (12., 12.5, -1., -0.5),
            (0., 0., -0.1, 0.1),
        ] {
            let re =
                MpfrInterval::new(Float::with_val(p, left), Float::with_val(p, right)).unwrap();
            let im =
                MpfrInterval::new(Float::with_val(p, bottom), Float::with_val(p, top)).unwrap();
            let enclosure = finite_transform("13", &coefficients, &re, &im).unwrap();
            for j in 0..=8 {
                // Use a tighter independent point enclosure; equal-precision
                // intervals need not contain each other around exact symmetry zeros.
                let x = Float::with_val(2 * p, left + (right - left) * j as f64 / 8.);
                let y = Float::with_val(2 * p, bottom + (top - bottom) * j as f64 / 8.);
                let point = finite_transform(
                    "13",
                    &coefficients,
                    &MpfrInterval::point(x),
                    &MpfrInterval::point(y),
                )
                .unwrap();
                for (outer, inner) in [
                    (&enclosure.0, &point.0),
                    (&enclosure.1, &point.1),
                    (&enclosure.2, &point.2),
                    (&enclosure.3, &point.3),
                ] {
                    assert!(outer.lower() <= inner.lower() && outer.upper() >= inner.upper(),
                        "rectangle=({left},{right},{bottom},{top}), point={j}; outer=[{},{}], point=[{},{}]",
                        outer.lower().to_string_radix(10,Some(12)), outer.upper().to_string_radix(10,Some(12)),
                        inner.lower().to_string_radix(10,Some(12)), inner.upper().to_string_radix(10,Some(12)));
                }
            }
        }
    }
    #[test]
    fn invalid_cutoffs_and_zero_or_nan_norm_return_errors() {
        let z = MpfrInterval::from_i64(0, 192);
        for cutoff in ["not-a-number", "0", "1", "-2"] {
            assert!(finite_transform(cutoff, &[Float::with_val(192, 1)], &z, &z).is_err());
        }
        assert!(finite_transform("9", &[Float::with_val(192, 0)], &z, &z).is_err());
        assert!(finite_transform(
            "9",
            &[Float::with_val(192, rug::float::Special::Nan)],
            &z,
            &z
        )
        .is_err());
        assert!(finite_transform("9", &[Float::with_val(192, 1)], &z, &z).is_ok());
    }
}
