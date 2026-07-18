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
