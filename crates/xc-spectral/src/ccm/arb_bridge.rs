//! Minimal safe ownership boundary for certified FLINT/Arb special functions.

use anyhow::{bail, Result};
use rug::Float;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use xc_numerics::mpfr_interval::MpfrInterval;

// FLINT slong follows the machine word width, including LLP64 Windows where
// C long is only 32 bits. The C bridge asserts this ABI contract at compile time.
type FlintSLong = isize;

unsafe extern "C" {
    fn xc_arb_flint_version() -> *const c_char;
    #[cfg(test)]
    fn xc_arb_slong_size() -> usize;
    fn xc_arb_complex_digamma_interval(
        out_re_lower: *mut c_void,
        out_re_upper: *mut c_void,
        out_im_lower: *mut c_void,
        out_im_upper: *mut c_void,
        in_re_lower: *const c_void,
        in_re_upper: *const c_void,
        in_im_lower: *const c_void,
        in_im_upper: *const c_void,
        precision: FlintSLong,
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
        precision: FlintSLong,
    );
    fn xc_flint_rational_polynomial_root_count(
        out_count: *mut FlintSLong,
        out_square_free: *mut c_int,
        out_lowers: *mut *mut c_void,
        out_uppers: *mut *mut c_void,
        output_capacity: FlintSLong,
        output_precision: FlintSLong,
        coefficients: *const *const c_char,
        coefficient_count: FlintSLong,
        lower: *const c_char,
        upper: *const c_char,
    ) -> c_int;
}

// Numerator construction has a separate ABI from transform evaluation. GMP
// rationals cross this boundary exactly, using the same system GMP already
// required by the MPFR/FLINT bridge; no decimal conversion or rounding occurs.
unsafe extern "C" {
    fn xc_flint_exact_secular_numerator(
        out_coefficients: *mut *mut c_void,
        output_capacity: FlintSLong,
        poles: *const *const c_void,
        weights: *const *const c_void,
        count: FlintSLong,
        weight_denominator_bits: FlintSLong,
        total_denominator_bits: FlintSLong,
    ) -> c_int;
}

/// The caller admits the exact-discovery workspace before entering this
/// function. Only nonzero residues contribute poles to the numerator.
pub(in crate::ccm) fn exact_secular_numerator(
    poles: &[Float],
    weights: &[Float],
    weight_denominator_bits: u128,
    total_denominator_bits: u128,
) -> Result<Vec<rug::Rational>> {
    if poles.len() != weights.len()
        || poles.is_empty()
        || poles.len() > 8193
        || poles.iter().chain(weights).any(|point| !point.is_finite())
        || weight_denominator_bits > total_denominator_bits
    {
        bail!("invalid exact secular numerator source");
    }
    let weight_bits = FlintSLong::try_from(weight_denominator_bits)?;
    let total_bits = FlintSLong::try_from(total_denominator_bits)?;
    let pole_pointers = poles
        .iter()
        .zip(weights)
        .filter(|(_, weight)| !weight.is_zero())
        .map(|(pole, _)| pole.as_raw().cast())
        .collect::<Vec<*const c_void>>();
    let weight_pointers = weights
        .iter()
        .filter(|weight| !weight.is_zero())
        .map(|weight| weight.as_raw().cast())
        .collect::<Vec<*const c_void>>();
    if pole_pointers.is_empty() {
        bail!("the secular point source is identically zero");
    }
    let mut coefficients = vec![rug::Rational::from(0); pole_pointers.len()];
    let mut outputs = coefficients
        .iter_mut()
        .map(|value| value.as_raw_mut().cast())
        .collect::<Vec<*mut c_void>>();
    let count = FlintSLong::try_from(pole_pointers.len())?;
    let status = unsafe {
        xc_flint_exact_secular_numerator(
            outputs.as_mut_ptr(),
            count,
            pole_pointers.as_ptr(),
            weight_pointers.as_ptr(),
            count,
            weight_bits,
            total_bits,
        )
    };
    if status != 0 {
        bail!("FLINT exact secular numerator failed with status {status}");
    }
    Ok(coefficients)
}

pub fn rational_polynomial_root_count(
    coefficients_ascending: &[rug::Rational],
    lower: &rug::Rational,
    upper: &rug::Rational,
) -> Result<(usize, bool)> {
    if coefficients_ascending.len() < 2
        || lower >= upper
        || coefficients_ascending.iter().skip(1).all(|x| x == &0)
    {
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
    let mut count: FlintSLong = 0;
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
            FlintSLong::try_from(pointers.len())?,
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

#[derive(Debug)]
struct RootIsolationFailure {
    status: i32,
}
impl std::fmt::Display for RootIsolationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FLINT/Arb exact root isolation failed with status {}",
            self.status
        )
    }
}
impl std::error::Error for RootIsolationFailure {}
pub(in crate::ccm) fn isolation_needs_more_precision(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<RootIsolationFailure>()
        .is_some_and(|failure| matches!(failure.status, 2 | 6))
}

pub fn rational_polynomial_real_roots(
    coefficients_ascending: &[rug::Rational],
    lower: &rug::Rational,
    upper: &rug::Rational,
    precision_bits: u32,
) -> Result<(Vec<MpfrInterval>, bool)> {
    if coefficients_ascending.len() < 2
        || lower >= upper
        || !(65..=1_000_000).contains(&precision_bits)
        || coefficients_ascending.iter().skip(1).all(|x| x == &0)
    {
        bail!(
            "FLINT root isolation requires a nonconstant polynomial, lower < upper, and HP precision"
        );
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
    let mut count: FlintSLong = 0;
    let mut square_free: c_int = 0;
    let status = unsafe {
        xc_flint_rational_polynomial_root_count(
            &mut count,
            &mut square_free,
            lower_pointers.as_mut_ptr(),
            upper_pointers.as_mut_ptr(),
            FlintSLong::try_from(capacity)?,
            checked_ffi_precision(precision_bits)?,
            pointers.as_ptr(),
            FlintSLong::try_from(pointers.len())?,
            lower_text.as_ptr(),
            upper_text.as_ptr(),
        )
    };
    if status != 0 || count < 0 || count as usize > capacity {
        return Err(RootIsolationFailure { status }.into());
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
    real.validate()?;
    imaginary.validate()?;
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
            checked_ffi_precision(precision)?,
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
    fn rust_count_width_matches_compiled_flint_slong() {
        assert_eq!(std::mem::size_of::<FlintSLong>(), unsafe {
            xc_arb_slong_size()
        });
    }

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
    fn xc_arb_finite_transform_context_new(
        cutoff: *const c_char,
        coefficients: *const *const c_void,
        count: FlintSLong,
        precision: FlintSLong,
    ) -> *mut c_void;
    fn xc_arb_finite_transform_context_free(context: *mut c_void);
    fn xc_arb_finite_transform_prepared(
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
        context: *const c_void,
    ) -> c_int;
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
        count: FlintSLong,
        precision: FlintSLong,
    ) -> std::ffi::c_int;
    fn xc_arb_argument(
        lo: *mut c_void,
        hi: *mut c_void,
        rl: *const c_void,
        rh: *const c_void,
        il: *const c_void,
        ih: *const c_void,
        precision: FlintSLong,
    );
}

/// Immutable source- and precision-bound native balls. No source pointers survive
/// construction, and each evaluation allocates its own mutable Arb workspace.
pub(crate) struct PreparedFiniteTransform {
    context: std::ptr::NonNull<c_void>,
    precision: u32,
    resources: Option<xc_core::RunResources>,
    // Released after the native context by the explicit Drop implementation.
    _reservation: Option<xc_core::MemoryReservation>,
}

// SAFETY: C exposes the owned context only through const reads during evaluation;
// its mutable temporaries and output slots belong exclusively to each call.
unsafe impl Send for PreparedFiniteTransform {}
unsafe impl Sync for PreparedFiniteTransform {}

impl Drop for PreparedFiniteTransform {
    fn drop(&mut self) {
        unsafe { xc_arb_finite_transform_context_free(self.context.as_ptr()) };
    }
}

impl PreparedFiniteTransform {
    pub(crate) fn new(cutoff: &str, coefficients: &[Float], precision: u32) -> Result<Self> {
        if coefficients.iter().any(|value| !value.is_finite()) {
            bail!("finite transform coefficients must be finite");
        }
        if precision < 2 || coefficients.is_empty() || coefficients.len().is_multiple_of(2) {
            bail!("invalid finite transform dimensions");
        }
        let count = FlintSLong::try_from(coefficients.len())?;
        let ffi_precision = checked_ffi_precision(precision)?;
        let resources = xc_core::RunResources::current();
        // Three mode arrays, 48 powers and scalar balls, plus native setup
        // temporaries. The factor four covers Arb objects, limbs and allocator
        // overhead; reservations precede both Rust and native allocations.
        let ball_bytes = 128u128 + u128::from(precision).div_ceil(64) * 8;
        let retained_bytes = u64::try_from(
            (coefficients.len() as u128 * 3 + 64) * ball_bytes * 4
                + coefficients.len() as u128 * std::mem::size_of::<*const c_void>() as u128
                + cutoff.len() as u128
                + 4096,
        )?;
        let reservation = resources
            .as_ref()
            .map(|resources| resources.try_reserve_retained(retained_bytes))
            .transpose()?;
        let c = CString::new(cutoff)?;
        let pointers = coefficients
            .iter()
            .map(|value| value.as_raw().cast())
            .collect::<Vec<*const c_void>>();
        let context = unsafe {
            xc_arb_finite_transform_context_new(c.as_ptr(), pointers.as_ptr(), count, ffi_precision)
        };
        let Some(context) = std::ptr::NonNull::new(context) else {
            bail!("finite Arb transform input or enclosure unresolved");
        };
        Ok(Self {
            context,
            precision,
            resources,
            _reservation: reservation,
        })
    }

    pub(crate) fn evaluate(
        &self,
        re: &MpfrInterval,
        im: &MpfrInterval,
    ) -> Result<(MpfrInterval, MpfrInterval, MpfrInterval, MpfrInterval)> {
        re.validate()?;
        im.validate()?;
        if re.precision() != im.precision() || re.precision() != self.precision {
            bail!("invalid finite transform dimensions");
        }
        // The creating run remains authoritative on Rayon workers even when
        // their thread-local RunResources scope has not been entered.
        let resources = self
            .resources
            .clone()
            .or_else(xc_core::RunResources::current);
        let workspace_bytes = u64::try_from(
            (128u128 + u128::from(self.precision).div_ceil(64) * 8) * 1024 * 4 + 4096,
        )?;
        let _workspace = resources
            .as_ref()
            .map(|resources| resources.reserve_working(workspace_bytes))
            .transpose()?;
        let mut values: [Float; 8] = std::array::from_fn(|_| Float::with_val(self.precision, 0));
        let out = values.each_mut().map(|value| value.as_raw_mut().cast());
        let status = unsafe {
            xc_arb_finite_transform_prepared(
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
                self.context.as_ptr(),
            )
        };
        if status != 1 {
            bail!("finite Arb transform input or enclosure unresolved");
        }
        let [rl, rh, il, ih, drl, drh, dil, dih] = values;
        Ok((
            MpfrInterval::new(rl, rh)?,
            MpfrInterval::new(il, ih)?,
            MpfrInterval::new(drl, drh)?,
            MpfrInterval::new(dil, dih)?,
        ))
    }
}

// Retain the original setup and arithmetic route as an independent test oracle.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn finite_transform(
    cutoff: &str,
    coefficients: &[Float],
    re: &MpfrInterval,
    im: &MpfrInterval,
) -> Result<(MpfrInterval, MpfrInterval, MpfrInterval, MpfrInterval)> {
    re.validate()?;
    im.validate()?;
    if coefficients.iter().any(|value| !value.is_finite()) {
        bail!("finite transform coefficients must be finite");
    }
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
            FlintSLong::try_from(pointers.len())?,
            checked_ffi_precision(p)?,
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
    re.validate()?;
    im.validate()?;
    if re.contains_zero() && im.contains_zero() {
        bail!("complex argument is undefined at zero");
    }
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
            checked_ffi_precision(p)?,
        );
    }
    Ok(MpfrInterval::new(lo, hi)?)
}

#[cfg(test)]
mod finite_transform_failure_tests {
    use super::*;

    type TransformValue = (MpfrInterval, MpfrInterval, MpfrInterval, MpfrInterval);

    fn assert_same_endpoints(actual: &TransformValue, expected: &TransformValue) {
        for (actual, expected) in [
            (&actual.0, &expected.0),
            (&actual.1, &expected.1),
            (&actual.2, &expected.2),
            (&actual.3, &expected.3),
        ] {
            assert_eq!(actual.precision(), expected.precision());
            assert_eq!(actual.lower().as_ord(), expected.lower().as_ord());
            assert_eq!(actual.upper().as_ord(), expected.upper().as_ord());
        }
    }

    #[test]
    fn prepared_transform_matches_one_shot_endpoints_at_every_carrier_branch() {
        for p in [32, 64, 128, 256, 512] {
            for count in [1, 3, 9] {
                let coefficients = (0..count)
                    .map(|j| Float::with_val(p + 19, (j * 7 % 17) - 8) / 17)
                    .collect::<Vec<_>>();
                let prepared = PreparedFiniteTransform::new("13", &coefficients, p).unwrap();
                let half = Float::with_val(p, 13).ln() / 2;
                let radius = Float::with_val(p, 1) >> (p / 2);
                let mut boxes = vec![
                    (MpfrInterval::from_i64(0, p), MpfrInterval::from_i64(0, p)),
                    (MpfrInterval::from_i64(70, p), MpfrInterval::from_i64(-1, p)),
                    (
                        MpfrInterval::point(Float::with_val(p, 1) >> 1000),
                        MpfrInterval::from_i64(0, p),
                    ),
                    (
                        MpfrInterval::new(Float::with_val(p, -1), Float::with_val(p, 1)).unwrap(),
                        MpfrInterval::new(Float::with_val(p, -0.2), Float::with_val(p, 0.2))
                            .unwrap(),
                    ),
                ];
                for mode in [-(count / 2), 0, count / 2] {
                    for offset in [0., 0.99, 1.01] {
                        let center: Float = (-Float::with_val(p, rug::float::Constant::Pi) * mode
                            + Float::with_val(p, offset))
                            / &half;
                        boxes.push((
                            MpfrInterval::point(center.clone()),
                            MpfrInterval::from_i64(0, p),
                        ));
                        boxes.push((
                            MpfrInterval::new(
                                Float::with_val(p, &center) - &radius,
                                Float::with_val(p, &center) + &radius,
                            )
                            .unwrap(),
                            MpfrInterval::new(-radius.clone(), radius.clone()).unwrap(),
                        ));
                    }
                }
                for (re, im) in boxes {
                    let expected = finite_transform("13", &coefficients, &re, &im).unwrap();
                    assert_same_endpoints(&prepared.evaluate(&re, &im).unwrap(), &expected);
                }
            }
        }
    }

    #[test]
    fn prepared_transform_owns_source_and_isolates_precision_and_concurrent_workspaces() {
        let p = 192;
        let mut source = [-2, 3, 5, 7, -11].map(|value| Float::with_val(p, value));
        let original = source.clone();
        let first = PreparedFiniteTransform::new("13", &source, p).unwrap();
        source[0] *= 17;
        let second = PreparedFiniteTransform::new("17", &source, p).unwrap();
        let third = PreparedFiniteTransform::new("13", &original, 2 * p).unwrap();
        // Exercise constant caches warmed at a much higher precision between
        // preparation and one-shot reuse on this thread.
        let zero = MpfrInterval::from_i64(0, p);
        let before_warming = first.evaluate(&zero, &zero).unwrap();
        let high_zero = MpfrInterval::from_i64(0, 4096);
        finite_transform("13", &[Float::with_val(4096, 1)], &high_zero, &high_zero).unwrap();
        assert_same_endpoints(&first.evaluate(&zero, &zero).unwrap(), &before_warming);
        assert_same_endpoints(
            &first.evaluate(&zero, &zero).unwrap(),
            &finite_transform("13", &original, &zero, &zero).unwrap(),
        );
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    use rayon::prelude::*;
                    (0..24).into_par_iter().for_each(|j| {
                        let re = MpfrInterval::new(
                            Float::with_val(p, j) / 17,
                            Float::with_val(p, j + j % 2) / 17,
                        )
                        .unwrap();
                        let im = MpfrInterval::from_i64(-1, p);
                        assert_same_endpoints(
                            &first.evaluate(&re, &im).unwrap(),
                            &finite_transform("13", &original, &re, &im).unwrap(),
                        );
                        assert_same_endpoints(
                            &second.evaluate(&re, &im).unwrap(),
                            &finite_transform("17", &source, &re, &im).unwrap(),
                        );
                        let high_re = MpfrInterval::new(
                            Float::with_val(2 * p, re.lower()),
                            Float::with_val(2 * p, re.upper()),
                        )
                        .unwrap();
                        let high_im = MpfrInterval::from_i64(-1, 2 * p);
                        assert_same_endpoints(
                            &third.evaluate(&high_re, &high_im).unwrap(),
                            &finite_transform("13", &original, &high_re, &high_im).unwrap(),
                        );
                    });
                });
        }
        assert!(first
            .evaluate(
                &MpfrInterval::from_i64(0, 2 * p),
                &MpfrInterval::from_i64(0, 2 * p)
            )
            .is_err());
        assert!(first
            .evaluate(
                &MpfrInterval::from_i64(0, p),
                &MpfrInterval::from_i64(0, 2 * p)
            )
            .is_err());
        drop(source);
        drop(original);
        first
            .evaluate(&MpfrInterval::from_i64(0, p), &MpfrInterval::from_i64(0, p))
            .unwrap();
    }

    #[test]
    fn prepared_transform_failures_preserve_source_validation_order_and_release_reservations() {
        let p = 128;
        let resources = xc_core::RunResources::new(8 << 20).unwrap();
        let zero = MpfrInterval::from_i64(0, p);
        let coefficients = [Float::with_val(p, 1)];
        let prepared = {
            let _scope = resources.enter();
            for (cutoff, source) in [
                ("not-a-number", vec![Float::with_val(p, 1)]),
                ("1", vec![Float::with_val(p, 1)]),
                ("9", vec![Float::with_val(p, 0)]),
                ("bad\0cutoff", vec![Float::with_val(p, 1)]),
                ("bad", vec![]),
                ("bad", vec![Float::with_val(p, 1); 2]),
                ("bad", vec![Float::with_val(p, rug::float::Special::Nan); 2]),
            ] {
                let expected = finite_transform(cutoff, &source, &zero, &zero).unwrap_err();
                let actual = PreparedFiniteTransform::new(cutoff, &source, p)
                    .err()
                    .unwrap();
                assert_eq!(actual.to_string(), expected.to_string());
                assert_eq!(resources.usage().retained_bytes, 0);
                assert_eq!(resources.usage().working_bytes, 0);
            }
            std::sync::Arc::new(PreparedFiniteTransform::new("13", &coefficients, p).unwrap())
        };
        assert!(resources.usage().retained_bytes > 0);
        let worker = prepared.clone();
        std::thread::spawn(move || {
            assert!(xc_core::RunResources::current().is_none());
            worker
                .evaluate(&MpfrInterval::from_i64(0, p), &MpfrInterval::from_i64(0, p))
                .unwrap();
        })
        .join()
        .unwrap();
        assert_eq!(resources.usage().working_bytes, 0);
        drop(prepared);
        assert_eq!(resources.usage().retained_bytes, 0);
        let bounded = xc_core::RunResources::new(128 << 10).unwrap();
        {
            let _scope = bounded.enter();
            let prepared = PreparedFiniteTransform::new("13", &coefficients, p).unwrap();
            let retained = bounded.usage().retained_bytes;
            assert!(prepared
                .evaluate(&zero, &zero)
                .unwrap_err()
                .to_string()
                .contains("run memory budget exceeded"));
            assert_eq!(bounded.usage().retained_bytes, retained);
            assert_eq!(bounded.usage().working_bytes, 0);
        }
        assert_eq!(bounded.usage().retained_bytes, 0);
        let tiny = xc_core::RunResources::new(1).unwrap();
        let _scope = tiny.enter();
        assert!(PreparedFiniteTransform::new("13", &coefficients, p).is_err());
        assert_eq!(tiny.usage().retained_bytes, 0);
        assert_eq!(tiny.usage().peak_bytes, 0);
    }

    #[test]
    #[ignore = "local bounded timing comparison; run explicitly with --nocapture"]
    fn benchmark_prepared_finite_transform() {
        for (p, calls, repeats) in [(256, 128, 5), (3386, 8, 3)] {
            let coefficients = (0..65)
                .map(|j| Float::with_val(p, (j * 7 % 31) - 15) / 17)
                .collect::<Vec<_>>();
            let start = std::time::Instant::now();
            let prepared = PreparedFiniteTransform::new("13", &coefficients, p).unwrap();
            eprintln!(
                "precision={p} prepared construction_ns={}",
                start.elapsed().as_nanos()
            );
            for rectangle in [false, true] {
                let re = MpfrInterval::new(
                    Float::with_val(p, 3),
                    Float::with_val(p, if rectangle { 3.125 } else { 3. }),
                )
                .unwrap();
                let im = MpfrInterval::from_i64(-1, p);
                let expected = finite_transform("13", &coefficients, &re, &im).unwrap();
                assert_same_endpoints(&prepared.evaluate(&re, &im).unwrap(), &expected);
                for repeat in 0..repeats {
                    for use_prepared in [repeat % 2 == 0, repeat % 2 != 0] {
                        let start = std::time::Instant::now();
                        for _ in 0..calls {
                            let value = if use_prepared {
                                prepared.evaluate(&re, &im).unwrap()
                            } else {
                                finite_transform("13", &coefficients, &re, &im).unwrap()
                            };
                            std::hint::black_box(value);
                        }
                        eprintln!("precision={p} prepared={use_prepared} rectangle={rectangle} repeat={repeat} calls={calls} elapsed_ns={}", start.elapsed().as_nanos());
                    }
                }
            }
        }
    }

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
                    assert!(
                        Float::with_val(p, interval.upper()) - interval.lower()
                            < Float::with_val(p, &radius) * 100,
                        "near-carrier enclosure lost useful precision: mode={mode}, radius=2^-{radius_bits}"
                    );
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
                            assert!(
                                outer.lower() <= inner.lower() && outer.upper() >= inner.upper(),
                                "near-carrier rectangle does not contain independent scalar enclosure"
                            );
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
                    assert!(
                        outer.lower() <= inner.lower() && outer.upper() >= inner.upper(),
                        "rectangle=({left},{right},{bottom},{top}), point={j}; outer=[{},{}], point=[{},{}]",
                        outer.lower().to_string_radix(10, Some(12)),
                        outer.upper().to_string_radix(10, Some(12)),
                        inner.lower().to_string_radix(10, Some(12)),
                        inner.upper().to_string_radix(10, Some(12))
                    );
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

// A u32 precision may exceed signed FLINT slong on a 32-bit target.
// Preserve the checked conversion on every supported target.
#[allow(clippy::unnecessary_fallible_conversions)]
fn checked_ffi_precision(precision: u32) -> anyhow::Result<FlintSLong> {
    Ok(FlintSLong::try_from(precision)?)
}
