//! Certified enclosures of a finite retained Fourier transform. This is not a
//! statement of convergence, RH, or accuracy of an uncertified source state.
/// Fourier sign shared by retained-root complex and enclosed contour producers.
pub(crate) const FOURIER_SEMANTICS: &str = "retained_fourier_minus_sign_at_original_coordinates_v1";

use super::{
    extended_research::*, research_completion::CompletionInputs, retained_evidence::*,
    state_geometry::RetainedState,
};
use anyhow::Result;

#[cfg(not(feature = "arb"))]
pub(crate) fn analyze(
    s: &RetainedState,
    _roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
    _i: Option<&CompletionInputs>,
    _source_precision: u32,
) -> Result<ExtendedAnalysis> {
    Ok(missing(report("transform_enclosure",s,o),"Arb feature required for outward-enclosed complex transforms; point samples remain available"))
}

#[cfg(feature = "arb")]
mod certified {
    use super::super::{
        arb_bridge,
        capture_runtime::{Checkpoints, Stage},
        research_completion::ContourPolicy,
    };
    use super::*;
    use anyhow::bail;
    use rug::{float::Round, Float};
    use xc_numerics::mpfr_interval::MpfrInterval as I;
    fn decimal(s: &str, p: u32) -> Result<I> {
        Ok(I::new(
            Float::with_val_round(p, Float::parse(s)?, Round::Down).0,
            Float::with_val_round(p, Float::parse(s)?, Round::Up).0,
        )?)
    }
    fn bounds(
        out: &mut std::collections::BTreeMap<String, String>,
        name: &str,
        x: &I,
    ) -> Result<()> {
        x.validate()?;
        let digits = Some((u64::from(x.precision()) * 30103 / 100000 + 10) as usize);
        out.insert(
            format!("{name}_lower"),
            x.lower().to_string_radix_round(10, digits, Round::Down),
        );
        out.insert(
            format!("{name}_upper"),
            x.upper().to_string_radix_round(10, digits, Round::Up),
        );
        Ok(())
    }
    #[test]
    fn serialized_bounds_enclose_exact_dyadic_endpoints() {
        use std::collections::BTreeMap;
        fn exact_decimal(s: &str) -> rug::Rational {
            use rug::{ops::Pow, Integer, Rational};
            let mut parts = s.split(['e', 'E']);
            let body = parts.next().unwrap();
            let exponent = parts.next().unwrap_or("0").parse::<i32>().unwrap();
            let places = body.split_once('.').map_or(0, |(_, f)| f.len() as i32);
            let digits = body.replace('.', "").parse::<Integer>().unwrap();
            let scale = exponent - places;
            let power = Integer::from(10).pow(scale.unsigned_abs());
            if scale >= 0 {
                Rational::from(digits * power)
            } else {
                Rational::from((digits, power))
            }
        }
        let mut checked = 0;
        for p in [64, 128, 256] {
            for shift in [-5000i32, 0, 5000] {
                for sign in [-1i32, 1] {
                    let x = ((Float::with_val(p, 1) / 3u32) << shift) * sign;
                    let interval = I::point(x.clone());
                    let mut output = BTreeMap::new();
                    bounds(&mut output, "value", &interval).unwrap();
                    let lower = exact_decimal(&output["value_lower"]);
                    let upper = exact_decimal(&output["value_upper"]);
                    let exact = x.to_rational().unwrap();
                    assert!(lower <= exact && exact <= upper);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 18);
    }
    fn widen(x: &I, error: &I) -> Result<I> {
        Ok(x.add(&I::new(-error.upper().clone(), error.upper().clone())?))
    }
    fn cdiv(a: &(I, I), b: &(I, I)) -> Result<(I, I)> {
        let den = b.0.square().add(&b.1.square());
        Ok((
            a.0.mul(&b.0).add(&a.1.mul(&b.1)).div(&den)?,
            a.1.mul(&b.0).sub(&a.0.mul(&b.1)).div(&den)?,
        ))
    }
    fn pair(s: &RetainedState, re: &I, im: &I) -> Result<(I, I, I, I)> {
        // The bridge is the existing plus-sign finite Fourier kernel.
        // Reflect its box and reverse its derivative to evaluate exp(-i*z*x)
        // at the original reported coordinates, including the entire contour.
        let (vr, vi, dr, di) =
            arb_bridge::finite_transform(&s.cutoff, &s.coefficients, &re.neg(), &im.neg())?;
        Ok((vr, vi, dr.neg(), di.neg()))
    }

    /// Residual-to-angle inequality against the replayed, isolated finite ground.
    /// Raw cutoff-free interval entries retain assembly uncertainty. The certificate's
    /// parity-invariance and matrix-assembly premises are explicitly inherited.
    pub(super) fn state_error(
        s: &RetainedState,
        o: &ExtensionOptions,
        c: &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
    ) -> Result<I> {
        // External certificates must bind their matrix to the named CCM
        // assembly; a self-consistent proof for another matrix is insufficient.
        let _stage = Stage::new("exact source certificate replay and reassembly");
        let check = super::super::sector_gap_certificate::verify_portable_ccm_sector_gap_certificate_with_reassembly(c);
        if !check.valid {
            bail!("source certificate verification failed: {:?}", check.errors);
        }
        state_error_from_verified_matrix(s, o, c)
    }

    pub(super) fn state_error_from_verified_matrix(
        s: &RetainedState,
        o: &ExtensionOptions,
        c: &super::super::sector_gap_certificate::PortableCcmSectorGapCertificate,
    ) -> Result<I> {
        if c.lambda_squared != s.cutoff
            || c.n_modes != s.modes
            || !c.certifies_finite_ground_state_simple
            || c.certified_finite_ground_parity != "even"
        {
            bail!("source certificate does not isolate this finite even ground");
        }
        let p = o.working_precision_bits;
        let n = s.coefficients.len();
        let v = s
            .coefficients
            .iter()
            .map(|v| I::point(Float::with_val(p, v)))
            .collect::<Vec<_>>();
        let norm = v
            .iter()
            .fold(I::from_i64(0, p), |sum, x| sum.add(&x.square()))
            .sqrt()?;
        let v = v
            .iter()
            .map(|x| x.div(&norm))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let e = decimal(&s.eigenvalue, p)?;
        let next_even =
            I::from_rational(&xc_certify::exact::parse(&c.even_first_excited.lower)?, p);
        let next_odd = I::from_rational(&xc_certify::exact::parse(&c.odd_ground.lower)?, p);
        let next = if next_even.lower() < next_odd.lower() {
            next_even
        } else {
            next_odd
        };
        let gap = next.sub(&e);
        if !gap.is_strictly_positive() {
            bail!("source energy is not below the certified complementary spectrum");
        }
        let mut residual = I::from_i64(0, p);
        for row in 0..n {
            let mut action = I::from_i64(0, p);
            for (col, v) in v.iter().enumerate() {
                let a = xc_certify::exact::parse_interval(&c.cutoff_free_tau[row * n + col])?;
                let a = I::new(
                    I::from_rational(a.lower(), p).lower().clone(),
                    I::from_rational(a.upper(), p).upper().clone(),
                )?;
                action = action.add(&a.mul(v));
            }
            residual = residual.add(&action.sub(&e.mul(&v[row])).square());
        }
        let angle = residual.sqrt()?.div(&gap)?;
        if angle.upper() >= &1 {
            bail!("source residual does not resolve a finite ground angle");
        }
        // The residual inequality bounds the error from above; it supplies no
        // positive lower bound, even when the trial energy is inexact.
        let upper_bound = I::from_i64(2, p).sqrt()?.mul(&angle);
        Ok(I::new(Float::with_val(p, 0), upper_bound.upper().clone())?)
    }
    type SegmentEnclosure = (I, I, I, I, I);
    #[derive(Clone)]
    struct Segment {
        a: (Float, Float),
        b: (Float, Float),
        depth: u32,
    }
    fn interval(a: &Float, b: &Float) -> Result<I> {
        Ok(I::new(a.clone().min(b), a.clone().max(b))?)
    }
    fn midpoint(a: &Float, b: &Float, p: u32) -> Float {
        (Float::with_val(p, a) + b) / 2u32
    }
    fn split(seg: &Segment, p: u32) -> (Segment, Segment) {
        let mid = (
            midpoint(&seg.a.0, &seg.b.0, p),
            midpoint(&seg.a.1, &seg.b.1, p),
        );
        (
            Segment {
                a: seg.a.clone(),
                b: mid.clone(),
                depth: seg.depth + 1,
            },
            Segment {
                a: mid,
                b: seg.b.clone(),
                depth: seg.depth + 1,
            },
        )
    }
    pub(crate) fn analyze(
        s: &RetainedState,
        roots: Option<&RetainedRoots>,
        o: &ExtensionOptions,
        input: Option<&CompletionInputs>,
        source_precision: u32,
    ) -> Result<ExtendedAnalysis> {
        let mut out = report("transform_enclosure", s, o);
        let p = o.working_precision_bits;
        let l = decimal(&s.cutoff, p)?.ln()?;
        let half = l.div(&I::from_i64(2, p))?;
        let mut source_error = None;
        if let Some(c) = input.and_then(|i| i.sector_certificate.as_ref()) {
            match state_error(s, o, c) {
                Ok(e) => {
                    bounds(&mut out.values, "unit_state_l2_error", &e)?;
                    source_error = Some(e);
                    out.reason=Some(format!("source certificate replayed; conditional on {} and the certificate's recorded cutoff-free assembly enclosure; no infinite source error claim",c.parity_invariance_premise));
                }
                Err(e) => {
                    out.outcome = "partial_unresolved".into();
                    out.reason=Some(format!("source certificate unavailable or unresolved: {e}; finite retained function still enclosed"));
                }
            }
        }
        let default_right = roots
            .and_then(|r| {
                r.dataset
                    .points
                    .iter()
                    .filter_map(|x| x.value.as_deref())
                    .filter_map(|x| {
                        scalar(x, r.dataset.precision_bits)
                            .ok()
                            .map(|v| Float::with_val(p, v))
                    })
                    .max_by(Float::total_cmp)
            })
            .unwrap_or_else(|| Float::with_val(p, 1))
            + 1u32;
        let default = ContourPolicy {
            left: "-1".into(),
            right: xc_numerics::prefix::lossless_decimal(&default_right),
            bottom: "-1".into(),
            top: "1".into(),
            maximum_depth: 20,
            maximum_segments: 4096,
        };
        let policy = input.and_then(|i| i.contour.as_ref()).unwrap_or(&default);
        // Contour endpoints are the exact dyadic numbers serialized below. No
        // claim is made about a subtly different decimal rectangle.
        let owner = if input.and_then(|i| i.contour.as_ref()).is_some() {
            source_precision
        } else {
            p
        };
        let left = Float::with_val(p, scalar(&policy.left, owner)?);
        let right = Float::with_val(p, scalar(&policy.right, owner)?);
        let bottom = Float::with_val(p, scalar(&policy.bottom, owner)?);
        let top = Float::with_val(p, scalar(&policy.top, owner)?);
        for (name, v) in [
            ("contour_left", &left),
            ("contour_right", &right),
            ("contour_bottom", &bottom),
            ("contour_top", &top),
        ] {
            put(&mut out.values, name, v);
        }
        let evaluate = |re: &I, im: &I| -> Result<(I, I, I, I)> {
            let (mut a, mut b, mut d, mut e) = pair(s, re, im)?;
            if let Some(error) = &source_error {
                let im_abs = im.lower().clone().abs().max(&im.upper().clone().abs());
                let allowance = l.sqrt()?.mul(&I::point(im_abs).mul(&half).exp()).mul(error);
                a = widen(&a, &allowance)?;
                b = widen(&b, &allowance)?;
                let slope = allowance.mul(&half);
                d = widen(&d, &slope)?;
                e = widen(&e, &slope)?;
            }
            Ok((a, b, d, e))
        };
        let zero = I::from_i64(0, p);
        let anchor = evaluate(&zero, &zero)?;
        bounds(&mut out.values, "origin_real", &anchor.0)?;
        bounds(&mut out.values, "origin_imaginary", &anchor.1)?;
        let vertices = [
            (left.clone(), bottom.clone()),
            (right.clone(), bottom),
            (right, top.clone()),
            (left, top),
        ];
        let mut pending = (0..4)
            .rev()
            .map(|j| Segment {
                a: vertices[j].clone(),
                b: vertices[(j + 1) % 4].clone(),
                depth: 0,
            })
            .collect::<Vec<_>>();
        let mut angle = I::from_i64(0, p);
        let mut unresolved_count = 0usize;
        let pi = I::pi(p);
        let store = Checkpoints::new(&(
            "finite-contour-segments-minus-fourier-v4-source-error-hull",
            FOURIER_SEMANTICS,
            source_precision,
            &s.manifest.content_digest,
            o,
            input,
        ))?;
        let _stage = Stage::new("adaptive certified contour");
        while let Some(seg) = pending.pop() {
            let mut rr = row(out.rows.len() + 1, "contour_segment");
            for (name, v) in [
                ("start_re", &seg.a.0),
                ("start_im", &seg.a.1),
                ("end_re", &seg.b.0),
                ("end_im", &seg.b.1),
            ] {
                put(&mut rr.values, name, v);
            }
            let key = serde_json::to_string(&rr.values)?;
            let evaluate_segment = || -> Result<Option<SegmentEnclosure>> {
                let (re, im, dr, di) = evaluate(
                    &interval(&seg.a.0, &seg.b.0)?,
                    &interval(&seg.a.1, &seg.b.1)?,
                )?;
                if re.contains_zero() && im.contains_zero() {
                    return Ok(None);
                }
                let a = evaluate(&I::point(seg.a.0.clone()), &I::point(seg.a.1.clone()))?;
                let b = evaluate(&I::point(seg.b.0.clone()), &I::point(seg.b.1.clone()))?;
                let ratio = cdiv(&(b.0, b.1), &(a.0, a.1))?;
                let turn = arb_bridge::argument(&ratio.0, &ratio.1)?;
                if turn.lower() <= &(-pi.lower().clone()) || turn.upper() >= pi.lower() {
                    return Ok(None);
                }
                Ok(Some((re, im, dr, di, turn)))
            };
            let saved = store.load::<AnalysisRow>(&key)?;
            if let Some(saved) = saved {
                if saved.outcome == "point_measurement" {
                    let turn = I::new(
                        scalar(&saved.values["argument_increment_lower"], p)?,
                        scalar(&saved.values["argument_increment_upper"], p)?,
                    )?;
                    angle = angle.add(&turn);
                    let mut saved = saved;
                    saved.ordinal = rr.ordinal;
                    out.rows.push(saved);
                    continue;
                }
            }
            match evaluate_segment() {
                Ok(Some((re, im, dr, di, turn))) => {
                    for (name, v) in [
                        ("value_real", re),
                        ("value_imaginary", im),
                        ("derivative_real", dr),
                        ("derivative_imaginary", di),
                        ("argument_increment", turn.clone()),
                    ] {
                        bounds(&mut rr.values, name, &v)?;
                    }
                    angle = angle.add(&turn);
                    rr.notes.push("Arb encloses the entire segment image in a convex rectangle excluding zero; endpoint argument increment is unambiguous".into());
                    if let Err(e) = store.save(&key, &rr) {
                        eprintln!("contour checkpoint unavailable: {e}");
                    }
                }
                _ => {
                    if seg.depth < policy.maximum_depth
                        && out.rows.len() + pending.len() + 2
                            < policy.maximum_segments.min(o.maximum_rows)
                    {
                        let (a, b) = split(&seg, p);
                        pending.push(b);
                        pending.push(a);
                        continue;
                    }
                    rr.outcome = "unresolved_denominator".into();
                    rr.notes.push("nonvanishing or argument increment unresolved within depth/segment budget; no count certified".into());
                    unresolved_count += 1;
                }
            }
            out.rows.push(rr);
        }
        bounds(&mut out.values, "contour_argument_sum", &angle)?;
        put(
            &mut out.values,
            "unresolved_segments",
            &Float::with_val(p, unresolved_count),
        );
        if unresolved_count == 0 {
            let winding = angle.div(&pi.mul(&I::from_i64(2, p)))?;
            bounds(&mut out.values, "winding", &winding)?;
            let lower = winding.lower().clone().ceil();
            let upper = winding.upper().clone().floor();
            if lower == upper && lower >= 0 {
                put(&mut out.values, "certified_finite_zero_count", &lower);
            } else {
                out.outcome = "partial_unresolved".into();
                out.reason = Some(
                    "boundary enclosed but unique nonnegative winding integer unresolved".into(),
                );
            }
        } else {
            out.outcome = "partial_unresolved".into();
            out.reason.get_or_insert_with(||"contour budget exhausted or boundary root; partial enclosures retained and no root count asserted".into());
        }
        if let Some(roots) = roots {
            let anchor_pair = (anchor.0.clone(), anchor.1.clone());
            for point in &roots.dataset.points {
                if out.rows.len() >= o.maximum_rows {
                    out.outcome = "partial_unresolved".into();
                    out.reason =
                        Some("root enclosure row budget reached after contour evaluation".into());
                    break;
                }
                let mut rr = row(out.rows.len() + 1, "retained_root_enclosure");
                put(
                    &mut rr.values,
                    "input_ordinal",
                    &Float::with_val(p, point.ordinal),
                );
                if let Some(t) = &point.value {
                    let re = I::from_float(&scalar(t, roots.dataset.precision_bits)?, p)?;
                    save_arithmetic_enclosure(&mut rr.values, "t", &re, p)?;
                    put(
                        &mut rr.values,
                        "input_point_precision_bits",
                        &Float::with_val(p, roots.dataset.precision_bits),
                    );
                    let (vr, vi, dr, di) = evaluate(&re, &zero)?;
                    for (name, v) in [
                        ("value_real", &vr),
                        ("value_imaginary", &vi),
                        ("derivative_real", &dr),
                        ("derivative_imaginary", &di),
                    ] {
                        bounds(&mut rr.values, name, v)?;
                    }
                    match cdiv(&(vr, vi), &anchor_pair) {
                        Ok((nr, ni)) => {
                            bounds(&mut rr.values, "normalized_real", &nr)?;
                            bounds(&mut rr.values, "normalized_imaginary", &ni)?;
                        }
                        Err(_) => {
                            rr.outcome = "unresolved_denominator".into();
                            rr.notes.push(
                                "origin normalization includes zero; raw enclosures retained"
                                    .into(),
                            );
                        }
                    }
                } else {
                    rr.outcome = "missing_input".into();
                }
                out.rows.push(rr);
            }
        }
        out.convention=if source_error.is_some(){"exp(-i*z*x); Arb finite transform and derivative enclosures with verified finite-matrix residual/gap source allowance; inherited parity and matrix-assembly premises; signed unit coefficient normalization; argument-principle count only when every segment is certified"}else{"exp(-i*z*x); Arb exact retained dyadic coefficient function, unit coefficient norm; log(C) and carriers enclosed; no eigenstate/assembly accuracy claim; argument-principle count only when every segment is certified"}.into();
        for row in &mut out.rows {
            if row.outcome == "point_measurement" {
                row.outcome = "certified_finite_enclosure".into();
            }
        }
        if out.outcome == "point_measurement" {
            out.outcome = "certified_finite_enclosure".into();
        }
        Ok(out)
    }
}
#[cfg(feature = "arb")]
pub(crate) use certified::analyze;

#[cfg(all(test, feature = "arb"))]
pub(crate) fn test_source_error_with_reassembly(
    s: &RetainedState,
    c: &super::sector_gap_certificate::PortableCcmSectorGapCertificate,
) -> Result<xc_numerics::mpfr_interval::MpfrInterval> {
    certified::state_error(s, &ExtensionOptions::for_source(s), c)
}

#[cfg(all(test, feature = "arb"))]
pub(crate) fn test_source_error(
    s: &RetainedState,
    c: &super::sector_gap_certificate::PortableCcmSectorGapCertificate,
) -> Result<xc_numerics::mpfr_interval::MpfrInterval> {
    // Exact manufactured-matrix controls exercise the residual/gap inequality
    // independently of CCM assembly. This helper exists only in unit tests.
    let check = super::sector_gap_certificate::verify_portable_ccm_sector_gap_certificate(c);
    if !check.valid {
        anyhow::bail!(
            "recorded test matrix verification failed: {:?}",
            check.errors
        );
    }
    certified::state_error_from_verified_matrix(s, &ExtensionOptions::for_source(s), c)
}
