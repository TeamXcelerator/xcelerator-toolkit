//! Directed entire finite Fourier samples; no contour or source-error certificate.
use crate::ccm::{
    capture_runtime::{row_blocks, CaptureResourcePolicy, Checkpoints},
    extended_research::{
        put, report, row, save_arithmetic_enclosure, unresolved, AnalysisRow, ExtendedAnalysis,
        ExtensionOptions,
    },
    retained_evidence::{
        finite_math::{abs, decimal, exponent, narrow, scale},
        point, scalar, RetainedRoots,
    },
    state_geometry::RetainedState,
};
use anyhow::Result;
use rug::{float::Round, Float};
use std::collections::BTreeMap;
use xc_numerics::mpfr_interval::{MpfrComplexBall as Z, MpfrInterval as I};
type Values = BTreeMap<String, I>;
fn zero(p: u32) -> I {
    I::from_i64(0, p)
}
fn constant(v: i64, p: u32) -> Result<Z> {
    Ok(Z::new(I::from_i64(v, p), zero(p))?)
}
fn real(z: &I) -> Result<Z> {
    Ok(Z::new(z.clone(), zero(z.precision()))?)
}
fn is_zero(x: &I) -> bool {
    x.lower().is_zero() && x.upper().is_zero()
}
fn times(z: &Z, x: &I) -> Result<Z> {
    Ok(Z::new(z.real().mul(x), z.imaginary().mul(x))?)
}
fn binary(z: &Z, e: i64) -> Result<Z> {
    Ok(Z::new(scale(z.real(), e)?, scale(z.imaginary(), e)?)?)
}
fn magnitude(z: &Z) -> Result<I> {
    z.validate()?;
    if is_zero(z.imaginary()) {
        return abs(z.real());
    }
    if is_zero(z.real()) {
        return abs(z.imaginary());
    }
    let e = exponent(&[z.real().clone(), z.imaginary().clone()])?;
    let reduced = binary(z, -e)?;
    scale(
        &reduced
            .real()
            .square()
            .add(&reduced.imaginary().square())
            .sqrt()?,
        e,
    )
}
fn divide(a: &Z, b: &Z) -> Result<Option<Z>> {
    a.validate()?;
    b.validate()?;
    if !b.excludes_zero() {
        return Ok(None);
    }
    let ae = exponent(&[a.real().clone(), a.imaginary().clone()])?;
    let be = exponent(&[b.real().clone(), b.imaginary().clone()])?;
    let a = binary(a, -ae)?;
    let b = binary(b, -be)?;
    let denominator = b.real().square().add(&b.imaginary().square());
    if denominator.contains_zero() {
        return Ok(None);
    }
    let numerator = a.mul(&b.conjugate())?;
    let q = Z::new(
        numerator.real().div(&denominator)?,
        numerator.imaginary().div(&denominator)?,
    )?;
    Ok(Some(binary(&q, ae - be)?))
}
fn sincos(z: &Z) -> Result<(Z, Z)> {
    let p = z.precision();
    let two = I::from_i64(2, p);
    let ep = z.imaginary().exp();
    let em = z.imaginary().neg().exp();
    let sinh = ep.sub(&em).div(&two)?;
    let cosh = ep.add(&em).div(&two)?;
    let sin = z.real().sin();
    let cos = z.real().cos();
    Ok((
        Z::new(sin.mul(&cosh), cos.mul(&sinh))?,
        Z::new(cos.mul(&cosh), sin.neg().mul(&sinh))?,
    ))
}
fn widen(z: &Z, error: &I, real_axis: bool, imaginary_axis: bool) -> Result<Z> {
    let delta = I::new(-error.upper().clone(), error.upper().clone())?;
    Ok(Z::new(
        if imaginary_axis {
            zero(z.precision())
        } else {
            z.real().add(&delta)
        },
        if real_axis {
            zero(z.precision())
        } else {
            z.imaginary().add(&delta)
        },
    )?)
}
// On |q|<=1/2, absolute tails are at most twice their next term.
// Keep -q/3 explicit; q^2/q can erase a representable small derivative.
fn small_sinc(q: &Z) -> Result<Option<(Z, Z)>> {
    let p = q.precision();
    let r = magnitude(q)?;
    if r.upper() > &Float::with_val(p, 0.5) {
        return Ok(None);
    }
    if r.upper().is_zero() {
        return Ok(Some((constant(1, p)?, constant(0, p)?)));
    }
    let square = q.mul(q)?.neg();
    let mut term = times(&square, &I::from_i64(1, p).div(&I::from_i64(6, p))?)?;
    let mut value = constant(1, p)?.add(&term)?;
    let mut dt = times(&q.neg(), &I::from_i64(1, p).div(&I::from_i64(3, p))?)?;
    let mut derivative = dt.clone();
    let tolerance = Float::with_val(p, 1) >> (p + 16);
    for k in 1..=p {
        let k = u64::from(k);
        let next = times(
            &term.mul(&square)?,
            &I::from_i64(1, p).div(&I::from_u64((2 * k + 2) * (2 * k + 3), p))?,
        )?;
        let next_d = times(
            &dt.mul(&square)?,
            &I::from_i64(1, p).div(&I::from_u64(2 * k * (2 * k + 3), p))?,
        )?;
        let error = magnitude(&next)?.mul(&I::from_i64(2, p));
        let de = magnitude(&next_d)?.mul(&I::from_i64(2, p));
        let relative = Float::with_val_round(p, de.upper() / r.upper(), Round::Up).0;
        if error.upper() <= &tolerance && relative <= tolerance {
            let real_axis = is_zero(q.imaginary());
            let imag_axis = is_zero(q.real());
            return Ok(Some((
                widen(&value, &error, real_axis || imag_axis, false)?,
                widen(&derivative, &de, real_axis, imag_axis)?,
            )));
        }
        value = value.add(&next)?;
        derivative = derivative.add(&next_d)?;
        term = next;
        dt = next_d;
    }
    Ok(None)
}
fn narrow_value(value: &I, requested: u32) -> Result<bool> {
    value.validate()?;
    let natural = if value.contains_zero() {
        Float::with_val(value.precision(), 1)
    } else {
        abs(value)?.upper().clone()
    };
    narrow(std::slice::from_ref(value), &natural, requested)
}
fn output_supported(value: &I, requested: u32) -> bool {
    value.with_precision(requested).is_ok()
        && point::output(value.midpoint_point().lower(), requested).is_ok()
}
fn narrow_values(values: &Values, requested: u32) -> Result<bool> {
    for v in values.values() {
        if !narrow_value(v, requested)? || !output_supported(v, requested) {
            return Ok(false);
        }
    }
    Ok(true)
}
struct Prepared {
    coefficients: Vec<I>,
    half: I,
    factor: I,
    anchor: I,
    precision: u32,
}
fn prepare(s: &RetainedState, p: u32) -> Result<Option<Prepared>> {
    let l = decimal(&s.cutoff, p)?.ln()?;
    if !l.is_strictly_positive() {
        return Ok(None);
    }
    let e = s
        .coefficients
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .ok_or_else(|| anyhow::anyhow!("zero complex transform source"))?;
    let coefficients = s
        .coefficients
        .iter()
        .map(|x| scale(&I::from_float(x, p)?, -i64::from(e)))
        .collect::<Result<Vec<_>>>()?;
    let q = coefficients
        .iter()
        .fold(zero(p), |sum, x| sum.add(&x.square()));
    let factor = l.sqrt()?.div(&q.sqrt()?)?.mul(&I::from_i64(
        i64::from(point::orientation(&s.coefficients, p)),
        p,
    ));
    let anchor = factor.mul(&coefficients[s.modes]);
    Ok(Some(Prepared {
        coefficients,
        half: l.div(&I::from_i64(2, p))?,
        factor,
        anchor,
        precision: p,
    }))
}
fn transform(s: &RetainedState, c: &Prepared, z: &Z) -> Result<Option<(Z, Z, I)>> {
    let p = c.precision;
    // Retained secular ordinates use exp(-i*z*x). Reflect the entire
    // complex coordinate in the existing plus kernel and apply the chain rule.
    let phase = times(&z.neg(), &c.half)?;
    let (sin, cos) = sincos(&phase)?;
    let pi = I::pi(p);
    let mut f = constant(0, p)?;
    let mut derivative = f.clone();
    let mut absolute = zero(p);
    for (index, (x, original)) in c.coefficients.iter().zip(&s.coefficients).enumerate() {
        if original.is_zero() {
            continue;
        }
        let j = i64::try_from(index)? - i64::try_from(s.modes)?;
        let q = phase.add(&real(&pi.mul(&I::from_i64(j, p)))?)?;
        let (a, b) = if magnitude(&q)?.upper() <= &Float::with_val(p, 0.5) {
            let Some((a, b)) = small_sinc(&q)? else {
                return Ok(None);
            };
            if j.unsigned_abs().is_multiple_of(2) {
                (a, b)
            } else {
                (a.neg(), b.neg())
            }
        } else {
            let Some(a) = divide(&sin, &q)? else {
                return Ok(None);
            };
            let Some(b) = divide(&cos.sub(&a)?, &q)? else {
                return Ok(None);
            };
            (a, b)
        };
        let multiplier = x.mul(&c.factor);
        let a = times(&a, &multiplier)?;
        let b = times(&b, &multiplier.mul(&c.half))?.neg();
        absolute = absolute.add(&magnitude(&a)?);
        f = f.add(&a)?;
        derivative = derivative.add(&b)?;
    }
    if is_zero(z.real()) && s.coefficients.iter().eq(s.coefficients.iter().rev()) {
        // An even entire function with real Taylor coefficients is real on
        // the imaginary axis; its derivative there is purely imaginary.
        f = Z::new(f.real().clone(), zero(p))?;
        derivative = Z::new(zero(p), derivative.imaginary().clone())?;
    }
    if is_zero(z.real()) && is_zero(z.imaginary()) {
        // Exact Fourier orthogonality fixes F(0), avoiding fictitious carrier noise.
        f = real(&c.anchor)?;
        if s.coefficients.iter().eq(s.coefficients.iter().rev()) {
            derivative = constant(0, p)?;
        }
    }
    Ok(Some((f, derivative, absolute)))
}
#[derive(Clone)]
enum Sample {
    Probe {
        ordinal: usize,
        t: Option<Float>,
        offset: i32,
        status: String,
    },
    Contour {
        side: usize,
        step: u32,
        maximum: Float,
    },
}
impl Sample {
    fn coordinate(&self, p: u32) -> Result<Option<Z>> {
        match self {
            Self::Probe { t, offset, .. } => t
                .as_ref()
                .map(|t| {
                    Ok(Z::new(
                        I::from_float(t, p)?,
                        I::from_i64(i64::from(*offset), p).div(&I::from_i64(4, p))?,
                    )?)
                })
                .transpose(),
            Self::Contour {
                side,
                step,
                maximum,
            } => {
                let left = I::from_i64(-1, p);
                let right = I::from_float(maximum, p)?.add(&I::from_i64(1, p));
                let width = right.sub(&left);
                let a = I::from_u64(u64::from(*step), p).div(&I::from_i64(16, p))?;
                let (re, im) = match side {
                    0 => (left.add(&width.mul(&a)), I::from_i64(-1, p)),
                    1 => (right, I::from_i64(-1, p).add(&a.mul(&I::from_i64(2, p)))),
                    2 => (right.sub(&width.mul(&a)), I::from_i64(1, p)),
                    _ => (left, I::from_i64(1, p).sub(&a.mul(&I::from_i64(2, p)))),
                };
                Ok(Some(Z::new(re, im)?))
            }
        }
    }
    fn ordinal(&self) -> usize {
        if let Self::Probe { ordinal, .. } = self {
            *ordinal
        } else {
            0
        }
    }
    fn label(&self) -> &'static str {
        if matches!(self, Self::Probe { .. }) {
            "complex_transform_sample"
        } else {
            "contour_sample"
        }
    }
}
fn save(
    values: &mut BTreeMap<String, String>,
    measurements: Values,
    requested: u32,
    used: u32,
) -> Result<()> {
    for (name, value) in measurements {
        save_arithmetic_enclosure(values, &name, &value, requested)?;
    }
    put(
        values,
        "arithmetic_precision_bits",
        &Float::with_val(requested, used),
    );
    Ok(())
}
fn measure_row(
    s: &RetainedState,
    sample: &Sample,
    index: usize,
    requested: u32,
) -> Result<AnalysisRow> {
    let mut row = row(index + 1, sample.label());
    if let Sample::Probe {
        ordinal,
        t: None,
        offset,
        status,
    } = sample
    {
        row.outcome = "missing_input".into();
        row.notes
            .push(format!("retained ordinate unavailable: {status}"));
        let p = requested + 64;
        save(
            &mut row.values,
            Values::from([
                ("input_ordinal".into(), I::from_u64(*ordinal as u64, p)),
                (
                    "z_im".into(),
                    I::from_i64(i64::from(*offset), p).div(&I::from_i64(4, p))?,
                ),
            ]),
            requested,
            p,
        )?;
        return Ok(row);
    }
    let mut fallback = None;
    let mut accepted = None;
    let mut last_error = None;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        let p = requested + guard;
        let attempt = (|| -> Result<Option<(Values, bool, bool)>> {
            let Some(c) = prepare(s, p)? else {
                return Ok(None);
            };
            let z = sample.coordinate(p)?.expect("present sample coordinate");
            let coordinates = Values::from([
                (
                    "input_ordinal".into(),
                    I::from_u64(sample.ordinal() as u64, p),
                ),
                ("z_re".into(), z.real().clone()),
                ("z_im".into(), z.imaginary().clone()),
            ]);
            if coordinates.values().all(|v| output_supported(v, requested)) {
                fallback = Some((coordinates.clone(), p));
            }
            let Some((f, d, a)) = transform(s, &c, &z)? else {
                return Ok(None);
            };
            let mut values = coordinates;
            for (name, v) in [
                ("value_re", f.real()),
                ("value_im", f.imaginary()),
                ("derivative_re", d.real()),
                ("derivative_im", d.imaginary()),
                ("sum_absolute_terms", &a),
            ] {
                values.insert(name.into(), v.clone());
            }
            if !narrow_values(&values, requested)? {
                return Ok(None);
            }
            let mut resolved = [false; 2];
            for (index, (name, ratio)) in [
                ("normalized", divide(&f, &real(&c.anchor)?)),
                ("log_derivative", divide(&d, &f)),
            ]
            .into_iter()
            .enumerate()
            {
                match ratio {
                    Ok(Some(z))
                        if narrow_value(z.real(), requested)?
                            && narrow_value(z.imaginary(), requested)?
                            && output_supported(z.real(), requested)
                            && output_supported(z.imaginary(), requested) =>
                    {
                        values.insert(format!("{name}_re"), z.real().clone());
                        values.insert(format!("{name}_im"), z.imaginary().clone());
                        resolved[index] = true;
                    }
                    Err(error) => last_error = Some(format!("ratio arithmetic: {error}")),
                    _ => {}
                }
            }
            values.insert(
                "normalization_denominator_resolved".into(),
                I::from_i64(i64::from(resolved[0]), p),
            );
            values.insert(
                "log_derivative_denominator_resolved".into(),
                I::from_i64(i64::from(resolved[1]), p),
            );
            let complete = resolved.iter().all(|v| *v);
            let finished = (resolved[0] || is_zero(&c.anchor))
                && (resolved[1] || (is_zero(f.real()) && is_zero(f.imaginary())));
            Ok(Some((values, complete, finished)))
        })();
        match attempt {
            Ok(Some((values, complete, finished))) => {
                accepted = Some((values, p, complete));
                if finished {
                    break;
                }
            }
            Err(error) => last_error = Some(format!("finite arithmetic: {error}")),
            _ => {}
        }
    }
    if let Some((values, p, complete)) = accepted {
        save(&mut row.values, values, requested, p)?;
        if !complete {
            row.outcome = "unresolved_denominator".into();
            row.notes.push("undefined, range-limited, or insufficiently enclosed ratio withheld; finite transform and derivative remain enclosed".into());
        }
        return Ok(row);
    }
    row.outcome = "cancellation_limited".into();
    row.notes
        .push("complex finite arithmetic unresolved within 4096 guard bits".into());
    if let Some(error) = last_error {
        row.notes.push(error);
    }
    if let Some((values, p)) = fallback {
        save(&mut row.values, values, requested, p)?;
    }
    Ok(row)
}
pub(super) fn analyze(
    s: &RetainedState,
    roots: Option<&RetainedRoots>,
    o: &ExtensionOptions,
) -> Result<ExtendedAnalysis> {
    let mut out = report("complex_transform", s, o);
    let requested = o.working_precision_bits;
    let policy = CaptureResourcePolicy::from_environment()?;
    // A block admits no more concurrent row work than this configured bound.
    // Host Rayon size can change scheduling, not resource-admission outcomes.
    let workers = policy.root_block_rows.max(1) as u64;
    let scratch = workers
        * (32 * s.coefficients.len() as u64 + 1024)
        * (u64::from(requested + 4096).div_ceil(8) + 64);
    if scratch
        > o.maximum_working_bytes
            .unwrap_or(policy.maximum_working_bytes)
    {
        return Ok(unresolved(
            out,
            "complex maximum-guard vector scratch exceeds working-byte budget",
        ));
    }
    let mut anchor = None;
    for guard in [64, 128, 256, 512, 1024, 2048, 4096] {
        if let Some(c) = prepare(s, requested + guard)? {
            if narrow_value(&c.anchor, requested)? && output_supported(&c.anchor, requested) {
                anchor = Some((c.anchor, requested + guard));
                break;
            }
        }
    }
    let Some((anchor, used)) = anchor else {
        return Ok(unresolved(
            out,
            "complex normalization anchor arithmetic unresolved within guard limit",
        ));
    };
    save(
        &mut out.values,
        Values::from([("normalization_anchor".into(), anchor)]),
        requested,
        used,
    )?;
    let mut samples = Vec::new();
    let mut maximum = Float::with_val(s.precision, 0);
    for offset in [-4, -1, 0, 1, 4] {
        samples.push(Sample::Probe {
            ordinal: 0,
            t: Some(Float::with_val(s.precision, 0)),
            offset,
            status: "origin".into(),
        });
    }
    if let Some(roots) = roots {
        for point in &roots.dataset.points {
            let t = point
                .value
                .as_ref()
                .map(|v| scalar(v, roots.dataset.precision_bits))
                .transpose()?;
            if let Some(t) = &t {
                if t > &maximum {
                    maximum = t.clone();
                }
            }
            for offset in [-4, -1, 0, 1, 4] {
                samples.push(Sample::Probe {
                    ordinal: point.ordinal,
                    t: t.clone(),
                    offset,
                    status: point.source_status.clone(),
                });
            }
        }
    }
    for side in 0..4 {
        for step in 0..16 {
            samples.push(Sample::Contour {
                side,
                step,
                maximum: maximum.clone(),
            });
        }
    }
    samples.push(Sample::Contour {
        side: 0,
        step: 0,
        maximum,
    });
    let checkpoints = Checkpoints::new(&(
        "complex-directed-minus-fourier-original-points-exact-cutoff-v2",
        crate::ccm::transform_enclosure::FOURIER_SEMANTICS,
        &s.manifest.content_digest,
        roots.map(|r| &r.manifest.content_digest),
        o,
    ))?;
    out.rows = row_blocks(&checkpoints, samples.len(), |index| {
        measure_row(s, &samples[index], index, requested)
    })?;
    if out
        .rows
        .iter()
        .any(|row| row.outcome != "point_measurement")
    {
        out.outcome = "partial_unresolved".into();
        out.reason=Some("some samples have missing points, unresolved arithmetic, or undefined/unresolved ratios; row qualifications retained".into());
    }
    out.convention="original stored coefficient and ordinate points; exact decimal cutoff; unit_L2_dx and F(z)/F(0); exp(-i*z*x); offsets 0,+/-0.25,+/-1 at every retained ordinal and origin; closed 16-interval-per-side counterclockwise rectangle with exact affine coordinates; outward finite arithmetic bounds; samples are not contour certificates and do not certify source errors".into();
    Ok(out)
}
