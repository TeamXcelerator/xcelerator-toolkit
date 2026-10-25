//! Correctly rounded implicit root motion of exact stored point data.
//! Directed, reversibly scaled arithmetic excludes source/assembly error.
//! Root and stored-point response guard limits are operational: the legacy attempt prefix is unchanged, and
//! only uniquely rounded results (or an exact interval zero) are accepted.
use crate::ccm::retained_evidence::finite_math::{scale, scale_float};
use anyhow::{bail, Result};
use rug::Float;
use xc_numerics::mpfr_interval::MpfrInterval as I;
const LEGACY_MAXIMUM_GUARD_BITS: u32 = 4096;
const MAXIMUM_CONFIGURED_GUARD_BITS: u32 = 1_048_576;

#[derive(Clone, Copy)]
pub(super) struct GuardPolicy {
    precision_bits: u32,
    maximum_guard_bits: u32,
}
impl GuardPolicy {
    pub(super) fn new(precision_bits: u32, maximum_guard_bits: u32) -> Result<Self> {
        validate_precision(precision_bits)?;
        if !maximum_guard_bits.is_power_of_two()
            || !(LEGACY_MAXIMUM_GUARD_BITS..=MAXIMUM_CONFIGURED_GUARD_BITS)
                .contains(&maximum_guard_bits)
        {
            bail!("invalid root-response maximum guard bits");
        }
        Ok(Self {
            precision_bits,
            maximum_guard_bits,
        })
    }
    pub(super) fn configured(precision_bits: u32) -> Result<Self> {
        validate_precision(precision_bits)?;
        let maximum = match std::env::var("XC_ROOT_RESPONSE_MAX_GUARD_BITS") {
            Ok(text) => parse_maximum_guard_bits(&text)?,
            // A precision-relative extension is useful at high precision, but
            // does not predict how many bits a particular cancellation needs.
            Err(std::env::VarError::NotPresent) => default_maximum_guard_bits(precision_bits)?,
            Err(_) => bail!("XC_ROOT_RESPONSE_MAX_GUARD_BITS is not valid Unicode"),
        };
        Self::new(precision_bits, maximum)
    }
    pub(super) fn guards(self) -> impl Iterator<Item = u32> {
        (6..=self.maximum_guard_bits.ilog2()).map(|power| 1u32 << power)
    }
    pub(super) fn maximum_guard_bits(self) -> u32 {
        self.maximum_guard_bits
    }
}
fn default_maximum_guard_bits(p: u32) -> Result<u32> {
    validate_precision(p)?;
    Ok((p + 64).next_power_of_two().max(LEGACY_MAXIMUM_GUARD_BITS))
}
fn parse_maximum_guard_bits(text: &str) -> Result<u32> {
    text.parse::<u32>()
        .ok()
        .filter(|value| {
            value.to_string() == text
                && value.is_power_of_two()
                && (LEGACY_MAXIMUM_GUARD_BITS..=MAXIMUM_CONFIGURED_GUARD_BITS).contains(value)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "XC_ROOT_RESPONSE_MAX_GUARD_BITS must be a decimal power of two from 4096 through 1048576"
            )
        })
}
fn validate_precision(p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p) {
        bail!("invalid root-response precision, finite source, shape or workspace");
    }
    Ok(())
}
pub(super) fn point_bytes(p: u32, guard: u32) -> Result<u128> {
    Ok(u128::from(work_precision(p, guard)?).div_ceil(8) + 96)
}
fn work_precision(p: u32, guard: u32) -> Result<u32> {
    validate_precision(p)?;
    if !guard.is_power_of_two() || !(64..=MAXIMUM_CONFIGURED_GUARD_BITS).contains(&guard) {
        bail!("invalid root-response geometry");
    }
    p.checked_add(guard)
        .ok_or_else(|| anyhow::anyhow!("root-response working precision overflow"))
}
fn reserve_attempt(bytes: u128) -> Result<Option<xc_core::MemoryReservation>> {
    let bytes = u64::try_from(bytes).map_err(|_| {
        anyhow::anyhow!("run memory budget exceeded: root-response workspace overflow")
    })?;
    if let Some(resources) = xc_core::RunResources::current() {
        // A child can be joined by an owner holding prepared geometry. Waiting
        // on that owner's memory would deadlock; refusal never advances guards.
        Ok(Some(resources.try_reserve_working(bytes)?))
    } else {
        let maximum = super::source_working_budget()?;
        if u128::from(bytes) > maximum {
            bail!("run memory budget exceeded: root-response workspace requested={bytes}, maximum={maximum}");
        }
        Ok(None)
    }
}

pub(super) fn validate(values: &[Float], p: u32) -> Result<()> {
    if !(64..=1_000_000).contains(&p)
        || values.is_empty()
        || values.len() > 16385
        || values.iter().any(|x| !x.is_finite() || x.prec() > p)
    {
        bail!("invalid root-response precision, finite source, shape or workspace");
    }
    Ok(())
}
fn exponent(values: &[Float]) -> Option<i64> {
    values
        .iter()
        .filter_map(Float::get_exp)
        .max()
        .map(i64::from)
}

pub(super) struct Geometry {
    denominator_reciprocals: Option<Vec<I>>,
    derivative: I,
    root: I,
    coordinates: i64,
    amplitudes: i64,
    exponent_range: (i32, i32),
}

// Owned by one source-bound PreparedRoot. No geometry crosses a source, root,
// precision or MPFR exponent-context boundary. Keep all reached derivatives,
// but retain pole reciprocals only at the latest reached precision so a long
// guard ladder cannot multiply the dense retained workspace.
pub(super) struct GeometryCache {
    entries: std::collections::BTreeMap<u32, CachedGeometry>,
    maximum_bytes: u64,
    exponent_range: (i32, i32),
}
struct CachedGeometry {
    geometry: Geometry,
    _base: xc_core::MemoryReservation,
    reciprocals: Option<xc_core::MemoryReservation>,
}
impl GeometryCache {
    pub(super) fn new(maximum_bytes: u64) -> Self {
        Self {
            entries: std::collections::BTreeMap::new(),
            maximum_bytes,
            exponent_range: (rug::float::exp_min(), rug::float::exp_max()),
        }
    }
    pub(super) fn clear(&mut self) -> bool {
        let released = !self.entries.is_empty();
        self.entries.clear();
        released
    }
    fn prepare(
        &mut self,
        weights: &[Float],
        poles: &[Float],
        root: &Float,
        p: u32,
        guard: u32,
    ) -> Result<Option<&Geometry>> {
        let context = (rug::float::exp_min(), rug::float::exp_max());
        if context != self.exponent_range {
            self.clear();
            self.exponent_range = context;
        }
        if self.entries.contains_key(&guard) {
            return Ok(self.entries.get(&guard).map(|entry| &entry.geometry));
        }
        let Some(resources) = xc_core::RunResources::current() else {
            return Ok(None);
        };
        let point = point_bytes(p, guard)?;
        let base = u64::try_from(point * 4 + 256)?;
        let reciprocals = u64::try_from(point * 2 * poles.len() as u128)?;
        // Earlier derivatives stay cached even when their larger reciprocal
        // arrays are displaced. The original guard sequence is never skipped.
        for entry in self.entries.values_mut() {
            entry.geometry.denominator_reciprocals.take();
            entry.reciprocals.take();
        }
        let used = self.entries.values().map(|e| e._base.bytes()).sum::<u64>();
        if used.saturating_add(base) > self.maximum_bytes {
            return Ok(None);
        }
        let Ok(base_reservation) = resources.try_reserve_working(base) else {
            return Ok(None);
        };
        let reciprocal_reservation =
            if used.saturating_add(base).saturating_add(reciprocals) <= self.maximum_bytes {
                resources.try_reserve_working(reciprocals).ok()
            } else {
                None
            };
        let geometry = Geometry::prepare(
            weights,
            poles,
            root,
            p,
            guard,
            reciprocal_reservation.is_some(),
        )?;
        self.entries.insert(
            guard,
            CachedGeometry {
                geometry,
                _base: base_reservation,
                reciprocals: reciprocal_reservation,
            },
        );
        Ok(self.entries.get(&guard).map(|entry| &entry.geometry))
    }
}
impl Geometry {
    pub(super) fn prepare(
        weights: &[Float],
        poles: &[Float],
        root: &Float,
        p: u32,
        guard: u32,
        retain: bool,
    ) -> Result<Self> {
        validate(weights, p)?;
        validate(poles, p)?;
        validate(std::slice::from_ref(root), p)?;
        if weights.len() != poles.len() {
            bail!("invalid root-response geometry");
        }
        let work = work_precision(p, guard)?;
        let _reservation = reserve_attempt(poles.len() as u128 * point_bytes(p, guard)? * 24)?;
        let coordinates = poles
            .iter()
            .chain(std::iter::once(root))
            .filter_map(Float::get_exp)
            .max()
            .map_or(0, i64::from);
        let amplitudes = exponent(weights)
            .ok_or_else(|| anyhow::anyhow!("root response has a zero secular derivative"))?;
        let root = I::point(scale_float(root, -coordinates, work)?);
        let mut out = Self {
            denominator_reciprocals: retain.then(|| Vec::with_capacity(poles.len())),
            derivative: I::from_i64(0, work),
            root,
            coordinates,
            amplitudes,
            exponent_range: (rug::float::exp_min(), rug::float::exp_max()),
        };
        for (weight, pole) in weights.iter().zip(poles) {
            let denominator = out.separation(pole)?;
            let term = I::point(scale_float(weight, -amplitudes, work)?)
                .div(&denominator)?
                .div(&denominator)?;
            out.derivative = out.derivative.add(&term);
            if let Some(stored) = &mut out.denominator_reciprocals {
                // MpfrInterval::div is exactly mul(reciprocal), including
                // endpoint rounding. Retain that same reciprocal, not a
                // nearest-rounded inverse or a differently rounded quotient.
                stored.push(denominator.reciprocal()?);
            }
        }
        out.derivative.validate()?;
        if out.derivative.lower().is_zero() && out.derivative.upper().is_zero() {
            bail!("root response has a zero secular derivative");
        }
        Ok(out)
    }
    fn separation(&self, pole: &Float) -> Result<I> {
        let d = self.root.sub(&I::point(scale_float(
            pole,
            -self.coordinates,
            self.root.precision(),
        )?));
        d.validate()?;
        if d.contains_zero() {
            bail!("root response encountered a secular pole or unresolved pole separation");
        }
        Ok(d)
    }
    fn response(
        &self,
        weights: &[Float],
        tangent: &[Float],
        poles: &[Float],
        motions: Option<&[Float]>,
        p: u32,
    ) -> Result<Option<Float>> {
        if self.derivative.contains_zero() {
            return Ok(None);
        }
        let work = self.root.precision();
        let _reservation =
            reserve_attempt(poles.len() as u128 * (u128::from(work).div_ceil(8) + 96) * 24)?;
        let tangent_exponent = exponent(tangent);
        let motion_exponent = motions.and_then(exponent);
        let fixed_exponent = tangent_exponent.map(|e| e - self.amplitudes + self.coordinates);
        let output_exponent = fixed_exponent
            .into_iter()
            .chain(motion_exponent)
            .max()
            .unwrap_or(0);
        let mut fixed = I::from_i64(0, work);
        let mut moving = I::from_i64(0, work);
        for (index, pole) in poles.iter().enumerate() {
            let temporary;
            let reciprocal = self
                .denominator_reciprocals
                .as_ref()
                .map(|stored| &stored[index]);
            let denominator = if reciprocal.is_none() {
                temporary = self.separation(pole)?;
                Some(&temporary)
            } else {
                None
            };
            let divide = |value: &I| -> Result<I> {
                if let Some(reciprocal) = reciprocal {
                    value.validate()?;
                    let result = value.mul(reciprocal);
                    result.validate()?;
                    Ok(result)
                } else {
                    Ok(value.div(denominator.unwrap())?)
                }
            };
            if let Some(e) = tangent_exponent {
                fixed = fixed.add(&divide(&I::point(scale_float(&tangent[index], -e, work)?))?);
            }
            if let (Some(velocities), Some(e)) = (motions, motion_exponent) {
                let product = I::point(scale_float(&weights[index], -self.amplitudes, work)?)
                    .mul(&I::point(scale_float(&velocities[index], -e, work)?));
                let term = divide(&divide(&product)?)?;
                moving = moving.add(&term);
            }
        }
        let mut source = I::from_i64(0, work);
        if let Some(e) = fixed_exponent {
            source = source.add(&scale(&fixed, e - output_exponent)?);
        }
        if let Some(e) = motion_exponent {
            source = source.add(&scale(&moving, e - output_exponent)?);
        }
        let response = scale(&source.div(&self.derivative)?, output_exponent)?;
        response.validate()?;
        let lo = Float::with_val(p, response.lower());
        let hi = Float::with_val(p, response.upper());
        if lo == hi
            && lo.is_finite()
            && (!lo.is_zero() || (response.lower().is_zero() && response.upper().is_zero()))
        {
            return Ok(Some(lo));
        }
        Ok(None)
    }
}

pub(super) fn evaluate(
    weights: &[Float],
    tangent: &[Float],
    poles: &[Float],
    motions: Option<&[Float]>,
    root: &Float,
    p: u32,
    prepared: Option<&Geometry>,
) -> Result<Float> {
    evaluate_with_policy(
        weights,
        tangent,
        poles,
        motions,
        root,
        GuardPolicy::configured(p)?,
        prepared,
    )
}

pub(super) fn evaluate_with_policy(
    weights: &[Float],
    tangent: &[Float],
    poles: &[Float],
    motions: Option<&[Float]>,
    root: &Float,
    policy: GuardPolicy,
    prepared: Option<&Geometry>,
) -> Result<Float> {
    evaluate_with_cache(
        weights, tangent, poles, motions, root, policy, prepared, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_with_cache(
    weights: &[Float],
    tangent: &[Float],
    poles: &[Float],
    motions: Option<&[Float]>,
    root: &Float,
    policy: GuardPolicy,
    prepared: Option<&Geometry>,
    mut cache: Option<&mut GeometryCache>,
) -> Result<Float> {
    let p = policy.precision_bits;
    validate(weights, p)?;
    validate(tangent, p)?;
    validate(poles, p)?;
    validate(std::slice::from_ref(root), p)?;
    if weights.len() != poles.len() || tangent.len() != poles.len() {
        bail!("root-response source dimensions differ");
    }
    if let Some(motions) = motions {
        validate(motions, p)?;
        if motions.len() != poles.len() {
            bail!("root-response pole-motion dimensions differ");
        }
    }
    for guard in policy.guards() {
        if guard == 64 {
            if let Some(g) = prepared
                .filter(|g| g.exponent_range == (rug::float::exp_min(), rug::float::exp_max()))
            {
                if g.root.precision() != work_precision(p, guard)? {
                    bail!("root-response prepared geometry precision mismatch");
                }
                if let Some(value) = g.response(weights, tangent, poles, motions, p)? {
                    return Ok(value);
                }
                continue;
            }
        }
        if let Some(cache) = cache.as_deref_mut() {
            if let Some(g) = cache.prepare(weights, poles, root, p, guard)? {
                if let Some(value) = g.response(weights, tangent, poles, motions, p)? {
                    return Ok(value);
                }
                continue;
            }
        }
        let g = Geometry::prepare(weights, poles, root, p, guard, false)?;
        if let Some(value) = g.response(weights, tangent, poles, motions, p)? {
            return Ok(value);
        }
    }
    bail!(
        "root-response rounding or nonzero derivative unresolved within {} guard bits",
        policy.maximum_guard_bits
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Rational;
    #[test]
    fn retained_reciprocals_preserve_interval_endpoints_and_guard_outcomes() {
        for p in [64, 128, 3386, 6708] {
            let weights = [1, -3, 2, 5].map(|v| Float::with_val(p, v));
            let poles = [-3, -1, 1, 3].map(|v| Float::with_val(p, v));
            let root = Float::with_val(p, 0.25);
            for guard in [64, 128, 4096] {
                let plain = Geometry::prepare(&weights, &poles, &root, p, guard, false).unwrap();
                let cached = Geometry::prepare(&weights, &poles, &root, p, guard, true).unwrap();
                for coefficients in [[1, -2, 3, -4], [0, 0, 0, 0], [7, 6, -8, -5]] {
                    let tangent = coefficients.map(|v| Float::with_val(p, v));
                    for motions in [None, Some(tangent.as_slice())] {
                        let a = plain
                            .response(&weights, &tangent, &poles, motions, p)
                            .unwrap();
                        let b = cached
                            .response(&weights, &tangent, &poles, motions, p)
                            .unwrap();
                        assert_eq!(
                            a.as_ref().map(Float::to_integer_exp),
                            b.as_ref().map(Float::to_integer_exp)
                        );
                        assert_eq!(
                            a.as_ref().map(Float::is_sign_negative),
                            b.as_ref().map(Float::is_sign_negative)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reached_geometry_reuse_preserves_cancellation_refusal_and_reservations() {
        let p = 64;
        let weights = [
            Float::with_val(p, 1),
            Float::with_val(p, -1),
            Float::with_val(p, 1) >> 5000u32,
        ];
        let poles = [-3, 3, 2].map(|v| Float::with_val(p, v));
        let root = Float::with_val(p, 0);
        let owner = xc_core::RunResources::with_worker_limit(16 << 20, 2).unwrap();
        let _scope = owner.enter();
        for budget in [0, 512, 1 << 20] {
            let mut cache = GeometryCache::new(budget);
            for sign in [-1, 0, 1] {
                let tangent = [
                    Float::with_val(p, 0),
                    Float::with_val(p, 0),
                    Float::with_val(p, sign) >> 5001u32,
                ];
                for limit in [4096, 8192] {
                    let policy = GuardPolicy::new(p, limit).unwrap();
                    let expected =
                        evaluate_with_policy(&weights, &tangent, &poles, None, &root, policy, None);
                    for _ in 0..2 {
                        let actual = evaluate_with_cache(
                            &weights,
                            &tangent,
                            &poles,
                            None,
                            &root,
                            policy,
                            None,
                            Some(&mut cache),
                        );
                        match (&expected, &actual) {
                            (Ok(a), Ok(b)) => assert!(a.total_cmp(b).is_eq()),
                            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                            _ => panic!("cache changed outcome"),
                        }
                    }
                }
            }
            assert!(owner.usage().working_bytes <= budget);
            cache.clear();
            assert_eq!(owner.usage().working_bytes, 0);
        }
    }
    #[test]
    fn exhaustive_response_range_independent_rational_oracles() {
        let mut checked = 0;
        for p in [64, 128, 256] {
            for case in 0i32..12 {
                let w = [case - 4, case + 1, 2 - case];
                let t = [case + 3, 1 - case, case - 7];
                let v = [case - 2, 3 - case, case + 1];
                let positions: [i32; 3] = [-2, -1, 1];
                let z = Rational::from((1, 4));
                let mut derivative = Rational::new();
                let mut fixed = Rational::new();
                let mut moving = Rational::new();
                for i in 0..3 {
                    let d = z.clone() - positions[i];
                    derivative += Rational::from(w[i]) / d.clone().square();
                    fixed += Rational::from(t[i]) / &d;
                    moving += Rational::from(w[i] * v[i]) / d.square();
                }
                assert_ne!(derivative, 0);
                let expected_fixed = Float::with_val(p, fixed.clone() / &derivative);
                let expected_total = Float::with_val(p, (fixed + moving) / &derivative);
                for c in [-700_000_000i32, 0, 700_000_000] {
                    for amplitude in [-700_000_000i32, 0, 700_000_000] {
                        let weights = w.map(|x| Float::with_val(p, x) << amplitude);
                        let tangent = t.map(|x| Float::with_val(p, x) << amplitude);
                        let poles = positions.map(|x| Float::with_val(p, x) << c);
                        let motions = v.map(|x| Float::with_val(p, x) << c);
                        let root = Float::with_val(p, &z) << c;
                        assert_eq!(
                            evaluate(&weights, &tangent, &poles, None, &root, p, None).unwrap(),
                            expected_fixed.clone() << c
                        );
                        assert_eq!(
                            evaluate(&weights, &tangent, &poles, Some(&motions), &root, p, None)
                                .unwrap(),
                            expected_total.clone() << c
                        );
                        for retain in [false, true] {
                            let geometry =
                                Geometry::prepare(&weights, &poles, &root, p, 64, retain).unwrap();
                            assert_eq!(
                                evaluate(
                                    &weights,
                                    &tangent,
                                    &poles,
                                    None,
                                    &root,
                                    p,
                                    Some(&geometry)
                                )
                                .unwrap(),
                                expected_fixed.clone() << c
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 324);
    }
    #[test]
    fn exhaustive_response_range_zero_motion_and_source_contracts() {
        let p = 128;
        let w = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let zeros = vec![Float::with_val(p, 0); 2];
        let poles = [Float::with_val(p, -1), Float::with_val(p, 1)];
        let root = Float::with_val(p, 0);
        assert_eq!(
            evaluate(&w, &zeros, &poles, Some(&zeros), &root, p, None).unwrap(),
            0
        );
        assert!(evaluate(&w[..1], &zeros, &poles, None, &root, p, None).is_err());
        assert!(evaluate(&w, &zeros, &poles, Some(&zeros[..1]), &root, p, None).is_err());
        assert!(evaluate(&w, &zeros, &poles, None, &poles[0], p, None).is_err());
        assert!(evaluate(&zeros, &zeros, &poles, None, &root, p, None).is_err());
        let cancelling = [Float::with_val(p, 1), Float::with_val(p, -1)];
        assert!(evaluate(&cancelling, &zeros, &poles, None, &root, p, None).is_err());
        for bad in [0, 63, 1_000_001, u32::MAX] {
            assert!(validate(&w, bad).is_err());
        }
        assert!(validate(&vec![Float::with_val(p, 0); 16386], p).is_err());
    }

    #[test]
    fn root_response_guard_policy_preserves_prefix_and_bounds() {
        let legacy = [64, 128, 256, 512, 1024, 2048, 4096];
        for maximum in [4096, 8192, 1_048_576] {
            let policy = GuardPolicy::new(64, maximum).unwrap();
            assert_eq!(policy.guards().take(7).collect::<Vec<_>>(), legacy);
            assert_eq!(policy.guards().last(), Some(maximum));
            assert_eq!(
                parse_maximum_guard_bits(&maximum.to_string()).unwrap(),
                maximum
            );
        }
        for text in [
            "",
            "0",
            "64",
            "4095",
            "4097",
            "+4096",
            "04096",
            " 4096",
            "4096\n",
            "2097152",
            "4294967296",
        ] {
            assert!(parse_maximum_guard_bits(text).is_err(), "{text:?}");
        }
        for (p, maximum) in [
            (64, 4096),
            (3386, 4096),
            (6708, 8192),
            (1_000_000, 1_048_576),
        ] {
            assert_eq!(default_maximum_guard_bits(p).unwrap(), maximum);
        }
        for p in [0, 63, 1_000_001, u32::MAX] {
            assert!(GuardPolicy::new(p, 4096).is_err());
            assert!(default_maximum_guard_bits(p).is_err());
        }
        for guard in [0, 32, 63, 65, 4097, 2_097_152, u32::MAX] {
            assert!(work_precision(64, guard).is_err());
        }
    }

    #[test]
    fn root_response_guard_extension_resolves_exact_stored_cancellation() {
        let p = 64;
        let weights = [
            Float::with_val(p, 1),
            Float::with_val(p, -1),
            Float::with_val(p, 1) >> 5000u32,
        ];
        let tangent = [
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, -1) >> 5001u32,
        ];
        let poles = [-3, 3, 2].map(|x| Float::with_val(p, x));
        let root = Float::with_val(p, 0);
        let zeros = [
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
        ];
        // Exact rational derivative and fixed numerator are both 2^-5002.
        let legacy = GuardPolicy::new(p, 4096).unwrap();
        let extended = GuardPolicy::new(p, 8192).unwrap();
        assert!(
            evaluate_with_policy(&weights, &tangent, &poles, None, &root, legacy, None)
                .unwrap_err()
                .to_string()
                .contains("within 4096 guard bits")
        );
        for retain in [false, true] {
            let geometry = Geometry::prepare(&weights, &poles, &root, p, 64, retain).unwrap();
            assert!(evaluate_with_policy(
                &weights,
                &tangent,
                &poles,
                None,
                &root,
                legacy,
                Some(&geometry)
            )
            .is_err());
            for motions in [None, Some(zeros.as_slice())] {
                for prepared in [None, Some(&geometry)] {
                    let value = evaluate_with_policy(
                        &weights, &tangent, &poles, motions, &root, extended, prepared,
                    )
                    .unwrap();
                    assert_eq!(value, 1);
                    assert_eq!(value.prec(), p);
                }
            }
        }
        let owner = xc_core::RunResources::new(
            u64::try_from(3 * point_bytes(p, 4096).unwrap() * 24).unwrap(),
        )
        .unwrap();
        let _scope = owner.enter();
        let refused = evaluate_with_policy(&weights, &tangent, &poles, None, &root, extended, None)
            .unwrap_err();
        assert!(refused.to_string().contains("run memory budget exceeded"));
        assert_eq!(owner.usage().working_bytes, 0);
    }

    #[test]
    fn root_response_guard_extension_preserves_legacy_values_and_zero_sign() {
        for p in [64, 128, 6708] {
            let weights = [Float::with_val(p, 1), Float::with_val(p, 1)];
            let poles = [-1, 1].map(|x| Float::with_val(p, x));
            let root = Float::with_val(p, 0);
            for coefficients in [[1.0, -1.0], [-1.0, 1.0], [0.0, -0.0], [-0.0, 0.0]] {
                let tangent = coefficients.map(|x| Float::with_val(p, x));
                let legacy = evaluate_with_policy(
                    &weights,
                    &tangent,
                    &poles,
                    None,
                    &root,
                    GuardPolicy::new(p, 4096).unwrap(),
                    None,
                )
                .unwrap();
                let extended = evaluate_with_policy(
                    &weights,
                    &tangent,
                    &poles,
                    None,
                    &root,
                    GuardPolicy::new(p, 1_048_576).unwrap(),
                    None,
                )
                .unwrap();
                assert_eq!(extended, legacy);
                assert_eq!(extended.prec(), legacy.prec());
                assert_eq!(extended.is_sign_negative(), legacy.is_sign_negative());
                assert_eq!(
                    super::super::lossless_hp_decimal(&extended),
                    super::super::lossless_hp_decimal(&legacy)
                );
            }
        }
    }

    #[test]
    fn root_response_guard_extension_never_invents_zero() {
        let p = 64;
        let weights = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let poles = [-3, 3].map(|x| Float::with_val(p, x));
        let root = Float::with_val(p, 0);
        // The true numerator is zero, but directed +/-1/3 never proves it.
        let unresolved = evaluate_with_policy(
            &weights,
            &weights,
            &poles,
            None,
            &root,
            GuardPolicy::new(p, 8192).unwrap(),
            None,
        )
        .unwrap_err();
        assert!(unresolved.to_string().contains("within 8192 guard bits"));
        let cancelling = [Float::with_val(p, 1), Float::with_val(p, -1)];
        assert!(evaluate_with_policy(
            &cancelling,
            &weights,
            &poles,
            None,
            &root,
            GuardPolicy::new(p, 8192).unwrap(),
            None
        )
        .is_err());
        assert!(evaluate_with_policy(
            &weights,
            &weights,
            &poles,
            None,
            &poles[0],
            GuardPolicy::new(p, 8192).unwrap(),
            None
        )
        .unwrap_err()
        .to_string()
        .contains("secular pole"));
    }

    #[test]
    fn root_response_guard_attempt_admission_is_immediate_and_incremental() {
        let p = 64;
        let weights = [Float::with_val(p, 1), Float::with_val(p, 1)];
        let poles = [-1, 1].map(|x| Float::with_val(p, x));
        let tangent = [Float::with_val(p, 1), Float::with_val(p, -1)];
        let root = Float::with_val(p, 0);
        let bytes = u64::try_from(2 * point_bytes(p, 64).unwrap() * 24).unwrap();
        let owner = xc_core::RunResources::new(bytes).unwrap();
        let _scope = owner.enter();
        let policy = GuardPolicy::new(p, 1_048_576).unwrap();
        assert_eq!(
            evaluate_with_policy(&weights, &tangent, &poles, None, &root, policy, None).unwrap(),
            1
        );
        assert_eq!(owner.usage().peak_bytes, bytes);
        assert_eq!(owner.usage().working_bytes, 0);
        let held = owner.try_reserve_working(1).unwrap();
        assert!(
            evaluate_with_policy(&weights, &tangent, &poles, None, &root, policy, None)
                .unwrap_err()
                .to_string()
                .contains("run memory budget exceeded")
        );
        drop(held);
        assert_eq!(owner.usage().working_bytes, 0);
        // Source admission is independent of the former unconditional 8 GiB
        // workspace ceiling; actual promoted allocations are admitted above.
        let high_precision = vec![Float::with_val(64, 1); 16385];
        assert!(validate(&high_precision, 1_000_000).is_ok());
        let large_owner = xc_core::RunResources::new(9u64 << 30).unwrap();
        let _scope = large_owner.enter();
        drop(reserve_attempt((8u128 << 30) + 1).unwrap());
        assert_eq!(large_owner.usage().working_bytes, 0);
    }
}
