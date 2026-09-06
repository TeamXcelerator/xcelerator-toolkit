//! Certified cutoff-free CCM Weil-matrix assembly.
//!
//! This is an independent implementation of the closed-form mathematics in
//! Groskin, arXiv:2607.02828.  It evaluates the complex digamma and trigamma
//! terms with system FLINT/Arb balls, encloses every remaining operation with
//! outward-rounded MPFR intervals, and retains the `W02`, `WR`, and `Wp`
//! components separately for cancellation audit.

use super::arb_bridge::{backend_version, complex_digamma, complex_trigamma};
use super::prime_powers_up_to;
use anyhow::{bail, Context, Result};
use rug::{Float, Rational};
use xc_cache::{sha256_hex, ContentDigest};
use xc_certify::exact::{
    build_portable_interval_inertia_certificate, interval_record, interval_symmetric_ldlt_inertia,
    IntervalInertiaResult,
};
use xc_certify::PortableIntervalInertiaCertificate;
use xc_numerics::interval::RationalInterval;
use xc_numerics::mpfr_interval::MpfrInterval;

/// Corrected finite-endpoint and aggregate-prime assembly identity.
/// Old inertia records may remain readable as records, but are not evidence
/// for this assembly. Sector certificates independently version their schema.
pub const ASSEMBLY_SEMANTICS: &str = "ccm-cutoff-free-zero-endpoint-aggregate-primes-v0.14.4-v1";

/// Deterministic conservative analytic-tail budget, with no floating-point
/// estimate of log(c). For c >= 2 and b=floor(log2(c)), the common special-value
/// tail is at most 6*2^(-2*M*b). The additional dimension allowance covers the
/// O(N) frequency factor and O(d) row-sum propagation in the archimedean form.
/// This controls the analytic series tail, NOT total assembly roundoff or a
/// spectral gap. Exact interval verification still decides certificate success.
pub fn recommended_geometric_terms(c: u64, modes: usize, precision_bits: u32) -> usize {
    let b = u64::from(63_u32.saturating_sub(c.leading_zeros())).max(1);
    let d = modes.saturating_mul(2).saturating_add(1);
    let dimension_bits = u64::from(usize::BITS - d.leading_zeros());
    let required = u64::from(precision_bits) + 2 * dimension_bits + 16;
    usize::try_from(required.div_ceil(2 * b))
        .unwrap_or(usize::MAX)
        .max(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutoffFreeConfig {
    pub integer_cutoff_c: u64,
    pub modes: usize,
    pub precision_bits: u32,
    pub geometric_terms: usize,
}

impl CutoffFreeConfig {
    pub fn new(integer_cutoff_c: u64, modes: usize, precision_bits: u32) -> Self {
        Self {
            integer_cutoff_c,
            modes,
            precision_bits,
            geometric_terms: recommended_geometric_terms(integer_cutoff_c, modes, precision_bits),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.integer_cutoff_c <= 1 {
            bail!("cutoff-free CCM requires integer c > 1");
        }
        if self.precision_bits < 64 {
            bail!("cutoff-free CCM requires at least 64 bits of precision");
        }
        if self.geometric_terms == 0 {
            bail!("cutoff-free CCM requires at least one geometric correction term");
        }
        let dimension = self.modes.checked_mul(2).and_then(|n| n.checked_add(1));
        if dimension.and_then(|n| n.checked_mul(n)).is_none()
            || self.modes > (i64::MAX as usize) / 4
            || self.geometric_terms > ((i64::MAX - 1) / 4) as usize
        {
            bail!("cutoff-free CCM dimensions or series indices overflow");
        }
        Ok(())
    }

    pub fn dimension(&self) -> usize {
        2 * self.modes + 1
    }
}

#[derive(Clone, Debug)]
pub struct CutoffFreeMatrix {
    pub config: CutoffFreeConfig,
    pub scalar_backend: String,
    pub w02: Vec<RationalInterval>,
    pub wr: Vec<RationalInterval>,
    pub wp: Vec<RationalInterval>,
    pub tau: Vec<RationalInterval>,
}

impl CutoffFreeMatrix {
    pub fn dimension(&self) -> usize {
        self.config.dimension()
    }

    pub fn certify_inertia(&self) -> Result<IntervalInertiaResult> {
        interval_symmetric_ldlt_inertia(&self.tau, self.dimension()).map_err(anyhow::Error::from)
    }

    /// Digest of the exact `W02`, `WR`, and `Wp` interval records used to
    /// assemble this matrix.  Derived certificates can bind the same
    /// component evidence without first running a full inertia proof.
    pub fn component_evidence_digest(&self) -> Result<ContentDigest> {
        let component_evidence = (
            ASSEMBLY_SEMANTICS,
            self.config.integer_cutoff_c,
            self.config.modes,
            self.config.precision_bits,
            self.config.geometric_terms,
            &self.scalar_backend,
            self.w02.iter().map(interval_record).collect::<Vec<_>>(),
            self.wr.iter().map(interval_record).collect::<Vec<_>>(),
            self.wp.iter().map(interval_record).collect::<Vec<_>>(),
        );
        let evidence_bytes = serde_json::to_vec(&component_evidence)
            .context("serialize cutoff-free component evidence")?;
        Ok(ContentDigest(sha256_hex(&evidence_bytes)))
    }

    pub fn portable_inertia_certificate(&self) -> Result<PortableIntervalInertiaCertificate> {
        build_portable_interval_inertia_certificate(
            &self.tau,
            self.dimension(),
            self.config.precision_bits,
            self.scalar_backend.clone(),
            self.component_evidence_digest()?,
            std::collections::BTreeMap::from([
                (
                    "assembly_semantics".to_owned(),
                    ASSEMBLY_SEMANTICS.to_owned(),
                ),
                (
                    "integer_cutoff_c".to_owned(),
                    self.config.integer_cutoff_c.to_string(),
                ),
                ("modes".to_owned(), self.config.modes.to_string()),
                (
                    "geometric_terms".to_owned(),
                    self.config.geometric_terms.to_string(),
                ),
            ]),
            vec![
                format!(
                    "cutoff-free CCM c={}, N={}, geometric_terms={}",
                    self.config.integer_cutoff_c, self.config.modes, self.config.geometric_terms
                ),
                "exact rational endpoints retain complete W02-WR-Wp assembly uncertainty"
                    .to_owned(),
            ],
        )
        .map_err(anyhow::Error::from)
    }
}

fn q(value: i64, denominator: i64, precision: u32) -> MpfrInterval {
    MpfrInterval::from_rational(&Rational::from((value, denominator)), precision)
}

fn nonnegative_remainder(bound: &MpfrInterval) -> Result<MpfrInterval> {
    MpfrInterval::new(Float::with_val(bound.precision(), 0), bound.upper().clone())
        .map_err(anyhow::Error::from)
}

fn symmetric_remainder(bound: &MpfrInterval) -> Result<MpfrInterval> {
    MpfrInterval::new(-bound.upper().clone(), bound.upper().clone()).map_err(anyhow::Error::from)
}

fn sinh(value: &MpfrInterval) -> Result<MpfrInterval> {
    let two = MpfrInterval::from_i64(2, value.precision());
    value
        .exp()
        .sub(&value.neg().exp())
        .div(&two)
        .map_err(Into::into)
}

#[derive(Clone)]
struct SpecialValues {
    s: MpfrInterval,
    cc: MpfrInterval,
    xc: MpfrInterval,
}

fn special_values(
    n: usize,
    l: &MpfrInterval,
    pi: &MpfrInterval,
    psi_quarter: &MpfrInterval,
    terms: usize,
) -> Result<SpecialValues> {
    let p = l.precision();
    // n=0 needs the SAME finite-endpoint correction as every other mode.
    // psi_1(1/4)/4 alone is the integral over [0,infinity), not [0,L].
    let n_interval = MpfrInterval::from_u64(n as u64, p);
    let b = pi.mul(&n_interval).div(l)?;
    let w = b.mul(&MpfrInterval::from_i64(2, p));
    let quarter = q(1, 4, p);
    let (digamma_re, digamma_im) = complex_digamma(&quarter, &b)?;
    let (trigamma_re, _) = complex_trigamma(&quarter, &b)?;

    let zero = MpfrInterval::from_i64(0, p);
    let mut gs = zero.clone();
    let mut gcc = zero.clone();
    let mut gx1 = zero.clone();
    let mut gx2 = zero;
    let w_squared = w.square();
    for k in 0..terms {
        let ck = q((4 * k + 1) as i64, 2, p);
        let exponent = ck.mul(l).neg();
        let e = exponent.exp();
        let ck_squared = ck.square();
        let denominator = ck_squared.add(&w_squared);
        gs = gs.add(&e.div(&denominator)?);
        gcc = gcc.add(&e.mul(&w_squared).div(&ck.mul(&denominator))?);
        gx1 = gx1.add(&e.mul(&ck).div(&denominator)?);
        gx2 = gx2.add(
            &e.mul(&ck_squared.sub(&w_squared))
                .div(&denominator.square())?,
        );
    }

    let next_ck = q((4 * terms + 1) as i64, 2, p);
    let numerator = next_ck.mul(l).neg().exp();
    let two = MpfrInterval::from_i64(2, p);
    let denominator = MpfrInterval::from_i64(1, p).sub(&l.mul(&two).neg().exp());
    let tail = numerator
        .div(&denominator)?
        .mul(&MpfrInterval::from_i64(4, p));
    let positive_tail = nonnegative_remainder(&tail)?;
    let signed_tail = symmetric_remainder(&tail)?;
    gs = gs.add(&positive_tail);
    gcc = gcc.add(&positive_tail);
    gx1 = gx1.add(&positive_tail);
    gx2 = gx2.add(&signed_tail);

    let s = digamma_im
        .div(&MpfrInterval::from_i64(2, p))?
        .sub(&w.mul(&gs));
    let cc = digamma_re
        .sub(psi_quarter)
        .div(&MpfrInterval::from_i64(-2, p))?
        .add(&gcc);
    let xc = trigamma_re
        .div(&MpfrInterval::from_i64(4, p))?
        .sub(&l.mul(&gx1))
        .sub(&gx2);
    // The sine and (cos-1) integrals vanish identically at zero frequency;
    // preserve that identity instead of subtracting two interval evaluations.
    if n == 0 {
        let zero = MpfrInterval::from_i64(0, p);
        Ok(SpecialValues {
            s: zero.clone(),
            cc: zero,
            xc,
        })
    } else {
        Ok(SpecialValues { s, cc, xc })
    }
}

fn signed_s(values: &[SpecialValues], n: i64) -> MpfrInterval {
    let value = values[n.unsigned_abs() as usize].s.clone();
    if n < 0 {
        value.neg()
    } else {
        value
    }
}

pub fn assemble(config: &CutoffFreeConfig) -> Result<CutoffFreeMatrix> {
    config.validate()?;
    let p = config.precision_bits;
    let c = MpfrInterval::from_u64(config.integer_cutoff_c, p);
    let l = c.ln()?;
    let pi = MpfrInterval::pi(p);
    let zero = MpfrInterval::from_i64(0, p);
    let quarter = q(1, 4, p);
    let (psi_quarter, _) = complex_digamma(&quarter, &zero)?;
    let special: Vec<SpecialValues> = (0..=config.modes)
        .map(|n| special_values(n, &l, &pi, &psi_quarter, config.geometric_terms))
        .collect::<Result<_>>()?;

    let u = c.sqrt()?;
    let one = MpfrInterval::from_i64(1, p);
    let two = MpfrInterval::from_i64(2, p);
    let four = MpfrInterval::from_i64(4, p);
    let log_two = two.ln()?;
    let j = u
        .add(&one)
        .ln()?
        .mul(&MpfrInterval::from_i64(-2, p))
        .add(&u.square().add(&one).ln()?)
        .add(&u.atan().mul(&two))
        .add(&log_two)
        .sub(&pi.div(&two)?);
    let kappa = four
        .mul(&pi)
        .mul(&c.sub(&one))
        .div(&c.add(&one))?
        .ln()?
        .add(&MpfrInterval::euler_gamma(p));

    let prime_data: Vec<(MpfrInterval, MpfrInterval, MpfrInterval)> =
        prime_powers_up_to(config.integer_cutoff_c)
            .into_iter()
            .map(|(power, prime, _)| {
                let power_value = MpfrInterval::from_u64(power, p);
                Ok((
                    power_value.ln()?,
                    MpfrInterval::from_u64(prime, p).ln()?,
                    power_value.sqrt()?,
                ))
            })
            .collect::<Result<_>>()?;

    // Sum the prime-power generators once per mode. Off-diagonal entries
    // are divided differences of these generators. Outward rounding remains
    // in force, and the changed enclosure arithmetic has a NEW identity.
    let mut sine_moments = Vec::with_capacity(config.modes + 1);
    let mut diagonal_moments = Vec::with_capacity(config.modes + 1);
    for n in 0..=config.modes {
        let nf = MpfrInterval::from_u64(n as u64, p);
        let mut sine = zero.clone();
        let mut diagonal = zero.clone();
        for (log_power, log_prime, sqrt_power) in &prime_data {
            let phase = pi.mul(&two).mul(&nf).mul(log_power).div(&l)?;
            let weight = log_prime.div(sqrt_power)?;
            if n != 0 {
                sine = sine.add(&phase.sin().mul(&weight));
            }
            diagonal = diagonal.add(
                &one.sub(&log_power.div(&l)?)
                    .mul(&two)
                    .mul(&phase.cos())
                    .mul(&weight),
            );
        }
        sine_moments.push(sine);
        diagonal_moments.push(diagonal);
    }
    let signed_moment = |n: i64| {
        let value = sine_moments[n.unsigned_abs() as usize].clone();
        if n < 0 {
            value.neg()
        } else {
            value
        }
    };

    let dimension = config.dimension();
    let count = dimension * dimension;
    let mut w02 = vec![zero.to_rational_interval(); count];
    let mut wr = w02.clone();
    let mut wp = w02.clone();
    let mut tau = w02.clone();
    let l_squared = l.square();
    let sixteen_pi_squared = pi.square().mul(&MpfrInterval::from_i64(16, p));
    let sinh_squared = sinh(&l.div(&four)?)?.square();

    for row in 0..dimension {
        let n = row as i64 - config.modes as i64;
        for column in row..dimension {
            let m = column as i64 - config.modes as i64;
            let nf = MpfrInterval::from_i64(n, p);
            let mf = MpfrInterval::from_i64(m, p);
            let numerator = l_squared.sub(&sixteen_pi_squared.mul(&mf).mul(&nf));
            let denominator = l_squared
                .add(&sixteen_pi_squared.mul(&mf.square()))
                .mul(&l_squared.add(&sixteen_pi_squared.mul(&nf.square())));
            let w02_cell = sinh_squared
                .mul(&MpfrInterval::from_i64(32, p))
                .mul(&l)
                .mul(&numerator)
                .div(&denominator)?;

            let wr_cell = if n == m {
                kappa
                    .add(&special[n.unsigned_abs() as usize].cc.mul(&two))
                    .add(&j)
                    .sub(&special[n.unsigned_abs() as usize].xc.mul(&two).div(&l)?)
            } else {
                signed_s(&special, m)
                    .sub(&signed_s(&special, n))
                    .div(&pi.mul(&MpfrInterval::from_i64(n - m, p)))?
            };

            let wp_cell = if n == m {
                diagonal_moments[n.unsigned_abs() as usize].clone()
            } else {
                signed_moment(m)
                    .sub(&signed_moment(n))
                    .div(&pi.mul(&MpfrInterval::from_i64(n - m, p)))?
            };
            let tau_cell = w02_cell.sub(&wr_cell).sub(&wp_cell);
            let indices = [row * dimension + column, column * dimension + row];
            for index in indices {
                w02[index] = w02_cell.to_rational_interval();
                wr[index] = wr_cell.to_rational_interval();
                wp[index] = wp_cell.to_rational_interval();
                tau[index] = tau_cell.to_rational_interval();
            }
        }
    }

    Ok(CutoffFreeMatrix {
        config: config.clone(),
        scalar_backend: format!("system-flint-arb-{}", backend_version()),
        w02,
        wr,
        wp,
        tau,
    })
}

pub fn certify(config: &CutoffFreeConfig) -> Result<(CutoffFreeMatrix, IntervalInertiaResult)> {
    let matrix = assemble(config).context("assemble cutoff-free CCM matrix")?;
    let inertia = matrix
        .certify_inertia()
        .context("certify cutoff-free CCM inertia")?;
    Ok((matrix, inertia))
}

pub fn certify_portable(
    config: &CutoffFreeConfig,
) -> Result<(CutoffFreeMatrix, PortableIntervalInertiaCertificate)> {
    let matrix = assemble(config).context("assemble cutoff-free CCM matrix")?;
    let certificate = matrix
        .portable_inertia_certificate()
        .context("build portable cutoff-free CCM inertia certificate")?;
    Ok((matrix, certificate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutoff_free_components_reconstruct_symmetric_tau() {
        let matrix = assemble(&CutoffFreeConfig::new(5, 2, 192)).unwrap();
        assert!(matrix.scalar_backend.starts_with("system-flint-arb-"));
        let dimension = matrix.dimension();
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                let transpose = column * dimension + row;
                assert_eq!(matrix.tau[index], matrix.tau[transpose]);
                let reconstructed = matrix.w02[index]
                    .sub(&matrix.wr[index])
                    .sub(&matrix.wp[index]);
                assert!(reconstructed.intersection(&matrix.tau[index]).is_some());
            }
        }
    }

    #[test]
    fn published_small_positive_matrix_is_certified() {
        let (_, certificate) = certify_portable(&CutoffFreeConfig::new(13, 4, 256)).unwrap();
        assert_eq!(certificate.positive, 9);
        assert_eq!(certificate.negative, 0);
        assert_eq!(certificate.zero_or_unresolved, 0);
        let encoded = serde_json::to_vec(&certificate).unwrap();
        let decoded: PortableIntervalInertiaCertificate = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, certificate);
        let report = xc_certify::exact::verify_portable_interval_inertia_certificate(&decoded);
        assert!(report.valid, "{:?}", report.errors);
        assert!(report
            .checks
            .iter()
            .any(|check| check.contains("independently replayed")));
    }

    #[test]
    #[ignore = "publication-scale 401x401, 9000-bit permanent regression"]
    fn groskin_c100_n200_publication_certificate() {
        let (_, inertia) = certify(&CutoffFreeConfig::new(100, 200, 9000)).unwrap();
        assert!(matches!(
            inertia,
            IntervalInertiaResult::Conclusive {
                positive: 401,
                negative: 0,
                ..
            }
        ));
    }
}

#[cfg(test)]
mod endpoint_regression {
    use super::*;

    #[test]
    fn zero_mode_matches_independent_defining_integral() {
        for (c, expected) in [
            (5, "2.25608966855868498643015465180094363462904"),
            (13, "2.88506309771709566996707209569382572588128"),
            (100, "3.67872561806049666284203395968608485646508"),
        ] {
            let matrix = assemble(&CutoffFreeConfig::new(c, 0, 192)).unwrap();
            let actual = Float::with_val(192, matrix.wr[0].midpoint());
            let expected = Float::with_val(192, Float::parse(expected).unwrap());
            let error = Float::with_val(192, actual - expected).abs();
            let tolerance = Float::with_val(192, Float::parse("1e-37").unwrap());
            assert!(
                error < tolerance,
                "zero-mode finite-endpoint regression at c={c}: {error}"
            );
        }
    }

    #[test]
    fn analytic_tail_budget_grows_with_precision() {
        let low = CutoffFreeConfig::new(13, 4, 256);
        let high = CutoffFreeConfig::new(13, 4, 2048);
        assert!(high.geometric_terms > low.geometric_terms);
        assert!(recommended_geometric_terms(100, 4, 256) < low.geometric_terms);
        let b = 3_u64;
        let d_bits = u64::from(usize::BITS - low.dimension().leading_zeros());
        assert!(2 * b * low.geometric_terms as u64 >= 256 + 2 * d_bits + 16);
    }

    #[test]
    fn zero_frequency_keeps_exact_vanishing_integrals() {
        let p = 192;
        let l = MpfrInterval::from_u64(13, p).ln().unwrap();
        let zero = MpfrInterval::from_i64(0, p);
        let (psi, _) = complex_digamma(&q(1, 4, p), &zero).unwrap();
        let values = special_values(0, &l, &MpfrInterval::pi(p), &psi, 64).unwrap();
        assert_eq!(values.s.to_rational_interval(), zero.to_rational_interval());
        assert_eq!(
            values.cc.to_rational_interval(),
            zero.to_rational_interval()
        );
        let (infinite, _) = complex_trigamma(&q(1, 4, p), &zero).unwrap();
        assert!(values.xc.upper() < infinite.div(&MpfrInterval::from_i64(4, p)).unwrap().lower());
    }

    #[test]
    fn aggregate_prime_entries_agree_with_direct_interval_sum() {
        let cfg = CutoffFreeConfig::new(13, 3, 192);
        let matrix = assemble(&cfg).unwrap();
        let p = cfg.precision_bits;
        let zero = MpfrInterval::from_i64(0, p);
        let one = MpfrInterval::from_i64(1, p);
        let two = MpfrInterval::from_i64(2, p);
        let pi = MpfrInterval::pi(p);
        let l = MpfrInterval::from_u64(13, p).ln().unwrap();
        for n in -3_i64..=3 {
            for m in -3_i64..=3 {
                let mut direct = zero.clone();
                for (power, prime, _) in prime_powers_up_to(13) {
                    let x = MpfrInterval::from_u64(power, p).ln().unwrap();
                    let phase = |mode: i64| {
                        pi.mul(&two)
                            .mul(&MpfrInterval::from_i64(mode, p))
                            .mul(&x)
                            .div(&l)
                            .unwrap()
                    };
                    let kernel = if n == m {
                        one.sub(&x.div(&l).unwrap()).mul(&two).mul(&phase(n).cos())
                    } else {
                        phase(m)
                            .sin()
                            .sub(&phase(n).sin())
                            .div(&pi.mul(&MpfrInterval::from_i64(n - m, p)))
                            .unwrap()
                    };
                    direct = direct.add(
                        &kernel
                            .mul(&MpfrInterval::from_u64(prime, p).ln().unwrap())
                            .div(&MpfrInterval::from_u64(power, p).sqrt().unwrap())
                            .unwrap(),
                    );
                }
                let index = (n + 3) as usize * 7 + (m + 3) as usize;
                assert!(matrix.wp[index]
                    .intersection(&direct.to_rational_interval())
                    .is_some());
            }
        }
    }
}
