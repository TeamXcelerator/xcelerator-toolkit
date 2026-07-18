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
            geometric_terms: 32,
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

    pub fn portable_inertia_certificate(&self) -> Result<PortableIntervalInertiaCertificate> {
        let component_evidence = (
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
        build_portable_interval_inertia_certificate(
            &self.tau,
            self.dimension(),
            self.config.precision_bits,
            self.scalar_backend.clone(),
            ContentDigest(sha256_hex(&evidence_bytes)),
            std::collections::BTreeMap::from([
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
    if n == 0 {
        let zero = MpfrInterval::from_i64(0, p);
        let quarter = q(1, 4, p);
        let (trigamma, _) = complex_trigamma(&quarter, &zero)?;
        return Ok(SpecialValues {
            s: zero.clone(),
            cc: zero,
            xc: trigamma.div(&MpfrInterval::from_i64(4, p))?,
        });
    }

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
    Ok(SpecialValues { s, cc, xc })
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

            let mut wp_cell = zero.clone();
            for (log_power, log_prime, sqrt_power) in &prime_data {
                let q_cell = if n == m {
                    let phase = pi.mul(&two).mul(&nf).mul(log_power).div(&l)?;
                    one.sub(&log_power.div(&l)?).mul(&two).mul(&phase.cos())
                } else {
                    let m_phase = pi.mul(&two).mul(&mf).mul(log_power).div(&l)?;
                    let n_phase = pi.mul(&two).mul(&nf).mul(log_power).div(&l)?;
                    m_phase
                        .sin()
                        .sub(&n_phase.sin())
                        .div(&pi.mul(&MpfrInterval::from_i64(n - m, p)))?
                };
                wp_cell = wp_cell.add(&q_cell.mul(log_prime).div(sqrt_power)?);
            }
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
