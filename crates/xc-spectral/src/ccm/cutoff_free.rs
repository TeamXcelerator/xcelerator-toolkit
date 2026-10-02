//! Certified cutoff-free CCM Weil-matrix assembly.
//!
//! This is an independent implementation of the closed-form mathematics in
//! Groskin, arXiv:2607.02828.  It evaluates the complex digamma and trigamma
//! terms with system FLINT/Arb balls, encloses every remaining operation with
//! outward-rounded MPFR intervals, and retains the `W02`, `WR`, and `Wp`
//! components separately for cancellation audit.

use super::arb_bridge::{backend_version, complex_digamma, complex_trigamma};
use super::try_prime_powers_up_to;
use anyhow::{bail, Context, Result};
use rug::{Float, Rational};
use xc_cache::ContentDigest;
use xc_certify::exact::{
    build_portable_interval_inertia_certificate_mpfr, interval_record,
    interval_symmetric_ldlt_inertia_mpfr, IntervalInertiaResult,
};
use xc_certify::{ExactRationalIntervalRecord, PortableIntervalInertiaCertificate};
use xc_numerics::interval::RationalInterval;
use xc_numerics::mpfr_interval::MpfrInterval;

/// Corrected finite-endpoint and aggregate-prime assembly identity.
/// Old inertia records may remain readable as records, but are not evidence
/// for this assembly. Sector certificates independently version their schema.
pub const ASSEMBLY_SEMANTICS: &str = "ccm-cutoff-free-zero-endpoint-aggregate-primes-v0.15.0-v1";

/// Schema-2 inertia arithmetic; independent of the unchanged assembly identity.
pub const INERTIA_SEMANTICS: &str = "mpfr-directed-interval-ldlt-v1";

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
        if !(64..=1_000_000).contains(&self.precision_bits) {
            bail!("cutoff-free CCM precision must be in 64..=1000000 bits");
        }
        if self.geometric_terms == 0 {
            bail!("cutoff-free CCM requires at least one geometric correction term");
        }
        if usize::try_from(self.integer_cutoff_c)
            .ok()
            .and_then(|n| n.checked_add(1))
            .is_none_or(|n| n > isize::MAX as usize)
        {
            bail!("cutoff-free CCM prime sieve bound exceeds platform capacity");
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

    /// Checked dimension for configurations received from external callers.
    pub fn checked_dimension(&self) -> Result<usize> {
        self.modes
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("cutoff-free CCM dimension overflows"))
    }

    /// Return 2*N+1. Panics if an unvalidated configuration overflows usize;
    /// use `checked_dimension` for externally supplied configurations.
    pub fn dimension(&self) -> usize {
        self.checked_dimension()
            .expect("cutoff-free CCM dimension overflows")
    }
}

/// Assembly produced by [`assemble`]. Public fields remain readable for consumers.
/// Mutating any field invalidates its private assembly binding; certificate and
/// evidence methods reject the changed object. Arbitrary interval matrices can
/// instead use the generic exact-inertia APIs without a CCM assembly claim.
#[derive(Clone, Debug)]
pub struct CutoffFreeMatrix {
    pub config: CutoffFreeConfig,
    pub scalar_backend: String,
    pub w02: Vec<RationalInterval>,
    pub wr: Vec<RationalInterval>,
    pub wp: Vec<RationalInterval>,
    pub tau: Vec<RationalInterval>,
    assembly_binding: ContentDigest,
}

impl CutoffFreeMatrix {
    pub fn dimension(&self) -> usize {
        self.config.dimension()
    }

    pub fn certify_inertia(&self) -> Result<IntervalInertiaResult> {
        self.validate_assembly()?;
        interval_symmetric_ldlt_inertia_mpfr(
            &self.tau,
            self.dimension(),
            self.config.precision_bits,
        )
        .map_err(anyhow::Error::from)
    }

    /// Check dimensions, exact component reconstruction, symmetry, and the
    /// immutable binding recorded by the assembler. This establishes provenance
    /// within this process; generic portable inertia replay alone checks only
    /// the matrix endpoints, not the special-function assembly theorem.
    pub fn validate_assembly(&self) -> Result<()> {
        self.validate_structure()?;
        if self.current_assembly_binding()? != self.assembly_binding {
            bail!("cutoff-free CCM assembly was modified after construction");
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<()> {
        self.config.validate()?;
        let dimension = self.config.checked_dimension()?;
        let count = dimension
            .checked_mul(dimension)
            .ok_or_else(|| anyhow::anyhow!("cutoff-free CCM matrix size overflows"))?;
        if self.scalar_backend.is_empty()
            || [&self.w02, &self.wr, &self.wp, &self.tau]
                .iter()
                .any(|values| values.len() != count)
        {
            bail!("cutoff-free CCM component shape or backend is invalid");
        }
        for row in 0..dimension {
            for column in row..dimension {
                let index = row * dimension + column;
                let transpose = column * dimension + row;
                for values in [&self.w02, &self.wr, &self.wp, &self.tau] {
                    if values[index] != values[transpose] {
                        bail!("cutoff-free CCM component is not exactly symmetric");
                    }
                }
                let reconstructed = self.w02[index].sub(&self.wr[index]).sub(&self.wp[index]);
                if self.tau[index].lower() > reconstructed.lower()
                    || self.tau[index].upper() < reconstructed.upper()
                {
                    bail!("cutoff-free CCM tau does not enclose its component reconstruction");
                }
            }
        }
        Ok(())
    }

    fn current_assembly_binding(&self) -> Result<ContentDigest> {
        let evidence = (
            "ccm-cutoff-free-in-memory-assembly-binding-v1",
            self.component_digest_unchecked()?,
            interval_records(&self.tau),
        );
        json_digest(&evidence)
    }

    /// Retain this assembly's record binding for later binds in this
    /// process. `component_evidence_digest` must be the value returned by
    /// [`Self::component_evidence_digest`] for this matrix, so the matrix has
    /// passed `validate_assembly`.
    pub(crate) fn retain_binding(
        &self,
        component_evidence_digest: ContentDigest,
    ) -> Result<AssemblyRecordBinding> {
        let binding = AssemblyRecordBinding {
            scalar_backend: self.scalar_backend.clone(),
            component_evidence_digest,
            tau_records_digest: records_digest(&interval_records(&self.tau))?,
        };
        let mut retained = VALIDATED_ASSEMBLIES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        retained.retain(|(config, _)| config != &self.config);
        if retained.len() >= 8 {
            retained.remove(0);
        }
        retained.push((self.config.clone(), binding.clone()));
        Ok(binding)
    }

    /// Digest of the exact `W02`, `WR`, and `Wp` interval records used to
    /// assemble this matrix.  Derived certificates can bind the same
    /// component evidence without first running a full inertia proof.
    pub fn component_evidence_digest(&self) -> Result<ContentDigest> {
        self.validate_assembly()?;
        self.component_digest_unchecked()
    }

    fn component_digest_unchecked(&self) -> Result<ContentDigest> {
        let component_evidence = (
            ASSEMBLY_SEMANTICS,
            self.config.integer_cutoff_c,
            self.config.modes,
            self.config.precision_bits,
            self.config.geometric_terms,
            &self.scalar_backend,
            interval_records(&self.w02),
            interval_records(&self.wr),
            interval_records(&self.wp),
        );
        json_digest(&component_evidence).context("serialize cutoff-free component evidence")
    }

    pub fn portable_inertia_certificate(&self) -> Result<PortableIntervalInertiaCertificate> {
        self.validate_assembly()?;
        build_portable_interval_inertia_certificate_mpfr(
            &self.tau,
            self.dimension(),
            self.config.precision_bits,
            self.scalar_backend.clone(),
            self.component_digest_unchecked()?,
            std::collections::BTreeMap::from([
                ("inertia_semantics".to_owned(), INERTIA_SEMANTICS.to_owned()),
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

/// Exact records of `values` in order; each record is converted on a worker.
pub(crate) fn interval_records(values: &[RationalInterval]) -> Vec<ExactRationalIntervalRecord> {
    use rayon::prelude::*;
    values.par_iter().map(interval_record).collect()
}

/// Digest of the canonical JSON bytes of exact interval records.
pub(crate) fn records_digest(records: &[ExactRationalIntervalRecord]) -> Result<ContentDigest> {
    json_digest(&records)
}

/// `ContentDigest::sha256(&serde_json::to_vec(value)?)`, hashing the same
/// bytes as they are written instead of buffering them.
fn json_digest(value: &impl serde::Serialize) -> Result<ContentDigest> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    serde_json::to_writer(&mut hash, value)?;
    Ok(ContentDigest(format!("{:x}", hash.finalize())))
}

/// What a sector certificate binds of one validated cutoff-free assembly:
/// its backend, component evidence digest, and Tau record digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AssemblyRecordBinding {
    pub scalar_backend: String,
    pub component_evidence_digest: ContentDigest,
    pub tau_records_digest: ContentDigest,
}

// Bindings of assemblies validated in this process, keyed by their exact
// configuration. Assembly is a deterministic function of the configuration,
// so a retained binding equals that of a fresh assembly. Never holds failures.
static VALIDATED_ASSEMBLIES: std::sync::Mutex<Vec<(CutoffFreeConfig, AssemblyRecordBinding)>> =
    std::sync::Mutex::new(Vec::new());

/// Record binding of the assembly at `config`: the binding retained by this
/// process for that exact configuration, or else a fresh validated assembly.
pub(crate) fn assembly_binding(config: &CutoffFreeConfig) -> Result<AssemblyRecordBinding> {
    let retained = VALIDATED_ASSEMBLIES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .find(|(key, _)| key == config)
        .map(|(_, binding)| binding.clone());
    if let Some(binding) = retained {
        return Ok(binding);
    }
    let matrix = assemble(config)?;
    let component_evidence_digest = matrix.component_evidence_digest()?;
    matrix.retain_binding(component_evidence_digest)
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

/// Mode-dependent geometric correction sums with their enclosed analytic
/// tail. Pure MPFR arithmetic, so modes may be evaluated on any thread.
struct GeometricSums {
    w: MpfrInterval,
    gs: MpfrInterval,
    gcc: MpfrInterval,
    gx1: MpfrInterval,
    gx2: MpfrInterval,
}

fn geometric_sums(
    n: usize,
    l: &MpfrInterval,
    pi: &MpfrInterval,
    terms: usize,
) -> Result<GeometricSums> {
    let p = l.precision();
    let n_interval = MpfrInterval::from_u64(n as u64, p);
    let b = pi.mul(&n_interval).div(l)?;
    let w = b.mul(&MpfrInterval::from_i64(2, p));

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
    Ok(GeometricSums {
        w,
        gs: gs.add(&positive_tail),
        gcc: gcc.add(&positive_tail),
        gx1: gx1.add(&positive_tail),
        gx2: gx2.add(&signed_tail),
    })
}

#[cfg(test)]
fn special_values(
    n: usize,
    l: &MpfrInterval,
    pi: &MpfrInterval,
    psi_quarter: &MpfrInterval,
    terms: usize,
) -> Result<SpecialValues> {
    special_values_with(n, l, pi, psi_quarter, || geometric_sums(n, l, pi, terms))
}

/// Special values of mode `n`. The Arb calls run on the calling thread in
/// mode order: Arb caches constants per thread, so they must keep their
/// original thread and sequence. `sums` supplies the pure MPFR part and is
/// consulted where the serial evaluation computed it, preserving error order.
fn special_values_with(
    n: usize,
    l: &MpfrInterval,
    pi: &MpfrInterval,
    psi_quarter: &MpfrInterval,
    sums: impl FnOnce() -> Result<GeometricSums>,
) -> Result<SpecialValues> {
    let p = l.precision();
    // n=0 needs the SAME finite-endpoint correction as every other mode.
    // psi_1(1/4)/4 alone is the integral over [0,infinity), not [0,L].
    let n_interval = MpfrInterval::from_u64(n as u64, p);
    let b = pi.mul(&n_interval).div(l)?;
    let quarter = q(1, 4, p);
    let (digamma_re, digamma_im) = complex_digamma(&quarter, &b)?;
    let (trigamma_re, _) = complex_trigamma(&quarter, &b)?;
    let GeometricSums {
        w,
        gs,
        gcc,
        gx1,
        gx2,
    } = sums()?;

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

/// Interval components of the cutoff-free finite Weil form, in centered
/// row-major order, with `tau = w02 - wr - wp` enclosed entrywise.
#[derive(Clone, Debug)]
pub struct CutoffFreeComponents {
    pub w02: Vec<RationalInterval>,
    pub wr: Vec<RationalInterval>,
    pub wp: Vec<RationalInterval>,
    pub tau: Vec<RationalInterval>,
}

/// Assemble every component for a cutoff enclosed by `c` (precision taken
/// from `c`). The closed forms depend on the cutoff only through
/// L = ln(c), sqrt(c), (c-1)/(c+1) and the prime powers up to `prime_cutoff`
/// = floor(c), so a fractional cutoff is admissible.
fn assemble_components(
    c: &MpfrInterval,
    prime_cutoff: u64,
    modes: usize,
    geometric_terms: usize,
) -> Result<CutoffFreeComponents> {
    let dimension = 2 * modes + 1;
    let zero = MpfrInterval::from_i64(0, c.precision()).to_rational_interval();
    let mut w02 = vec![zero; dimension * dimension];
    let mut wr = w02.clone();
    let mut wp = w02.clone();
    let mut tau = w02.clone();
    visit_mapped_cells(
        c,
        prime_cutoff,
        modes,
        geometric_terms,
        |_, _, cell| Ok(cell.each_ref().map(MpfrInterval::to_rational_interval)),
        |row, column, [w02_cell, wr_cell, wp_cell, tau_cell]| {
            for index in [row * dimension + column, column * dimension + row] {
                w02[index] = w02_cell.clone();
                wr[index] = wr_cell.clone();
                wp[index] = wp_cell.clone();
                tau[index] = tau_cell.clone();
            }
            Ok(())
        },
    )?;
    Ok(CutoffFreeComponents { w02, wr, wp, tau })
}

/// Visit every upper-triangle cell (row <= column) with its W02, WR, Wp and
/// Tau enclosures, without retaining the dense matrices.
pub(crate) fn visit_cells<F>(
    c: &MpfrInterval,
    prime_cutoff: u64,
    modes: usize,
    geometric_terms: usize,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(
        usize,
        usize,
        &MpfrInterval,
        &MpfrInterval,
        &MpfrInterval,
        &MpfrInterval,
    ) -> Result<()>,
{
    visit_mapped_cells(
        c,
        prime_cutoff,
        modes,
        geometric_terms,
        |_, _, cell| Ok(cell),
        |row, column, [w02, wr, wp, tau]| visit(row, column, &w02, &wr, &wp, &tau),
    )
}

/// Cells per parallel block; bounds the enclosures held before visiting.
const CELL_BLOCK: usize = 4096;

/// Evaluate the upper-triangle cells in parallel blocks and map each cell on
/// its worker, then visit them serially in row-major order. Each enclosure
/// keeps the serial MPFR operation sequence, the Arb special functions keep
/// their thread and call order, and the first failure in serial order wins.
fn visit_mapped_cells<T, M, F>(
    c: &MpfrInterval,
    prime_cutoff: u64,
    modes: usize,
    geometric_terms: usize,
    map: M,
    mut visit: F,
) -> Result<()>
where
    T: Send,
    M: Fn(usize, usize, [MpfrInterval; 4]) -> Result<T> + Sync,
    F: FnMut(usize, usize, T) -> Result<()>,
{
    use rayon::prelude::*;
    xc_numerics::mpfr_interval::ensure_uniform_exponent_range()?;
    let p = c.precision();
    let l = c.ln()?;
    let pi = MpfrInterval::pi(p);
    let zero = MpfrInterval::from_i64(0, p);
    let quarter = q(1, 4, p);
    let (psi_quarter, _) = complex_digamma(&quarter, &zero)?;
    let mut sums = (0..=modes)
        .into_par_iter()
        .map(|n| Some(geometric_sums(n, &l, &pi, geometric_terms)))
        .collect::<Vec<_>>();
    let special: Vec<SpecialValues> = (0..=modes)
        .map(|n| {
            special_values_with(n, &l, &pi, &psi_quarter, || {
                sums[n]
                    .take()
                    .expect("each mode's geometric sums are used once")
            })
        })
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
        try_prime_powers_up_to(prime_cutoff)?
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
    let moments = (0..=modes)
        .into_par_iter()
        .map(|n| -> Result<(MpfrInterval, MpfrInterval)> {
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
            Ok((sine, diagonal))
        })
        .collect::<Vec<_>>();
    let mut sine_moments = Vec::with_capacity(modes + 1);
    let mut diagonal_moments = Vec::with_capacity(modes + 1);
    for moment in moments {
        let (sine, diagonal) = moment?;
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

    let dimension = 2 * modes + 1;
    let l_squared = l.square();
    let sixteen_pi_squared = pi.square().mul(&MpfrInterval::from_i64(16, p));
    let sinh_squared = sinh(&l.div(&four)?)?.square();

    let cell = |row: usize, column: usize| -> Result<T> {
        let n = row as i64 - modes as i64;
        let m = column as i64 - modes as i64;
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
        map(row, column, [w02_cell, wr_cell, wp_cell, tau_cell])
    };

    let mut cells =
        (0..dimension).flat_map(|row| (row..dimension).map(move |column| (row, column)));
    let mut block = Vec::with_capacity(CELL_BLOCK);
    loop {
        block.clear();
        block.extend(cells.by_ref().take(CELL_BLOCK));
        if block.is_empty() {
            return Ok(());
        }
        let values = block
            .par_iter()
            .map(|&(row, column)| cell(row, column))
            .collect::<Vec<_>>();
        for (&(row, column), value) in block.iter().zip(values) {
            visit(row, column, value?)?;
        }
    }
}

/// Number of geometric correction terms for an arbitrary cutoff c > 1. The
/// analytic tail after M terms is at most 4*exp(-(4M+1)L/2)/(1-exp(-2L)); this
/// chooses M so it is below 2^-(p + 2*dimension_bits + 16). Integer cutoffs
/// keep `recommended_geometric_terms`. The tail is always enclosed, so this
/// only controls enclosure width, never validity.
pub fn geometric_terms_for_length(log_cutoff: f64, modes: usize, precision_bits: u32) -> usize {
    let d = modes.saturating_mul(2).saturating_add(1);
    let dimension_bits = f64::from(usize::BITS - d.leading_zeros());
    let required = f64::from(precision_bits) + 2.0 * dimension_bits + 16.0;
    let terms = (required * std::f64::consts::LN_2 / (2.0 * log_cutoff)).ceil();
    if terms.is_finite() && terms >= 1.0 {
        (terms as usize).min(10_000_000)
    } else {
        1
    }
}

/// Resolve an exact decimal cutoff, which may be fractional.
pub(crate) fn resolve_cutoff(
    cutoff: &str,
    modes: usize,
    precision_bits: u32,
) -> Result<ResolvedCutoff> {
    let exact = super::research::ExactCutoff::parse(cutoff)?;
    let value = exact.value().clone();
    if value <= 1 {
        bail!("cutoff-free CCM requires a cutoff greater than one");
    }
    let prime_cutoff = exact.prime_cutoff();
    let integer = (value.denom() == &1).then_some(prime_cutoff);
    let geometric_terms = match integer {
        Some(c) => recommended_geometric_terms(c, modes, precision_bits),
        None => geometric_terms_for_length(value.to_f64().ln(), modes, precision_bits),
    };
    let config = CutoffFreeConfig {
        integer_cutoff_c: prime_cutoff.max(2),
        modes,
        precision_bits,
        geometric_terms,
    };
    config.validate()?;
    let c = MpfrInterval::from_rational(&value, precision_bits);
    Ok(ResolvedCutoff {
        c,
        prime_cutoff,
        geometric_terms,
    })
}

/// Exact cutoff enclosure, prime cutoff and analytic term count.
pub(crate) struct ResolvedCutoff {
    pub c: MpfrInterval,
    pub prime_cutoff: u64,
    pub geometric_terms: usize,
}

/// Assemble the cutoff-free components for an exact decimal cutoff, which may
/// be fractional, at `precision_bits`. Integer text reproduces `assemble`.
pub fn assemble_components_at_cutoff(
    cutoff: &str,
    modes: usize,
    precision_bits: u32,
) -> Result<CutoffFreeComponents> {
    let resolved = resolve_cutoff(cutoff, modes, precision_bits)?;
    assemble_components(
        &resolved.c,
        resolved.prime_cutoff,
        modes,
        resolved.geometric_terms,
    )
}

pub fn assemble(config: &CutoffFreeConfig) -> Result<CutoffFreeMatrix> {
    config.validate()?;
    let p = config.precision_bits;
    let c = MpfrInterval::from_u64(config.integer_cutoff_c, p);
    let CutoffFreeComponents { w02, wr, wp, tau } = assemble_components(
        &c,
        config.integer_cutoff_c,
        config.modes,
        config.geometric_terms,
    )?;
    let mut matrix = CutoffFreeMatrix {
        config: config.clone(),
        scalar_backend: format!("system-flint-arb-{}", backend_version()),
        w02,
        wr,
        wp,
        tau,
        assembly_binding: ContentDigest::sha256(b"unsealed"),
    };
    matrix.validate_structure()?;
    matrix.assembly_binding = matrix.current_assembly_binding()?;
    Ok(matrix)
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

    type CellLog = Vec<(usize, usize, [(u32, RationalInterval); 4])>;

    fn log_cell(log: &mut CellLog, row: usize, column: usize, cells: [&MpfrInterval; 4]) {
        log.push((
            row,
            column,
            cells.map(|cell| (cell.precision(), cell.to_rational_interval())),
        ));
    }

    #[test]
    fn parallel_cell_visit_is_bit_identical_to_serial_reference_at_any_thread_count() {
        for (cutoff, modes, bits) in [("13", 6, 512), ("5", 3, 192), ("13.5", 4, 320)] {
            let resolved = resolve_cutoff(cutoff, modes, bits).unwrap();
            let visit = |log: &mut CellLog, parallel: bool| {
                let record = |row, column, a: &_, b: &_, c: &_, d: &_| {
                    log_cell(log, row, column, [a, b, c, d]);
                    Ok(())
                };
                let (c, primes, terms) =
                    (&resolved.c, resolved.prime_cutoff, resolved.geometric_terms);
                if parallel {
                    visit_cells(c, primes, modes, terms, record)
                } else {
                    super::reference::reference_visit_cells(c, primes, modes, terms, record)
                }
            };
            let mut expected = CellLog::new();
            visit(&mut expected, false).unwrap();
            assert_eq!(expected.len(), (2 * modes + 1) * (2 * modes + 2) / 2);
            let stop = expected[expected.len() / 2].clone();
            for threads in [1, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for _ in 0..2 {
                    let mut actual = CellLog::new();
                    pool.install(|| visit(&mut actual, true)).unwrap();
                    assert_eq!(actual, expected, "{cutoff} at {threads} threads");
                    // The first failure in row-major order ends the visit.
                    let mut visited = 0;
                    let error = pool
                        .install(|| {
                            visit_cells(
                                &resolved.c,
                                resolved.prime_cutoff,
                                modes,
                                resolved.geometric_terms,
                                |row, column, _, _, _, _| {
                                    visited += 1;
                                    if (row, column) == (stop.0, stop.1) {
                                        bail!("stop at {row},{column}");
                                    }
                                    Ok(())
                                },
                            )
                        })
                        .unwrap_err();
                    assert_eq!(error.to_string(), format!("stop at {},{}", stop.0, stop.1));
                    assert_eq!(visited, expected.len() / 2 + 1);
                }
            }
        }
    }

    #[test]
    fn parallel_assembly_and_records_match_serial_reference() {
        let config = CutoffFreeConfig::new(13, 5, 384);
        let dimension = config.dimension();
        let zero = MpfrInterval::from_i64(0, 384).to_rational_interval();
        let mut expected = vec![vec![zero; dimension * dimension]; 4];
        super::reference::reference_visit_cells(
            &MpfrInterval::from_u64(13, 384),
            13,
            config.modes,
            config.geometric_terms,
            |row, column, a, b, c, d| {
                for index in [row * dimension + column, column * dimension + row] {
                    for (values, cell) in expected.iter_mut().zip([a, b, c, d]) {
                        values[index] = cell.to_rational_interval();
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let serial_records =
            |values: &[RationalInterval]| values.iter().map(interval_record).collect::<Vec<_>>();
        let serial_component_digest = ContentDigest(xc_cache::sha256_hex(
            &serde_json::to_vec(&(
                ASSEMBLY_SEMANTICS,
                config.integer_cutoff_c,
                config.modes,
                config.precision_bits,
                config.geometric_terms,
                format!("system-flint-arb-{}", backend_version()),
                serial_records(&expected[0]),
                serial_records(&expected[1]),
                serial_records(&expected[2]),
            ))
            .unwrap(),
        ));
        for threads in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let matrix = pool.install(|| assemble(&config)).unwrap();
            for (actual, expected) in [&matrix.w02, &matrix.wr, &matrix.wp, &matrix.tau]
                .into_iter()
                .zip(&expected)
            {
                assert_eq!(actual, expected);
                assert_eq!(
                    pool.install(|| interval_records(actual)),
                    serial_records(expected)
                );
            }
            let component = pool.install(|| matrix.component_evidence_digest()).unwrap();
            assert_eq!(component, serial_component_digest);
            // A retained binding is exactly the binding of a fresh assembly.
            let fresh = AssemblyRecordBinding {
                scalar_backend: matrix.scalar_backend.clone(),
                component_evidence_digest: component.clone(),
                tau_records_digest: ContentDigest::sha256(
                    &serde_json::to_vec(&serial_records(&expected[3])).unwrap(),
                ),
            };
            assert_eq!(matrix.retain_binding(component).unwrap(), fresh);
            assert_eq!(assembly_binding(&config).unwrap(), fresh);
        }
    }

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
                for (power, prime, _) in super::super::prime_powers_up_to(13) {
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

/// Serial assembly retained verbatim from before the parallel visit, used to
/// prove the parallel visit bit-identical.
#[cfg(test)]
mod reference {
    use super::*;

    fn reference_special_values(
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

    /// Visit every upper-triangle cell (row <= column) with its W02, WR, Wp and
    /// Tau enclosures, without retaining the dense matrices.
    pub(super) fn reference_visit_cells<F>(
        c: &MpfrInterval,
        prime_cutoff: u64,
        modes: usize,
        geometric_terms: usize,
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(
            usize,
            usize,
            &MpfrInterval,
            &MpfrInterval,
            &MpfrInterval,
            &MpfrInterval,
        ) -> Result<()>,
    {
        let p = c.precision();
        let l = c.ln()?;
        let pi = MpfrInterval::pi(p);
        let zero = MpfrInterval::from_i64(0, p);
        let quarter = q(1, 4, p);
        let (psi_quarter, _) = complex_digamma(&quarter, &zero)?;
        let special: Vec<SpecialValues> = (0..=modes)
            .map(|n| reference_special_values(n, &l, &pi, &psi_quarter, geometric_terms))
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
            try_prime_powers_up_to(prime_cutoff)?
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
        let mut sine_moments = Vec::with_capacity(modes + 1);
        let mut diagonal_moments = Vec::with_capacity(modes + 1);
        for n in 0..=modes {
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

        let dimension = 2 * modes + 1;
        let l_squared = l.square();
        let sixteen_pi_squared = pi.square().mul(&MpfrInterval::from_i64(16, p));
        let sinh_squared = sinh(&l.div(&four)?)?.square();

        for row in 0..dimension {
            let n = row as i64 - modes as i64;
            for column in row..dimension {
                let m = column as i64 - modes as i64;
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
                visit(row, column, &w02_cell, &wr_cell, &wp_cell, &tau_cell)?;
            }
        }
        Ok(())
    }
}
