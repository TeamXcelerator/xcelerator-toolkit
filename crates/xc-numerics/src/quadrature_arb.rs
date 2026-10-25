// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

use rug::Float;
use serde::{Deserialize, Serialize};
use xc_cache::CacheError;

/// Explicit HP rule generation semantics. Selecting a backend never falls back.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GaussLegendreGenerator {
    /// Released point-Newton computation and all released cache identities.
    #[default]
    LegacyNewton,
    /// Certified FLINT/Arb enclosures, uniquely rounded nearest/ties-to-even.
    ArbCertified,
}

impl GaussLegendreGenerator {
    pub fn is_default(&self) -> bool {
        *self == Self::LegacyNewton
    }
    pub fn is_legacy(&self) -> bool {
        self.is_default()
    }
    pub fn semantic_id(self) -> &'static str {
        match self {
            Self::LegacyNewton => "newton-legendre-roots-v1",
            Self::ArbCertified => "arb-certified-nearest-v1",
        }
    }
    /// Checked before lookup, including RequireReuse and compatibility paths.
    pub fn ensure_available(self) -> Result<(), CacheError> {
        if self == Self::ArbCertified && !cfg!(feature = "arb") {
            return Err(error("certified Arb quadrature requires feature arb"));
        }
        Ok(())
    }
    /// Validate generation limits before a dependent artifact lookup. This
    /// admits a bounded request; it does not promise the guard ladder resolves.
    pub fn validate_rule(self, order: usize, precision_bits: u32) -> Result<(), CacheError> {
        self.ensure_available()?;
        if self.is_default() {
            super::hp::validate_gl_domain(order, precision_bits)
        } else {
            domain(order, precision_bits).map(|_| ())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Certificate {
    pub semantics: String,
}
type CertifiedTable = ((Vec<Float>, Vec<Float>), Certificate);

fn error(message: impl Into<String>) -> CacheError {
    CacheError::InvalidManifest(message.into())
}

pub(super) fn domain(n: usize, p: u32) -> Result<u64, CacheError> {
    GaussLegendreGenerator::ArbCertified.ensure_available()?;
    if n == 0 || n > 16384 || !(64..=16384).contains(&p) {
        return Err(error(
            "certified Arb quadrature requires order 1..=16384 and precision 64..=16384",
        ));
    }
    // Byte-sized, limb-rounded scalars, not bits interpreted as bytes. The
    // bridge owns two ball vectors, two output Float vectors, pointer arrays,
    // and serial polynomial/rounding scratch. Sixteen scalar equivalents per
    // node plus fixed scratch intentionally leave native-allocation headroom.
    let scalar = (u64::from(p) + 576).div_ceil(64) * 8 + 128;
    let bytes = (n as u64)
        .checked_mul(scalar)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| v.checked_add(64 << 20))
        .ok_or_else(|| error("certified Arb quadrature workspace overflow"))?;
    Ok(bytes)
}

#[cfg(all(test, feature = "arb"))]
mod admission_tests {
    use super::*;

    #[test]
    fn arb_large_rule_admission_counts_bytes_and_combined_ownership() {
        let bytes = domain(8552, 3386).unwrap();
        assert!(bytes < (192 << 20));
        let resources = xc_core::RunResources::new(32 << 30).unwrap();
        let reservations = (0..16)
            .map(|_| resources.try_reserve_working(bytes).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(resources.usage().working_bytes, 16 * bytes);
        drop(reservations);
        assert_eq!(resources.usage().working_bytes, 0);
        assert!(xc_core::RunResources::new(bytes - 1)
            .unwrap()
            .try_reserve_working(bytes)
            .is_err());
    }
}

pub(super) fn evaluate(
    n: usize,
    p: u32,
    proof: Option<&Certificate>,
) -> Result<CertifiedTable, CacheError> {
    let bytes = domain(n, p)?;
    let _reservation = xc_core::RunResources::current()
        .map(|r| r.try_reserve_working(bytes))
        .transpose()
        .map_err(|e| error(e.to_string()))?;
    #[cfg(feature = "arb")]
    {
        native::evaluate(n, p, proof)
    }
    #[cfg(not(feature = "arb"))]
    {
        let _ = proof;
        Err(error("certified Arb quadrature requires feature arb"))
    }
}

#[cfg(feature = "arb")]
mod native {
    use super::*;
    use std::ffi::{c_int, c_void};
    unsafe extern "C" {
        fn xc_gl_native_compatible() -> c_int;
        fn xc_gl_mpfr_size() -> usize;
        fn xc_gl_slong_size() -> usize;
        fn xc_gl_emin() -> i64;
        fn xc_gl_emax() -> i64;
        fn xc_gl_threads() -> c_int;
        fn xc_gl_certified(n: usize, p: u32, x: *const *mut c_void, w: *const *mut c_void)
            -> c_int;
    }
    pub(super) fn evaluate(
        n: usize,
        p: u32,
        proof: Option<&Certificate>,
    ) -> Result<CertifiedTable, CacheError> {
        if unsafe { xc_gl_native_compatible() } != 1 {
            return Err(error(
                "certified quadrature requires thread-safe MPFR and matching FLINT, GMP, and MPFR headers and runtime libraries",
            ));
        }
        let mut nodes = vec![Float::with_val(p, 0); n];
        let mut weights = vec![Float::with_val(p, 0); n];
        // Match Rust's MPFR context without mutating either library's TLS.
        // A broad finite range is required; restricted ranges refuse explicitly.
        let margin = i64::from(4 * (p + 512) + 4096);
        unsafe {
            if xc_gl_slong_size() != std::mem::size_of::<isize>()
                || xc_gl_mpfr_size() != std::mem::size_of_val(&*nodes[0].as_raw())
                || xc_gl_emin() != i64::from(rug::float::exp_min())
                || xc_gl_emax() != i64::from(rug::float::exp_max())
                || xc_gl_emin() > -margin
                || xc_gl_emax() < margin
                || xc_gl_threads() != 1
            {
                return Err(error(
                    "certified quadrature requires compatible MPFR ABI/range and one FLINT thread",
                ));
            }
        }
        if proof.is_some_and(|proof| {
            proof.semantics != GaussLegendreGenerator::ArbCertified.semantic_id()
        }) {
            return Err(error("certified quadrature proof semantics mismatch"));
        }
        let x: Vec<_> = nodes
            .iter_mut()
            .map(|v| v.as_raw_mut().cast::<c_void>())
            .collect();
        let w: Vec<_> = weights
            .iter_mut()
            .map(|v| v.as_raw_mut().cast::<c_void>())
            .collect();
        // Reconstruct certified indexed root/weight enclosures on EVERY replay.
        // Stable metadata alone is not evidence; untrusted ball strings are never parsed.
        let status = unsafe { xc_gl_certified(n, p, x.as_ptr(), w.as_ptr()) };
        if status != 0 {
            return Err(error(format!("certified quadrature n={n}, precision={p} refused: native status {status} (1 domain/context, 2 guard exhausted, 4 validation)")));
        }
        let certificate = Certificate {
            semantics: GaussLegendreGenerator::ArbCertified
                .semantic_id()
                .to_owned(),
        };
        Ok(((nodes, weights), certificate))
    }

    #[test]
    fn native_bridge_and_rug_share_mpfr_thread_context() {
        // A separate MPFR copy can report the same default range. Changing
        // this test thread's range proves the bridge observes Rug's context.
        struct Restore(std::os::raw::c_long);
        impl Drop for Restore {
            fn drop(&mut self) {
                assert_eq!(unsafe { gmp_mpfr_sys::mpfr::set_emin(self.0) }, 0);
            }
        }
        let original = unsafe { gmp_mpfr_sys::mpfr::get_emin() };
        let _restore = Restore(original);
        let probe: i32 = -100_003;
        assert_eq!(unsafe { gmp_mpfr_sys::mpfr::set_emin(probe.into()) }, 0);
        assert_eq!(unsafe { xc_gl_native_compatible() }, 1);
        assert_eq!(unsafe { xc_gl_emin() }, i64::from(probe));
        assert_eq!(rug::float::exp_min(), probe);
        assert_eq!(unsafe { xc_gl_emax() }, i64::from(rug::float::exp_max()));
    }
}
