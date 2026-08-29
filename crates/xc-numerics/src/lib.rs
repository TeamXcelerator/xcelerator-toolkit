// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision numerical primitives.
//!
//! Domain-agnostic numerical building blocks used across the toolkit:
//!
//! - **`quadrature`**: Gauss-Legendre nodes/weights at f64 and HP, with
//!   a disk cache for HP nodes (parameter-independent across runs).
//! - **`linalg`**: HP linear algebra — dense LU factorization with
//!   partial pivoting (`lu_factor` / `lu_solve`), banded tridiagonal
//!   LU via Thomas with partial pivoting (`tridiag_lu_factor_hp` /
//!   `tridiag_lu_solve_hp`; O(n) factor and solve), inverse iteration
//!   for smallest eigenpair (with optional forced-even projection
//!   and dual convergence-floor docs), ℓ² normalization, Rayleigh
//!   quotient.
//! - **`root_finding`**: bisection and Newton refinement helpers.
//! - **`fmt`**: HP-only display/comparison helpers (no f64 fallbacks).
//!   Use these wherever you would otherwise call `to_f64()` on a
//!   `rug::Float` for printing or comparison.
//! - **`eigen`**: HP symmetric eigendecomposition. Householder
//!   tridiagonalization, symmetric tridiagonal QR with Wilkinson
//!   shifts, and shifted inverse iteration for one eigenvector at
//!   a known eigenvalue. Truly dynamic in working precision.
//!
//! The HP tier is gated behind the `hp` feature.

pub mod grid_integral;
pub mod primes;
pub mod quadrature;
pub mod root_finding;

/// Debug logging macro controlled by [`set_hp_debug_logging`].
static HP_DEBUG_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Explicitly enables or disables diagnostic HP progress logging.
pub fn set_hp_debug_logging(enabled: bool) {
    HP_DEBUG_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[doc(hidden)]
pub fn hp_debug_enabled() -> bool {
    HP_DEBUG_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[macro_export]
macro_rules! hp_debug {
    ($($arg:tt)*) => {
        if $crate::hp_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

#[cfg(feature = "hp")]
pub mod linalg;

#[cfg(feature = "hp")]
pub mod fmt;

#[cfg(feature = "hp")]
pub mod eigen;

#[cfg(feature = "hp")]
pub mod interval;

#[cfg(feature = "hp")]
pub mod mpfr_interval;

#[cfg(feature = "hp")]
pub mod reduction;

// Not gated behind `hp`: resolved scheduling policy and its tests remain
// available to f64-only planners, while no process environment or platform
// detection can silently alter execution.
pub mod hp_runtime;
