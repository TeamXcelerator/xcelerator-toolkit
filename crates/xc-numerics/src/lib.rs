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

pub mod quadrature;
pub mod root_finding;
pub mod primes;

/// Debug logging macro — only prints when `XCELERATOR_DEBUG=1` is set.
/// Same contract as `xc_spectral::hp_debug!` — use for diagnostics only.
#[macro_export]
macro_rules! hp_debug {
    ($($arg:tt)*) => {
        if std::env::var("XCELERATOR_DEBUG").as_deref() == Ok("1") {
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
pub mod hp_runtime;
