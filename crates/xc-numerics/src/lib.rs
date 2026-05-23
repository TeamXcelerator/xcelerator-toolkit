// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! High-precision numerical primitives.
//!
//! Domain-agnostic numerical building blocks used across the toolkit:
//!
//! - **`quadrature`**: Gauss-Legendre nodes/weights at f64 and HP, with
//!   a disk cache for HP nodes (parameter-independent across runs).
//! - **`linalg`**: HP linear algebra — LU factorization with partial
//!   pivoting, LU solve, inverse iteration for smallest eigenpair,
//!   ℓ² normalization, Rayleigh quotient.
//! - **`root_finding`**: bisection and Newton refinement helpers.
//!
//! The HP tier is gated behind the `hp` feature.

pub mod quadrature;
pub mod root_finding;
pub mod primes;

#[cfg(feature = "hp")]
pub mod linalg;
