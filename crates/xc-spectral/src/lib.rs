// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Spectral methods for analytic number theory.
//!
//! Building blocks for spectral approaches to the Riemann Hypothesis
//! and related problems:
//!
//! - **`ccm`**: Connes-Consani-Moscovici 2025 construction. Weil
//!   quadratic form on the V_n basis, smallest-eigenvector computation,
//!   rational-function root extraction. f64 + HP tiers.
//! - **`prolate`**: Prolate-wave operator PW_λ on `[-λ, λ]`,
//!   Sturm-Liouville eigenfunctions, the ℰ map, comparison against
//!   ξ_λ for Lemma 7.2-style tests.
//! - **`mellin`**: Truncated completed eta function Λ_λ(s) and
//!   ξ-weighted variants on the critical line.
//! - **`hermite`** (planned): Hermite expansions of test functions on
//!   ℝ, Polya-Schur and Laguerre-Polya class membership tests.
//!
//! The HP tier is gated behind the `hp` feature.

pub mod ccm;
pub mod prolate;
pub mod mellin;
pub mod yakaboylu;
pub mod lfunction;
