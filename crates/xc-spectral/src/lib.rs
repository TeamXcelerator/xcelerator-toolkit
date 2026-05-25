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
//!   ξ_λ for Lemma 7.2-style tests. HP eigenvalue spectrum is cached
//!   to disk for re-runs at the same `(λ², n_grid, prec)`.
//! - **`mellin`**: Truncated completed eta function `Λ_λ(s)` and
//!   `ξ`-weighted variants on the critical line, with parallelized
//!   zero scanners.
//! - **`yakaboylu`**: Yakaboylu's Hilbert–Pólya framework. The
//!   `V_R(s, s')` matrix element, W-positivity tests on the critical
//!   line, indefiniteness on synthetic off-line zeros. f64 + HP
//!   tiers.
//! - **`lfunction`**: Dirichlet L-function character specs (`χ₃, χ₄,
//!   χ₅, χ₇`) and twisted prime-power enumeration. Used to extend
//!   the CCM construction to non-trivial L-functions.
//!
//! The HP tier is gated behind the `hp` feature.

pub mod ccm;
pub mod prolate;
pub mod mellin;
pub mod yakaboylu;
pub mod lfunction;
