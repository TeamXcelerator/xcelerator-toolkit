// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Riemann zeta function utilities.
//!
//! - **`zeros`**: Loaders for the canonical reference zero file with
//!   integrity verification (SHA-256). Provides both HP-string and
//!   f64-truncated views.
//!
//! The reference data file is generated reproducibly via PARI/GP
//! (`scripts/generate_zeros.sh` in the repository root) and is
//! cross-validated against universally-tabulated first-10 zeros.

pub mod zeros;
