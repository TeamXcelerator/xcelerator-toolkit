// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Riemann zeta function utilities.
//!
//! - **`zeros`**: Loaders for the canonical reference zero file.
//!   Provides HP-string, nearest-rounded f64, and `rug::Float` views.
//!
//! The bundled table is finite decimal reference data. Its historical provenance
//! attributes its computation to Arb and a leading-digit tabulation comparison;
//! the loader does not replay those computations or provide root certificates.
//! Extra binary storage precision cannot establish digits beyond the source table.

pub mod zeros;
