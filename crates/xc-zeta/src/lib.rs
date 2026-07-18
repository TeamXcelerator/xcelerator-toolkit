// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Riemann zeta function utilities.
//!
//! - **`zeros`**: Loaders for the canonical reference zero file.
//!   Provides HP-string, f64-truncated, and `rug::Float` views.
//!
//! The bundled reference data was computed with rigorous Arb interval
//! arithmetic. Its leading 1,000 digits were independently cross-checked
//! against a standard published tabulation.

pub mod zeros;
