// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Stable, domain-independent contracts for Xcelerator Toolkit v0.13.0.
//!
//! This crate deliberately contains no numerical algorithms.  It defines the
//! typed language shared by operators, solvers, caches, certificates, and
//! domain modules so that a mathematical problem is not confused with the
//! algorithm used to solve it.

mod analytic_context;
mod archive;
mod artifact_plan;
mod assurance;
mod capability;
mod config;
mod config_resolution;
mod diagnostics;
mod error;
mod optimization;
mod performance;
mod precision;
mod progress;
mod provenance;
mod publication;
mod research;
mod resource;
mod secret;
mod status;
mod subspace;
mod target;

pub use analytic_context::*;
pub use archive::*;
pub use artifact_plan::*;
pub use assurance::*;
pub use capability::*;
pub use config::*;
pub use config_resolution::*;
pub use diagnostics::*;
pub use error::*;
pub use optimization::*;
pub use performance::*;
pub use precision::*;
pub use progress::*;
pub use provenance::*;
pub use publication::*;
pub use research::*;
pub use resource::*;
pub use secret::*;
pub use status::*;
pub use subspace::*;
pub use target::*;
