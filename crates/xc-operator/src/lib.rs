// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Reusable operator contracts and small trusted operator implementations.
//!
//! The core solver layer sees mathematical actions, dimensions, and bounds;
//! it does not require every problem to materialize a dense matrix.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub mod batch;
pub mod checkpoint;
pub mod vector_storage;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixStructure {
    Dense,
    PackedSymmetric,
    Diagonal,
    Tridiagonal,
    Banded { lower: usize, upper: usize },
    MatrixFree,
    Composite,
    RankOneUpdate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperatorMetadata {
    pub name: String,
    pub dimension: usize,
    pub structure: MatrixStructure,
    pub scalar_backend: String,
    pub symmetric: bool,
    pub exact_action: bool,
    pub tags: Vec<String>,
}

impl OperatorMetadata {
    pub fn new(
        name: impl Into<String>,
        dimension: usize,
        structure: MatrixStructure,
        scalar_backend: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            dimension,
            structure,
            scalar_backend: scalar_backend.into(),
            symmetric: false,
            exact_action: true,
            tags: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperatorError {
    DimensionMismatch { expected: usize, actual: usize },
    InvalidData(String),
    ApplicationFailed(String),
}

impl Display for OperatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidData(message) => write!(f, "invalid operator data: {message}"),
            Self::ApplicationFailed(message) => write!(f, "operator application failed: {message}"),
        }
    }
}

impl Error for OperatorError {}

/// Absolute error contract for one operator application.
#[derive(Clone, Debug, PartialEq)]
pub enum ApplicationErrorBound<S> {
    /// The action is exact up to arithmetic in its declared scalar backend.
    Exact,
    /// The returned vector differs from the mathematical action by at most
    /// this absolute 2-norm bound.
    Absolute(S),
}

/// Matrix-free contract for a finite-dimensional linear map.
///
/// # Mathematical semantics
/// `apply` computes `y = A x` for the operator described by `metadata`; the
/// dimension and basis ordering are part of that mathematical identity.
///
/// # Precision
/// Precision is determined by `S` and by the implementation metadata. An
/// implementation must not silently substitute a lower-precision scalar.
///
/// # Failure states
/// Implementations return `OperatorError` for dimension mismatch, invalid
/// structure, cancellation, or an unavailable action rather than partial data.
///
/// # Assurance and validity
/// `application_error_bound` states whether the action is exact up to declared
/// arithmetic or carries a rigorous absolute bound. The trait alone does not
/// certify a spectrum or continuum claim.
///
/// # Cache effects
/// The trait has no implicit cache behavior. Implementations that load an
/// artifact must expose that dependency in their surrounding plan/provenance.
///
/// # Example
/// Compiled example: `crates/xc-operator/examples/matrix_action.rs`.
pub trait LinearOperator<S>: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply(&self, x: &[S], y: &mut [S]) -> Result<(), OperatorError>;
    fn metadata(&self) -> OperatorMetadata;

    /// A mathematically valid upper bound on the induced 2-norm when known.
    /// Returning `None` is preferable to returning an unsafe estimate.
    fn norm_bound(&self) -> Option<S>
    where
        S: Clone,
    {
        None
    }

    /// A bound on numerical or analytic approximation introduced by one
    /// application, distinct from ordinary scalar roundoff.
    fn application_error_bound(&self) -> ApplicationErrorBound<S>
    where
        S: Clone,
    {
        ApplicationErrorBound::Exact
    }
}

pub trait SymmetricOperator<S>: LinearOperator<S> {}
pub trait PositiveDefiniteMetric<S>: SymmetricOperator<S> {}

/// Domain-independent description of a user-supplied finite basis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BasisMetadata {
    pub name: String,
    pub ambient_dimension: usize,
    pub basis_dimension: usize,
    pub scalar_backend: String,
    pub orthonormal_claimed: bool,
    pub tags: Vec<String>,
}

/// Public basis contract. A basis may be nonorthogonal; metric and
/// orthogonalization policy are supplied independently.
pub trait Basis<S>: Send + Sync {
    fn ambient_dimension(&self) -> usize;
    fn basis_dimension(&self) -> usize;
    fn vector(&self, index: usize, output: &mut [S]) -> Result<(), OperatorError>;
    fn metadata(&self) -> BasisMetadata;
}

/// A linear idempotent action supplied independently of any domain module.
/// Implementations are responsible for validating their projector semantics.
pub trait Projector<S>: LinearOperator<S> {}

/// A convergence or approximation action supplied independently of any
/// solver or domain module. Approximate actions use
/// [`LinearOperator::application_error_bound`].
pub trait Preconditioner<S>: LinearOperator<S> {}

pub fn basis_vector_alloc<S>(basis: &dyn Basis<S>, index: usize) -> Result<Vec<S>, OperatorError>
where
    S: Default + Clone,
{
    if index >= basis.basis_dimension() {
        return Err(OperatorError::InvalidData(format!(
            "basis index {index} is outside dimension {}",
            basis.basis_dimension()
        )));
    }
    let mut output = vec![S::default(); basis.ambient_dimension()];
    basis.vector(index, &mut output)?;
    Ok(output)
}

pub fn apply_alloc<S>(operator: &dyn LinearOperator<S>, x: &[S]) -> Result<Vec<S>, OperatorError>
where
    S: Default + Clone,
{
    let mut y = vec![S::default(); operator.dimension()];
    operator.apply(x, &mut y)?;
    Ok(y)
}

fn check_dimensions(n: usize, x: &[f64], y: &[f64]) -> Result<(), OperatorError> {
    if x.len() != n {
        return Err(OperatorError::DimensionMismatch {
            expected: n,
            actual: x.len(),
        });
    }
    if y.len() != n {
        return Err(OperatorError::DimensionMismatch {
            expected: n,
            actual: y.len(),
        });
    }
    Ok(())
}

/// Trusted row-major dense symmetric f64 reference operator.
#[derive(Clone, Debug)]
pub struct DenseSymmetricF64 {
    n: usize,
    data: Vec<f64>,
    norm_bound: f64,
    name: String,
}

impl DenseSymmetricF64 {
    pub fn new(
        name: impl Into<String>,
        n: usize,
        data: Vec<f64>,
        symmetry_tolerance: f64,
    ) -> Result<Self, OperatorError> {
        if n == 0 {
            return Err(OperatorError::InvalidData(
                "dimension must be positive".to_owned(),
            ));
        }
        if data.len() != n * n {
            return Err(OperatorError::DimensionMismatch {
                expected: n * n,
                actual: data.len(),
            });
        }
        if !symmetry_tolerance.is_finite() || symmetry_tolerance < 0.0 {
            return Err(OperatorError::InvalidData(
                "symmetry tolerance must be finite and nonnegative".to_owned(),
            ));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(OperatorError::InvalidData(
                "matrix entries must be finite".to_owned(),
            ));
        }
        for i in 0..n {
            for j in 0..i {
                let a = data[i * n + j];
                let b = data[j * n + i];
                if (a - b).abs() > symmetry_tolerance {
                    return Err(OperatorError::InvalidData(format!(
                        "matrix is not symmetric at ({i}, {j}): {a} vs {b}"
                    )));
                }
            }
        }
        let norm_bound = (0..n)
            .map(|i| (0..n).map(|j| data[i * n + j].abs()).sum::<f64>())
            .fold(0.0, f64::max);
        Ok(Self {
            n,
            data,
            norm_bound,
            name: name.into(),
        })
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.n && col < self.n {
            Some(self.data[row * self.n + col])
        } else {
            None
        }
    }
}

impl LinearOperator<f64> for DenseSymmetricF64 {
    fn dimension(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        check_dimensions(self.n, x, y)?;
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = (0..self.n).map(|j| self.data[i * self.n + j] * x[j]).sum();
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata =
            OperatorMetadata::new(self.name.clone(), self.n, MatrixStructure::Dense, "f64");
        metadata.symmetric = true;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.norm_bound)
    }
}

impl SymmetricOperator<f64> for DenseSymmetricF64 {}

/// Symmetric matrix stored as its packed lower triangle in row-major order.
#[derive(Clone, Debug)]
pub struct PackedSymmetricF64 {
    n: usize,
    lower: Vec<f64>,
    norm_bound: f64,
    name: String,
}

impl PackedSymmetricF64 {
    pub fn new(name: impl Into<String>, n: usize, lower: Vec<f64>) -> Result<Self, OperatorError> {
        let expected = n
            .checked_add(1)
            .and_then(|next| n.checked_mul(next))
            .and_then(|value| value.checked_div(2))
            .ok_or_else(|| OperatorError::InvalidData("packed dimension overflow".to_owned()))?;
        if n == 0 || lower.len() != expected {
            return Err(OperatorError::DimensionMismatch {
                expected,
                actual: lower.len(),
            });
        }
        if lower.iter().any(|value| !value.is_finite()) {
            return Err(OperatorError::InvalidData(
                "packed entries must be finite".to_owned(),
            ));
        }
        let get = |row: usize, column: usize| {
            let (high, low) = if row >= column {
                (row, column)
            } else {
                (column, row)
            };
            lower[high * (high + 1) / 2 + low]
        };
        let norm_bound = (0..n)
            .map(|row| (0..n).map(|column| get(row, column).abs()).sum())
            .fold(0.0, f64::max);
        Ok(Self {
            n,
            lower,
            norm_bound,
            name: name.into(),
        })
    }

    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        if row >= self.n || column >= self.n {
            return None;
        }
        let (high, low) = if row >= column {
            (row, column)
        } else {
            (column, row)
        };
        Some(self.lower[high * (high + 1) / 2 + low])
    }
}

impl LinearOperator<f64> for PackedSymmetricF64 {
    fn dimension(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        check_dimensions(self.n, x, y)?;
        y.fill(0.0);
        for row in 0..self.n {
            for column in 0..=row {
                let value = self.lower[row * (row + 1) / 2 + column];
                y[row] += value * x[column];
                if row != column {
                    y[column] += value * x[row];
                }
            }
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.n,
            MatrixStructure::PackedSymmetric,
            "f64",
        );
        metadata.symmetric = true;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.norm_bound)
    }
}

impl SymmetricOperator<f64> for PackedSymmetricF64 {}

/// Symmetric band matrix stored by upper diagonals. `bands[0]` is the main
/// diagonal and `bands[d]` contains entries `(i, i + d)`.
#[derive(Clone, Debug)]
pub struct SymmetricBandedF64 {
    bands: Vec<Vec<f64>>,
    norm_bound: f64,
    name: String,
}

impl SymmetricBandedF64 {
    pub fn new(name: impl Into<String>, bands: Vec<Vec<f64>>) -> Result<Self, OperatorError> {
        let n = bands.first().map_or(0, Vec::len);
        if n == 0 || bands.len() > n {
            return Err(OperatorError::InvalidData(
                "banded operator requires a nonempty main diagonal".to_owned(),
            ));
        }
        for (distance, band) in bands.iter().enumerate() {
            let expected = n - distance;
            if band.len() != expected {
                return Err(OperatorError::DimensionMismatch {
                    expected,
                    actual: band.len(),
                });
            }
            if band.iter().any(|value| !value.is_finite()) {
                return Err(OperatorError::InvalidData(
                    "band entries must be finite".to_owned(),
                ));
            }
        }
        let mut row_sums = vec![0.0; n];
        for (distance, band) in bands.iter().enumerate() {
            for (row, value) in band.iter().enumerate() {
                row_sums[row] += value.abs();
                if distance != 0 {
                    row_sums[row + distance] += value.abs();
                }
            }
        }
        let norm_bound = row_sums.into_iter().fold(0.0, f64::max);
        Ok(Self {
            bands,
            norm_bound,
            name: name.into(),
        })
    }

    pub fn bandwidth(&self) -> usize {
        self.bands.len() - 1
    }
}

impl LinearOperator<f64> for SymmetricBandedF64 {
    fn dimension(&self) -> usize {
        self.bands[0].len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        check_dimensions(self.dimension(), x, y)?;
        y.fill(0.0);
        for (distance, band) in self.bands.iter().enumerate() {
            for (row, value) in band.iter().enumerate() {
                y[row] += value * x[row + distance];
                if distance != 0 {
                    y[row + distance] += value * x[row];
                }
            }
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.dimension(),
            MatrixStructure::Banded {
                lower: self.bandwidth(),
                upper: self.bandwidth(),
            },
            "f64",
        );
        metadata.symmetric = true;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.norm_bound)
    }
}

impl SymmetricOperator<f64> for SymmetricBandedF64 {}

type MatrixFreeActionF64 = dyn Fn(&[f64], &mut [f64]) -> Result<(), OperatorError> + Send + Sync;

/// Domain-independent symmetric matrix-free action with explicit norm and
/// per-application approximation contracts.
#[derive(Clone)]
pub struct MatrixFreeSymmetricF64 {
    dimension: usize,
    action: Arc<MatrixFreeActionF64>,
    norm_bound: Option<f64>,
    error_bound: ApplicationErrorBound<f64>,
    name: String,
}

impl MatrixFreeSymmetricF64 {
    pub fn exact(
        name: impl Into<String>,
        dimension: usize,
        norm_bound: Option<f64>,
        action: impl Fn(&[f64], &mut [f64]) -> Result<(), OperatorError> + Send + Sync + 'static,
    ) -> Result<Self, OperatorError> {
        Self::new(
            name,
            dimension,
            norm_bound,
            ApplicationErrorBound::Exact,
            action,
        )
    }

    pub fn approximate(
        name: impl Into<String>,
        dimension: usize,
        norm_bound: Option<f64>,
        absolute_error_bound: f64,
        action: impl Fn(&[f64], &mut [f64]) -> Result<(), OperatorError> + Send + Sync + 'static,
    ) -> Result<Self, OperatorError> {
        Self::new(
            name,
            dimension,
            norm_bound,
            ApplicationErrorBound::Absolute(absolute_error_bound),
            action,
        )
    }

    fn new(
        name: impl Into<String>,
        dimension: usize,
        norm_bound: Option<f64>,
        error_bound: ApplicationErrorBound<f64>,
        action: impl Fn(&[f64], &mut [f64]) -> Result<(), OperatorError> + Send + Sync + 'static,
    ) -> Result<Self, OperatorError> {
        if dimension == 0 {
            return Err(OperatorError::InvalidData(
                "matrix-free dimension must be positive".to_owned(),
            ));
        }
        if norm_bound.is_some_and(|bound| !bound.is_finite() || bound < 0.0) {
            return Err(OperatorError::InvalidData(
                "matrix-free norm bound must be finite and nonnegative".to_owned(),
            ));
        }
        if matches!(error_bound, ApplicationErrorBound::Absolute(bound) if !bound.is_finite() || bound < 0.0)
        {
            return Err(OperatorError::InvalidData(
                "application error bound must be finite and nonnegative".to_owned(),
            ));
        }
        Ok(Self {
            dimension,
            action: Arc::new(action),
            norm_bound,
            error_bound,
            name: name.into(),
        })
    }
}

impl LinearOperator<f64> for MatrixFreeSymmetricF64 {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        check_dimensions(self.dimension, x, y)?;
        (self.action)(x, y)
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.dimension,
            MatrixStructure::MatrixFree,
            "f64",
        );
        metadata.symmetric = true;
        metadata.exact_action = self.error_bound == ApplicationErrorBound::Exact;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        self.norm_bound
    }

    fn application_error_bound(&self) -> ApplicationErrorBound<f64> {
        self.error_bound.clone()
    }
}

impl SymmetricOperator<f64> for MatrixFreeSymmetricF64 {}

#[derive(Clone, Debug)]
pub struct DiagonalF64 {
    diagonal: Vec<f64>,
    name: String,
}

impl DiagonalF64 {
    pub fn new(name: impl Into<String>, diagonal: Vec<f64>) -> Result<Self, OperatorError> {
        if diagonal.is_empty() || diagonal.iter().any(|x| !x.is_finite()) {
            return Err(OperatorError::InvalidData(
                "diagonal must be nonempty and finite".to_owned(),
            ));
        }
        Ok(Self {
            diagonal,
            name: name.into(),
        })
    }

    pub fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }
}

impl LinearOperator<f64> for DiagonalF64 {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        check_dimensions(self.dimension(), x, y)?;
        for ((yi, di), xi) in y.iter_mut().zip(&self.diagonal).zip(x) {
            *yi = di * xi;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.dimension(),
            MatrixStructure::Diagonal,
            "f64",
        );
        metadata.symmetric = true;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.diagonal.iter().map(|x| x.abs()).fold(0.0, f64::max))
    }
}

impl SymmetricOperator<f64> for DiagonalF64 {}

#[derive(Clone, Debug)]
pub struct TridiagonalF64 {
    diagonal: Vec<f64>,
    off_diagonal: Vec<f64>,
    name: String,
}

impl TridiagonalF64 {
    pub fn new(
        name: impl Into<String>,
        diagonal: Vec<f64>,
        off_diagonal: Vec<f64>,
    ) -> Result<Self, OperatorError> {
        if diagonal.is_empty() || off_diagonal.len() + 1 != diagonal.len() {
            return Err(OperatorError::InvalidData(
                "tridiagonal requires off_diagonal.len() + 1 == diagonal.len()".to_owned(),
            ));
        }
        if diagonal
            .iter()
            .chain(&off_diagonal)
            .any(|value| !value.is_finite())
        {
            return Err(OperatorError::InvalidData(
                "tridiagonal entries must be finite".to_owned(),
            ));
        }
        Ok(Self {
            diagonal,
            off_diagonal,
            name: name.into(),
        })
    }

    pub fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    pub fn off_diagonal(&self) -> &[f64] {
        &self.off_diagonal
    }
}

impl LinearOperator<f64> for TridiagonalF64 {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        let n = self.dimension();
        check_dimensions(n, x, y)?;
        for i in 0..n {
            let mut value = self.diagonal[i] * x[i];
            if i > 0 {
                value += self.off_diagonal[i - 1] * x[i - 1];
            }
            if i + 1 < n {
                value += self.off_diagonal[i] * x[i + 1];
            }
            y[i] = value;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.dimension(),
            MatrixStructure::Tridiagonal,
            "f64",
        );
        metadata.symmetric = true;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        let n = self.dimension();
        let mut bound: f64 = 0.0;
        for i in 0..n {
            let mut row = self.diagonal[i].abs();
            if i > 0 {
                row += self.off_diagonal[i - 1].abs();
            }
            if i + 1 < n {
                row += self.off_diagonal[i].abs();
            }
            bound = bound.max(row);
        }
        Some(bound)
    }
}

impl SymmetricOperator<f64> for TridiagonalF64 {}

pub struct ShiftedF64<'a> {
    base: &'a dyn SymmetricOperator<f64>,
    shift: f64,
}

impl<'a> ShiftedF64<'a> {
    pub fn new(base: &'a dyn SymmetricOperator<f64>, shift: f64) -> Result<Self, OperatorError> {
        if !shift.is_finite() {
            return Err(OperatorError::InvalidData(
                "shift must be finite".to_owned(),
            ));
        }
        Ok(Self { base, shift })
    }
}

impl LinearOperator<f64> for ShiftedF64<'_> {
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.base.apply(x, y)?;
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi -= self.shift * xi;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = self.base.metadata();
        metadata.name = format!("{} - ({}) I", metadata.name, self.shift);
        metadata.structure = MatrixStructure::Composite;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        self.base.norm_bound().map(|b| b + self.shift.abs())
    }

    fn application_error_bound(&self) -> ApplicationErrorBound<f64> {
        self.base.application_error_bound()
    }
}

impl SymmetricOperator<f64> for ShiftedF64<'_> {}

pub struct NegatedF64<'a> {
    base: &'a dyn SymmetricOperator<f64>,
}

impl<'a> NegatedF64<'a> {
    pub fn new(base: &'a dyn SymmetricOperator<f64>) -> Self {
        Self { base }
    }
}

impl LinearOperator<f64> for NegatedF64<'_> {
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.base.apply(x, y)?;
        for yi in y {
            *yi = -*yi;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = self.base.metadata();
        metadata.name = format!("-({})", metadata.name);
        metadata.structure = MatrixStructure::Composite;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        self.base.norm_bound()
    }

    fn application_error_bound(&self) -> ApplicationErrorBound<f64> {
        self.base.application_error_bound()
    }
}

impl SymmetricOperator<f64> for NegatedF64<'_> {}

pub struct RankOneUpdateF64<'a> {
    base: &'a dyn SymmetricOperator<f64>,
    alpha: f64,
    vector: Vec<f64>,
}

impl<'a> RankOneUpdateF64<'a> {
    pub fn new(
        base: &'a dyn SymmetricOperator<f64>,
        alpha: f64,
        vector: Vec<f64>,
    ) -> Result<Self, OperatorError> {
        if vector.len() != base.dimension() {
            return Err(OperatorError::DimensionMismatch {
                expected: base.dimension(),
                actual: vector.len(),
            });
        }
        if !alpha.is_finite() || vector.iter().any(|value| !value.is_finite()) {
            return Err(OperatorError::InvalidData(
                "rank-one update values must be finite".to_owned(),
            ));
        }
        Ok(Self {
            base,
            alpha,
            vector,
        })
    }
}

impl LinearOperator<f64> for RankOneUpdateF64<'_> {
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.base.apply(x, y)?;
        let dot: f64 = self.vector.iter().zip(x).map(|(a, b)| a * b).sum();
        for (yi, vi) in y.iter_mut().zip(&self.vector) {
            *yi += self.alpha * vi * dot;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = self.base.metadata();
        metadata.name = format!("{} + rank-one update", metadata.name);
        metadata.structure = MatrixStructure::RankOneUpdate;
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        let vector_norm_sq: f64 = self.vector.iter().map(|v| v * v).sum();
        self.base
            .norm_bound()
            .map(|bound| bound + self.alpha.abs() * vector_norm_sq)
    }

    fn application_error_bound(&self) -> ApplicationErrorBound<f64> {
        self.base.application_error_bound()
    }
}

impl SymmetricOperator<f64> for RankOneUpdateF64<'_> {}

pub struct GeneralizedEigenProblem<'a, S> {
    pub operator: &'a dyn SymmetricOperator<S>,
    pub metric: &'a dyn PositiveDefiniteMetric<S>,
}

impl<'a, S> GeneralizedEigenProblem<'a, S> {
    pub fn new(
        operator: &'a dyn SymmetricOperator<S>,
        metric: &'a dyn PositiveDefiniteMetric<S>,
    ) -> Result<Self, OperatorError> {
        if operator.dimension() != metric.dimension() {
            return Err(OperatorError::DimensionMismatch {
                expected: operator.dimension(),
                actual: metric.dimension(),
            });
        }
        Ok(Self { operator, metric })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UserNonorthogonalBasis;

    impl Basis<f64> for UserNonorthogonalBasis {
        fn ambient_dimension(&self) -> usize {
            2
        }

        fn basis_dimension(&self) -> usize {
            2
        }

        fn vector(&self, index: usize, output: &mut [f64]) -> Result<(), OperatorError> {
            check_dimensions(2, output, output)?;
            match index {
                0 => output.copy_from_slice(&[1.0, 0.0]),
                1 => output.copy_from_slice(&[1.0, 1.0]),
                _ => {
                    return Err(OperatorError::InvalidData(
                        "basis index is out of range".to_owned(),
                    ));
                }
            }
            Ok(())
        }

        fn metadata(&self) -> BasisMetadata {
            BasisMetadata {
                name: "user-nonorthogonal".to_owned(),
                ambient_dimension: 2,
                basis_dimension: 2,
                scalar_backend: "f64".to_owned(),
                orthonormal_claimed: false,
                tags: vec!["test-project".to_owned()],
            }
        }
    }

    struct UserMetric {
        dense: DenseSymmetricF64,
    }

    impl LinearOperator<f64> for UserMetric {
        fn dimension(&self) -> usize {
            self.dense.dimension()
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
            self.dense.apply(x, y)
        }

        fn metadata(&self) -> OperatorMetadata {
            self.dense.metadata()
        }

        fn norm_bound(&self) -> Option<f64> {
            self.dense.norm_bound()
        }
    }

    impl SymmetricOperator<f64> for UserMetric {}
    impl PositiveDefiniteMetric<f64> for UserMetric {}

    struct UserIdentityProjector;

    impl LinearOperator<f64> for UserIdentityProjector {
        fn dimension(&self) -> usize {
            2
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
            check_dimensions(2, x, y)?;
            y.copy_from_slice(x);
            Ok(())
        }

        fn metadata(&self) -> OperatorMetadata {
            OperatorMetadata::new("user-projector", 2, MatrixStructure::MatrixFree, "f64")
        }
    }

    impl Projector<f64> for UserIdentityProjector {}

    struct UserDiagonalPreconditioner;

    impl LinearOperator<f64> for UserDiagonalPreconditioner {
        fn dimension(&self) -> usize {
            2
        }

        fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
            check_dimensions(2, x, y)?;
            y[0] = x[0];
            y[1] = 0.5 * x[1];
            Ok(())
        }

        fn metadata(&self) -> OperatorMetadata {
            OperatorMetadata::new("user-preconditioner", 2, MatrixStructure::Diagonal, "f64")
        }
    }

    impl Preconditioner<f64> for UserDiagonalPreconditioner {}

    #[test]
    fn dense_operator_applies_and_bounds() {
        let a = DenseSymmetricF64::new("a", 2, vec![2.0, -1.0, -1.0, 3.0], 0.0).unwrap();
        let mut y = vec![0.0; 2];
        a.apply(&[4.0, 5.0], &mut y).unwrap();
        assert_eq!(y, vec![3.0, 11.0]);
        assert_eq!(a.norm_bound(), Some(4.0));
    }

    #[test]
    fn user_project_defines_nonorthogonal_basis_metric_projector_and_preconditioner() {
        let basis = UserNonorthogonalBasis;
        assert_eq!(basis_vector_alloc(&basis, 0).unwrap(), vec![1.0, 0.0]);
        assert_eq!(basis_vector_alloc(&basis, 1).unwrap(), vec![1.0, 1.0]);
        assert!(!basis.metadata().orthonormal_claimed);

        // Gram matrix of the two user basis vectors: [[1, 1], [1, 2]].
        let metric = UserMetric {
            dense: DenseSymmetricF64::new("user-gram-metric", 2, vec![1.0, 1.0, 1.0, 2.0], 0.0)
                .unwrap(),
        };
        let projector = UserIdentityProjector;
        let preconditioner = UserDiagonalPreconditioner;

        fn apply_role(
            action: &dyn LinearOperator<f64>,
            input: &[f64],
        ) -> Result<Vec<f64>, OperatorError> {
            apply_alloc(action, input)
        }
        fn accept_metric(_metric: &dyn PositiveDefiniteMetric<f64>) {}
        fn accept_projector(_projector: &dyn Projector<f64>) {}
        fn accept_preconditioner(_preconditioner: &dyn Preconditioner<f64>) {}

        accept_metric(&metric);
        accept_projector(&projector);
        accept_preconditioner(&preconditioner);
        assert_eq!(apply_role(&metric, &[2.0, 3.0]).unwrap(), vec![5.0, 8.0]);
        assert_eq!(apply_role(&projector, &[2.0, 3.0]).unwrap(), vec![2.0, 3.0]);
        assert_eq!(
            apply_role(&preconditioner, &[2.0, 3.0]).unwrap(),
            vec![2.0, 1.5]
        );
    }

    #[test]
    fn tridiagonal_matches_dense_action() {
        let t = TridiagonalF64::new("t", vec![2.0, 3.0, 4.0], vec![-1.0, 0.5]).unwrap();
        let mut y = vec![0.0; 3];
        t.apply(&[1.0, 2.0, 3.0], &mut y).unwrap();
        assert_eq!(y, vec![0.0, 6.5, 13.0]);
    }

    #[test]
    fn rejects_nonsymmetric_dense_data() {
        let err = DenseSymmetricF64::new("bad", 2, vec![1.0, 2.0, 3.0, 4.0], 0.0);
        assert!(err.is_err());
    }

    #[test]
    fn packed_banded_structured_and_matrix_free_share_one_contract() {
        let x = [1.0, 2.0, -1.0, 3.0];
        let expected = [4.0, 11.0, 1.0, 18.0];
        let dense = DenseSymmetricF64::new(
            "dense",
            4,
            vec![
                4.0, 1.0, 2.0, 0.0, 1.0, 3.0, -1.0, 1.0, 2.0, -1.0, 5.0, 2.0, 0.0, 1.0, 2.0, 6.0,
            ],
            0.0,
        )
        .unwrap();
        let packed = PackedSymmetricF64::new(
            "packed",
            4,
            vec![4.0, 1.0, 3.0, 2.0, -1.0, 5.0, 0.0, 1.0, 2.0, 6.0],
        )
        .unwrap();
        let banded = SymmetricBandedF64::new(
            "banded",
            vec![
                vec![4.0, 3.0, 5.0, 6.0],
                vec![1.0, -1.0, 2.0],
                vec![2.0, 1.0],
            ],
        )
        .unwrap();
        let diagonal = DiagonalF64::new("diagonal", vec![3.0, 3.0, 5.0, 6.0]).unwrap();
        let structured = RankOneUpdateF64::new(&diagonal, 1.0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let matrix_free =
            MatrixFreeSymmetricF64::exact("matrix-free", 4, Some(9.0), |input, output| {
                let matrix = [
                    4.0, 1.0, 2.0, 0.0, 1.0, 3.0, -1.0, 1.0, 2.0, -1.0, 5.0, 2.0, 0.0, 1.0, 2.0,
                    6.0,
                ];
                for row in 0..4 {
                    output[row] = (0..4)
                        .map(|column| matrix[row * 4 + column] * input[column])
                        .sum();
                }
                Ok(())
            })
            .unwrap();

        for operator in [
            &dense as &dyn SymmetricOperator<f64>,
            &packed,
            &banded,
            &matrix_free,
        ] {
            let mut output = [0.0; 4];
            operator.apply(&x, &mut output).unwrap();
            assert_eq!(output, expected);
        }

        let mut structured_output = [0.0; 4];
        structured.apply(&x, &mut structured_output).unwrap();
        assert_eq!(structured_output[0], 4.0);
        assert_eq!(
            structured.metadata().structure,
            MatrixStructure::RankOneUpdate
        );
    }

    #[test]
    fn approximate_matrix_free_action_reports_its_bound() {
        let operator = MatrixFreeSymmetricF64::approximate(
            "bounded-approximation",
            2,
            Some(2.0),
            0.01,
            |input, output| {
                output[0] = input[0] + 0.006;
                output[1] = input[1] + 0.008;
                Ok(())
            },
        )
        .unwrap();
        let mut output = [0.0; 2];
        operator.apply(&[1.0, -1.0], &mut output).unwrap();
        let actual_error = ((output[0] - 1.0).powi(2) + (output[1] + 1.0).powi(2)).sqrt();
        assert!(actual_error <= 0.01 + f64::EPSILON);
        assert_eq!(
            operator.application_error_bound(),
            ApplicationErrorBound::Absolute(0.01)
        );
        assert!(!operator.metadata().exact_action);
        let shifted = ShiftedF64::new(&operator, 0.5).unwrap();
        assert_eq!(
            shifted.application_error_bound(),
            ApplicationErrorBound::Absolute(0.01)
        );
    }
}

// ===========================================================================
// Arbitrary-precision reference operators
// ===========================================================================

#[cfg(feature = "hp")]
#[derive(Clone, Debug)]
pub struct DenseSymmetricHp {
    n: usize,
    data: Vec<rug::Float>,
    norm_bound: rug::Float,
    precision_bits: u32,
    name: String,
}

#[cfg(feature = "hp")]
impl DenseSymmetricHp {
    pub fn new(
        name: impl Into<String>,
        n: usize,
        data: Vec<rug::Float>,
        precision_bits: u32,
        symmetry_tolerance: &rug::Float,
    ) -> Result<Self, OperatorError> {
        use rug::Float;
        if n == 0 {
            return Err(OperatorError::InvalidData(
                "dimension must be positive".to_owned(),
            ));
        }
        if data.len() != n * n {
            return Err(OperatorError::DimensionMismatch {
                expected: n * n,
                actual: data.len(),
            });
        }
        if precision_bits < 32 || symmetry_tolerance < &Float::with_val(precision_bits, 0) {
            return Err(OperatorError::InvalidData(
                "HP precision must be at least 32 bits and symmetry tolerance nonnegative"
                    .to_owned(),
            ));
        }
        let data: Vec<Float> = data
            .into_iter()
            .map(|value| Float::with_val(precision_bits, value))
            .collect();
        for row in 0..n {
            for column in 0..row {
                let mut difference = data[row * n + column].clone();
                difference -= &data[column * n + row];
                difference.abs_mut();
                if difference > symmetry_tolerance.clone() {
                    return Err(OperatorError::InvalidData(format!(
                        "HP matrix is not symmetric at ({row}, {column})"
                    )));
                }
            }
        }
        let mut norm_bound = Float::with_val(precision_bits, 0);
        for row in 0..n {
            let mut row_sum = Float::with_val(precision_bits, 0);
            for column in 0..n {
                let mut term = data[row * n + column].clone();
                term.abs_mut();
                row_sum += term;
            }
            if row_sum > norm_bound {
                norm_bound = row_sum;
            }
        }
        Ok(Self {
            n,
            data,
            norm_bound,
            precision_bits,
            name: name.into(),
        })
    }

    pub fn data(&self) -> &[rug::Float] {
        &self.data
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
}

#[cfg(feature = "hp")]
impl LinearOperator<rug::Float> for DenseSymmetricHp {
    fn dimension(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[rug::Float], y: &mut [rug::Float]) -> Result<(), OperatorError> {
        if x.len() != self.n {
            return Err(OperatorError::DimensionMismatch {
                expected: self.n,
                actual: x.len(),
            });
        }
        if y.len() != self.n {
            return Err(OperatorError::DimensionMismatch {
                expected: self.n,
                actual: y.len(),
            });
        }
        for (row, output) in self.data.chunks_exact(self.n).zip(y.iter_mut()) {
            let mut sum = rug::Float::with_val(self.precision_bits, 0);
            for (entry, component) in row.iter().zip(x) {
                let mut term = entry.clone();
                term *= component;
                sum += term;
            }
            *output = sum;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.n,
            MatrixStructure::Dense,
            "rug_mpfr",
        );
        metadata.symmetric = true;
        metadata
            .tags
            .push(format!("precision_bits={}", self.precision_bits));
        metadata
    }

    fn norm_bound(&self) -> Option<rug::Float> {
        Some(self.norm_bound.clone())
    }
}

#[cfg(feature = "hp")]
impl SymmetricOperator<rug::Float> for DenseSymmetricHp {}

#[cfg(feature = "hp")]
#[derive(Clone, Debug)]
pub struct TridiagonalHp {
    diagonal: Vec<rug::Float>,
    off_diagonal: Vec<rug::Float>,
    norm_bound: rug::Float,
    precision_bits: u32,
    name: String,
}

#[cfg(feature = "hp")]
impl TridiagonalHp {
    pub fn new(
        name: impl Into<String>,
        diagonal: Vec<rug::Float>,
        off_diagonal: Vec<rug::Float>,
        precision_bits: u32,
    ) -> Result<Self, OperatorError> {
        use rug::Float;
        if diagonal.is_empty() || off_diagonal.len() + 1 != diagonal.len() {
            return Err(OperatorError::InvalidData(
                "HP tridiagonal requires off_diagonal.len() + 1 == diagonal.len()".to_owned(),
            ));
        }
        if precision_bits < 32 {
            return Err(OperatorError::InvalidData(
                "HP precision must be at least 32 bits".to_owned(),
            ));
        }
        let diagonal: Vec<Float> = diagonal
            .into_iter()
            .map(|value| Float::with_val(precision_bits, value))
            .collect();
        let off_diagonal: Vec<Float> = off_diagonal
            .into_iter()
            .map(|value| Float::with_val(precision_bits, value))
            .collect();
        let mut norm_bound = Float::with_val(precision_bits, 0);
        for row in 0..diagonal.len() {
            let mut row_sum = diagonal[row].clone();
            row_sum.abs_mut();
            if row > 0 {
                let mut term = off_diagonal[row - 1].clone();
                term.abs_mut();
                row_sum += term;
            }
            if row + 1 < diagonal.len() {
                let mut term = off_diagonal[row].clone();
                term.abs_mut();
                row_sum += term;
            }
            if row_sum > norm_bound {
                norm_bound = row_sum;
            }
        }
        Ok(Self {
            diagonal,
            off_diagonal,
            norm_bound,
            precision_bits,
            name: name.into(),
        })
    }

    pub fn diagonal(&self) -> &[rug::Float] {
        &self.diagonal
    }

    pub fn off_diagonal(&self) -> &[rug::Float] {
        &self.off_diagonal
    }

    /// Working precision retained by the operator and used for every action.
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
}

#[cfg(feature = "hp")]
impl LinearOperator<rug::Float> for TridiagonalHp {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    fn apply(&self, x: &[rug::Float], y: &mut [rug::Float]) -> Result<(), OperatorError> {
        let n = self.dimension();
        if x.len() != n {
            return Err(OperatorError::DimensionMismatch {
                expected: n,
                actual: x.len(),
            });
        }
        if y.len() != n {
            return Err(OperatorError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }
        for row in 0..n {
            let mut value = self.diagonal[row].clone();
            value *= &x[row];
            if row > 0 {
                let mut term = self.off_diagonal[row - 1].clone();
                term *= &x[row - 1];
                value += term;
            }
            if row + 1 < n {
                let mut term = self.off_diagonal[row].clone();
                term *= &x[row + 1];
                value += term;
            }
            y[row] = value;
        }
        Ok(())
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            self.name.clone(),
            self.dimension(),
            MatrixStructure::Tridiagonal,
            "rug_mpfr",
        );
        metadata.symmetric = true;
        metadata
            .tags
            .push(format!("precision_bits={}", self.precision_bits));
        metadata
    }

    fn norm_bound(&self) -> Option<rug::Float> {
        Some(self.norm_bound.clone())
    }
}

#[cfg(feature = "hp")]
impl SymmetricOperator<rug::Float> for TridiagonalHp {}

#[cfg(all(test, feature = "hp"))]
mod hp_operator_tests {
    use super::*;
    use rug::ops::Pow;
    use rug::Float;

    #[test]
    fn hp_tridiagonal_action_matches_expected() {
        let precision = 256;
        let operator = TridiagonalHp::new(
            "hp-t",
            vec![
                Float::with_val(precision, 2),
                Float::with_val(precision, 3),
                Float::with_val(precision, 4),
            ],
            vec![
                Float::with_val(precision, -1),
                Float::with_val(precision, 0.5),
            ],
            precision,
        )
        .unwrap();
        let x = vec![
            Float::with_val(precision, 1),
            Float::with_val(precision, 2),
            Float::with_val(precision, 3),
        ];
        let mut y = vec![Float::with_val(precision, 0); 3];
        operator.apply(&x, &mut y).unwrap();
        assert_eq!(y[0], 0);
        assert_eq!(y[1], 6.5);
        assert_eq!(y[2], 13);
    }

    #[test]
    fn explicit_operator_precision_controls_below_f64_resolution() {
        fn action_at(precision_bits: u32) -> Float {
            let source_precision = 256;
            let mut distinguished_entry = Float::with_val(source_precision, 1);
            distinguished_entry += Float::with_val(source_precision, 2).pow(-200);
            let operator = DenseSymmetricHp::new(
                "explicit-precision",
                2,
                vec![
                    distinguished_entry,
                    Float::with_val(source_precision, 0),
                    Float::with_val(source_precision, 0),
                    Float::with_val(source_precision, 1),
                ],
                precision_bits,
                &Float::with_val(precision_bits, 0),
            )
            .unwrap();
            assert_eq!(operator.precision_bits(), precision_bits);

            let x = vec![
                Float::with_val(precision_bits, 1),
                Float::with_val(precision_bits, 0),
            ];
            let mut y = vec![Float::with_val(precision_bits, 0); 2];
            operator.apply(&x, &mut y).unwrap();
            y.remove(0)
        }

        let low = action_at(128);
        let high = action_at(256);
        assert_eq!(low, 1);
        assert!(high > 1);
        let mut expected = Float::with_val(256, 1);
        expected += Float::with_val(256, 2).pow(-200);
        assert_eq!(high, expected);
    }
}
