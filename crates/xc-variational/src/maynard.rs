//! Exact finite-degree Maynard-Tao `M_k` reference engine.
//!
//! This module intentionally starts with the slow, transparent O(N^2)
//! reference formulas.  Matrix-free and structured accelerations must be
//! cross-checked against this implementation before they can be trusted.

use rug::{float::Round, Float, Integer, Rational};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use xc_certify::ExactRationalRecord;
use xc_operator::{
    LinearOperator, MatrixStructure, OperatorError, OperatorMetadata, PositiveDefiniteMetric,
    SymmetricOperator,
};

pub fn mk_artifact_reuse_plan() -> xc_core::ArtifactReusePlan {
    use xc_core::{ArtifactReuseNode, ArtifactReusePlan};
    let node = |kind: &str, dependencies: &[&str], invalidated_by: &[&str]| ArtifactReuseNode {
        kind: kind.to_owned(),
        independently_cacheable: true,
        dependencies: dependencies
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        invalidated_by: invalidated_by
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
    };
    ArtifactReusePlan {
        schema_version: 1,
        domain: "mk".to_owned(),
        semantics_version: "maynard-mk-v0.13.0-v1".to_owned(),
        artifacts: vec![
            node("symmetric_basis", &[], &["k", "degree", "basis_semantics"]),
            node(
                "exact_forms",
                &["symmetric_basis"],
                &["integration_semantics", "normalization"],
            ),
            node(
                "operator_representation",
                &["exact_forms"],
                &["acceleration", "approximation_certificate"],
            ),
            node(
                "solver_candidate",
                &["operator_representation"],
                &["solver_plan", "precision_policy", "seed_policy"],
            ),
            node(
                "scale_diagnostics",
                &["operator_representation", "solver_candidate"],
                &["resource_policy", "diagnostic_policy"],
            ),
            node(
                "exact_candidate_certificate",
                &["exact_forms", "solver_candidate"],
                &["certificate_policy", "rationalization_policy"],
            ),
        ],
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MkError {
    InvalidProblem(String),
    DimensionMismatch { expected: usize, actual: usize },
    NonPositiveDenominator,
}

/// Symmetric `M_k` adapter to the common generalized capability planner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MkSolverPlanningRequest {
    pub basis_dimension: usize,
    pub requested_candidates: usize,
    pub assurance: xc_core::AssuranceLevel,
    pub precision: xc_core::PrecisionPolicy,
    pub matrix_materialized: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MkSolverPlanner;

impl xc_solver::DomainSolverPlanner for MkSolverPlanner {
    type Request = MkSolverPlanningRequest;

    fn domain_id(&self) -> &'static str {
        "maynard_tao_mk_symmetric"
    }

    fn solver_input(
        &self,
        request: &Self::Request,
    ) -> Result<xc_solver::SolverPlannerInput, xc_solver::SolverError> {
        Ok(xc_solver::SolverPlannerInput {
            structure: if request.matrix_materialized {
                xc_operator::MatrixStructure::Dense
            } else {
                xc_operator::MatrixStructure::MatrixFree
            },
            dimension: request.basis_dimension,
            target: xc_core::EigenTarget::AlgebraicLargest,
            requested_eigenpairs: request.requested_candidates,
            assurance: request.assurance,
            precision: request.precision,
            matrix_materialized: request.matrix_materialized,
            generalized: true,
        })
    }

    fn planning_rationale(&self, request: &Self::Request) -> Vec<String> {
        vec![format!(
            "M_k maximizes J/I over {} requested candidate subspaces without forming I inverse",
            request.requested_candidates
        )]
    }
}

impl Display for MkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProblem(message) => write!(f, "invalid M_k problem: {message}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonPositiveDenominator => {
                f.write_str("M_k Rayleigh denominator must be positive")
            }
        }
    }
}

impl Error for MkError {}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct MultiIndex(pub Vec<u32>);

impl MultiIndex {
    pub fn total_degree(&self) -> usize {
        self.0.iter().map(|&x| x as usize).sum()
    }

    pub fn dimension(&self) -> usize {
        self.0.len()
    }
}

pub fn enumerate_multi_indices(k: usize, degree: usize) -> Result<Vec<MultiIndex>, MkError> {
    if k == 0 {
        return Err(MkError::InvalidProblem("k must be positive".to_owned()));
    }
    fn recurse(
        position: usize,
        k: usize,
        remaining: usize,
        current: &mut Vec<u32>,
        output: &mut Vec<MultiIndex>,
    ) {
        if position + 1 == k {
            for value in 0..=remaining {
                current.push(value as u32);
                output.push(MultiIndex(current.clone()));
                current.pop();
            }
            return;
        }
        for value in 0..=remaining {
            current.push(value as u32);
            recurse(position + 1, k, remaining - value, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    recurse(0, k, degree, &mut Vec::with_capacity(k), &mut output);
    output.sort_by(|a, b| {
        a.total_degree()
            .cmp(&b.total_degree())
            .then_with(|| a.0.cmp(&b.0))
    });
    output.dedup();
    Ok(output)
}

fn rational(numerator: Integer, denominator: Integer) -> Rational {
    Rational::from((numerator, denominator))
}

#[derive(Clone, Debug)]
pub struct MkMonomialReference {
    k: usize,
    degree: usize,
    indices: Vec<MultiIndex>,
    factorials: Vec<Integer>,
}

impl MkMonomialReference {
    pub fn new(k: usize, degree: usize) -> Result<Self, MkError> {
        let indices = enumerate_multi_indices(k, degree)?;
        // Largest factorial index in J is k + 1 + 2D; numerator uses 2D + 2.
        let maximum = (k + 1 + 2 * degree).max(2 * degree + 2);
        let mut factorials = Vec::with_capacity(maximum + 1);
        let mut current = Integer::from(1);
        factorials.push(current.clone());
        for n in 1..=maximum {
            current *= n as u32;
            factorials.push(current.clone());
        }
        Ok(Self {
            k,
            degree,
            indices,
            factorials,
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn dimension(&self) -> usize {
        self.indices.len()
    }

    pub fn indices(&self) -> &[MultiIndex] {
        &self.indices
    }

    fn factorial(&self, n: usize) -> &Integer {
        &self.factorials[n]
    }

    pub fn i_entry(&self, a: &MultiIndex, b: &MultiIndex) -> Result<Rational, MkError> {
        self.check_indices(a, b)?;
        let mut numerator = Integer::from(1);
        for i in 0..self.k {
            numerator *= self.factorial((a.0[i] + b.0[i]) as usize);
        }
        let denominator = self
            .factorial(self.k + a.total_degree() + b.total_degree())
            .clone();
        Ok(rational(numerator, denominator))
    }

    pub fn j_entry(
        &self,
        distinguished_axis: usize,
        a: &MultiIndex,
        b: &MultiIndex,
    ) -> Result<Rational, MkError> {
        self.check_indices(a, b)?;
        if distinguished_axis >= self.k {
            return Err(MkError::InvalidProblem(format!(
                "distinguished axis {distinguished_axis} is outside 0..{}",
                self.k
            )));
        }
        let am = a.0[distinguished_axis] as usize;
        let bm = b.0[distinguished_axis] as usize;
        let mut numerator = self.factorial(am + bm + 2).clone();
        for i in 0..self.k {
            if i != distinguished_axis {
                numerator *= self.factorial((a.0[i] + b.0[i]) as usize);
            }
        }
        let mut denominator = Integer::from((am + 1) as u64);
        denominator *= (bm + 1) as u64;
        denominator *= self.factorial(self.k + 1 + a.total_degree() + b.total_degree());
        Ok(rational(numerator, denominator))
    }

    pub fn j_total_entry(&self, a: &MultiIndex, b: &MultiIndex) -> Result<Rational, MkError> {
        let mut total = Rational::from((0, 1));
        for m in 0..self.k {
            total += self.j_entry(m, a, b)?;
        }
        Ok(total)
    }

    pub fn dense_i_exact(&self) -> Result<Vec<Rational>, MkError> {
        let mut matrix = Vec::with_capacity(self.dimension() * self.dimension());
        for row in &self.indices {
            for column in &self.indices {
                matrix.push(self.i_entry(row, column)?);
            }
        }
        Ok(matrix)
    }

    pub fn dense_j_total_exact(&self) -> Result<Vec<Rational>, MkError> {
        let mut matrix = Vec::with_capacity(self.dimension() * self.dimension());
        for row in &self.indices {
            for column in &self.indices {
                matrix.push(self.j_total_entry(row, column)?);
            }
        }
        Ok(matrix)
    }

    pub fn dense_i_f64(&self) -> Result<Vec<f64>, MkError> {
        Ok(self
            .dense_i_exact()?
            .into_iter()
            .map(|value| value.to_f64())
            .collect())
    }

    pub fn dense_j_total_f64(&self) -> Result<Vec<f64>, MkError> {
        Ok(self
            .dense_j_total_exact()?
            .into_iter()
            .map(|value| value.to_f64())
            .collect())
    }

    pub fn quadratic_i(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        self.check_coefficients(coefficients)?;
        self.quadratic_form(coefficients, |a, b| self.i_entry(a, b))
    }

    pub fn quadratic_j_total(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        self.check_coefficients(coefficients)?;
        self.quadratic_form(coefficients, |a, b| self.j_total_entry(a, b))
    }

    pub fn rayleigh_quotient(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        let denominator = self.quadratic_i(coefficients)?;
        if denominator <= 0 {
            return Err(MkError::NonPositiveDenominator);
        }
        Ok(self.quadratic_j_total(coefficients)? / denominator)
    }

    /// Evaluates and records an exact Maynard--Tao finite-space Rayleigh quotient.
    ///
    /// # Mathematical semantics
    /// Computes the exact `J/I` quotient for the supplied coefficients in the
    /// full monomial polynomial space represented by this reference engine.
    ///
    /// # Precision
    /// All matrix entries, quadratic forms, and the quotient use arbitrary-size
    /// exact rational arithmetic; no binary64 conversion occurs.
    ///
    /// # Failure states
    /// A coefficient dimension mismatch, invalid basis, arithmetic construction
    /// failure, or nonpositive denominator returns `MkError` and no certificate.
    ///
    /// # Assurance and validity
    /// The certificate proves replay of this exact finite Rayleigh quotient. It
    /// does not by itself prove global optimality outside the declared space or
    /// validate every analytic step of a prime-gap theorem.
    ///
    /// # Cache effects
    /// This method has no implicit cache effects. Persisting or publishing the
    /// certificate uses the common artifact and provenance layer.
    ///
    /// # Example
    /// Compiled example: `crates/xc-variational/examples/mk_constant.rs`.
    pub fn certificate(&self, coefficients: &[Rational]) -> Result<MkRayleighCertificate, MkError> {
        let numerator = self.quadratic_j_total(coefficients)?;
        let denominator = self.quadratic_i(coefficients)?;
        if denominator <= 0 {
            return Err(MkError::NonPositiveDenominator);
        }
        let quotient = numerator.clone() / denominator.clone();
        Ok(MkRayleighCertificate {
            k: self.k,
            degree: self.degree,
            dimension: self.dimension(),
            search_space: "full monomial polynomial space".to_owned(),
            numerator: exact_record(&numerator),
            denominator: exact_record(&denominator),
            quotient: exact_record(&quotient),
        })
    }

    /// Slow matrix-free f64 reference action for exploratory and validation
    /// comparisons.  Every entry is generated from the exact formula before
    /// explicit conversion to f64.
    pub fn apply_j_total_f64(&self, x: &[f64], y: &mut [f64]) -> Result<(), MkError> {
        if x.len() != self.dimension() || y.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len().min(y.len()),
            });
        }
        for (row, yi) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (col, &xj) in x.iter().enumerate() {
                sum += self
                    .j_total_entry(&self.indices[row], &self.indices[col])?
                    .to_f64()
                    * xj;
            }
            *yi = sum;
        }
        Ok(())
    }

    pub fn apply_i_f64(&self, x: &[f64], y: &mut [f64]) -> Result<(), MkError> {
        if x.len() != self.dimension() || y.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len().min(y.len()),
            });
        }
        for (row, yi) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (col, &xj) in x.iter().enumerate() {
                sum += self
                    .i_entry(&self.indices[row], &self.indices[col])?
                    .to_f64()
                    * xj;
            }
            *yi = sum;
        }
        Ok(())
    }

    fn quadratic_form<F>(
        &self,
        coefficients: &[Rational],
        mut entry: F,
    ) -> Result<Rational, MkError>
    where
        F: FnMut(&MultiIndex, &MultiIndex) -> Result<Rational, MkError>,
    {
        let mut total = Rational::from((0, 1));
        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                if coefficients[i] == 0 || coefficients[j] == 0 {
                    continue;
                }
                let mut term = coefficients[i].clone();
                term *= &coefficients[j];
                term *= entry(&self.indices[i], &self.indices[j])?;
                total += term;
            }
        }
        Ok(total)
    }

    fn check_coefficients(&self, coefficients: &[Rational]) -> Result<(), MkError> {
        if coefficients.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: coefficients.len(),
            });
        }
        Ok(())
    }

    fn check_indices(&self, a: &MultiIndex, b: &MultiIndex) -> Result<(), MkError> {
        if a.dimension() != self.k || b.dimension() != self.k {
            return Err(MkError::InvalidProblem(
                "multi-index dimension does not match k".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Exact coefficient vector for Maynard's published `M_5 > 2` witness.
///
/// This is Equation (8.16) of *Small gaps between primes* (Annals of
/// Mathematics 181 (2015), arXiv:1311.4600):
///
/// `P = (1-P1)P2 + (7/10)(1-P1)^2 + (1/14)P2 - (3/14)(1-P1)`.
///
/// The expression is expanded independently into the canonical full
/// monomial basis.  Higher-degree coordinates, when present, are exactly
/// zero.
pub fn maynard_2015_m5_candidate(
    reference: &MkMonomialReference,
) -> Result<Vec<Rational>, MkError> {
    if reference.k() != 5 || reference.degree() < 3 {
        return Err(MkError::InvalidProblem(
            "Maynard 2015 M5 witness requires k=5 and degree at least 3".to_owned(),
        ));
    }
    reference
        .indices()
        .iter()
        .map(|index| {
            let degree = index.total_degree();
            let nonzero = index.0.iter().filter(|value| **value != 0).count();
            let maximum = index.0.iter().copied().max().unwrap_or(0);
            let coefficient = match (degree, nonzero, maximum) {
                (0, 0, 0) => Rational::from((17, 35)),
                (1, 1, 1) => Rational::from((-83, 70)),
                (2, 1, 2) => Rational::from((62, 35)),
                (2, 2, 1) => Rational::from((7, 5)),
                (3, 1, 3) | (3, 2, 2) => Rational::from((-1, 1)),
                _ => Rational::from((0, 1)),
            };
            Ok(coefficient)
        })
        .collect()
}

/// Produce the exact finite-form certificate for the published Maynard
/// witness.  The expected quotient is `1417255/708216`, strictly above 2.
pub fn maynard_2015_m5_certificate() -> Result<MkRayleighCertificate, MkError> {
    let reference = MkMonomialReference::new(5, 3)?;
    let candidate = maynard_2015_m5_candidate(&reference)?;
    reference.certificate(&candidate)
}

/// Canonical nonincreasing integer partition used to index fully symmetric
/// orbit sums. Zero parts are omitted; the empty partition represents the
/// constant polynomial.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct IntegerPartition(pub Vec<u32>);

impl IntegerPartition {
    pub fn total_degree(&self) -> usize {
        self.0.iter().map(|&part| part as usize).sum()
    }

    pub fn length(&self) -> usize {
        self.0.len()
    }

    pub fn validate(&self, maximum_length: usize) -> Result<(), MkError> {
        if self.length() > maximum_length {
            return Err(MkError::InvalidProblem(format!(
                "partition length {} exceeds k={maximum_length}",
                self.length()
            )));
        }
        if self.0.contains(&0) || self.0.windows(2).any(|parts| parts[0] < parts[1]) {
            return Err(MkError::InvalidProblem(
                "partition parts must be positive and nonincreasing".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Enumerate every partition of total degree at most `degree` with at most
/// `k` positive parts. Ordering is by total degree and then lexicographic part
/// sequence, and is therefore stable across runs.
pub fn enumerate_integer_partitions(
    k: usize,
    degree: usize,
) -> Result<Vec<IntegerPartition>, MkError> {
    if k == 0 {
        return Err(MkError::InvalidProblem("k must be positive".to_owned()));
    }

    fn recurse(
        remaining: usize,
        maximum_part: usize,
        maximum_length: usize,
        current: &mut Vec<u32>,
        output: &mut Vec<IntegerPartition>,
    ) {
        if remaining == 0 {
            output.push(IntegerPartition(current.clone()));
            return;
        }
        if current.len() == maximum_length {
            return;
        }
        for part in (1..=remaining.min(maximum_part)).rev() {
            current.push(part as u32);
            recurse(remaining - part, part, maximum_length, current, output);
            current.pop();
        }
    }

    let mut output = vec![IntegerPartition(Vec::new())];
    for total_degree in 1..=degree {
        recurse(
            total_degree,
            total_degree,
            k,
            &mut Vec::with_capacity(k),
            &mut output,
        );
    }
    output.sort_by(|left, right| {
        left.total_degree()
            .cmp(&right.total_degree())
            .then_with(|| left.0.cmp(&right.0))
    });
    output.dedup();
    Ok(output)
}

fn next_permutation(values: &mut [u32]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .expect("a permutation successor exists after the pivot");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

fn partition_orbit(partition: &IntegerPartition, k: usize) -> Result<Vec<MultiIndex>, MkError> {
    partition.validate(k)?;
    let mut padded = partition.0.clone();
    padded.resize(k, 0);
    padded.sort_unstable();
    let mut orbit = vec![MultiIndex(padded.clone())];
    while next_permutation(&mut padded) {
        orbit.push(MultiIndex(padded.clone()));
    }
    Ok(orbit)
}

/// One unnormalized monomial orbit sum
/// `m_lambda = sum_{alpha in orbit(lambda)} x^alpha`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SymmetricMonomialOrbit {
    pub partition: IntegerPartition,
    pub members: Vec<MultiIndex>,
}

/// Exact reference engine for the declared fully symmetric polynomial
/// subspace. Basis vectors are unnormalized orbit sums, never implicit orbit
/// averages; this normalization is part of the serialized finite semantics.
#[derive(Clone, Debug)]
pub struct MkSymmetricReference {
    monomial: MkMonomialReference,
    orbits: Vec<SymmetricMonomialOrbit>,
    monomial_positions: BTreeMap<MultiIndex, usize>,
}

impl MkSymmetricReference {
    pub const SEARCH_SPACE: &'static str =
        "fully symmetric polynomial subspace in unnormalized monomial orbit-sum basis";

    pub fn new(k: usize, degree: usize) -> Result<Self, MkError> {
        let monomial = MkMonomialReference::new(k, degree)?;
        let partitions = enumerate_integer_partitions(k, degree)?;
        let orbits = partitions
            .into_iter()
            .map(|partition| {
                Ok(SymmetricMonomialOrbit {
                    members: partition_orbit(&partition, k)?,
                    partition,
                })
            })
            .collect::<Result<Vec<_>, MkError>>()?;
        let monomial_positions = monomial
            .indices()
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, index)| (index, position))
            .collect();
        Ok(Self {
            monomial,
            orbits,
            monomial_positions,
        })
    }

    pub fn k(&self) -> usize {
        self.monomial.k()
    }

    pub fn degree(&self) -> usize {
        self.monomial.degree()
    }

    pub fn dimension(&self) -> usize {
        self.orbits.len()
    }

    pub fn full_monomial_dimension(&self) -> usize {
        self.monomial.dimension()
    }

    pub fn orbits(&self) -> &[SymmetricMonomialOrbit] {
        &self.orbits
    }

    pub fn i_entry(&self, row: usize, column: usize) -> Result<Rational, MkError> {
        self.check_entry_indices(row, column)?;
        let mut total = Rational::from((0, 1));
        for left in &self.orbits[row].members {
            for right in &self.orbits[column].members {
                total += self.monomial.i_entry(left, right)?;
            }
        }
        Ok(total)
    }

    pub fn j_total_entry(&self, row: usize, column: usize) -> Result<Rational, MkError> {
        self.check_entry_indices(row, column)?;
        let mut total = Rational::from((0, 1));
        for left in &self.orbits[row].members {
            for right in &self.orbits[column].members {
                total += self.monomial.j_total_entry(left, right)?;
            }
        }
        Ok(total)
    }

    pub fn dense_i_exact(&self) -> Result<Vec<Rational>, MkError> {
        self.dense_exact(|row, column| self.i_entry(row, column))
    }

    pub fn dense_j_total_exact(&self) -> Result<Vec<Rational>, MkError> {
        self.dense_exact(|row, column| self.j_total_entry(row, column))
    }

    pub fn dense_i_f64(&self) -> Result<Vec<f64>, MkError> {
        Ok(self
            .dense_i_exact()?
            .into_iter()
            .map(|value| value.to_f64())
            .collect())
    }

    pub fn dense_j_total_f64(&self) -> Result<Vec<f64>, MkError> {
        Ok(self
            .dense_j_total_exact()?
            .into_iter()
            .map(|value| value.to_f64())
            .collect())
    }

    /// Embed symmetric orbit coefficients into the complete monomial basis.
    /// This is also the independent source-of-truth route used by exact
    /// quotient tests and certificates.
    pub fn expand_coefficients_exact(
        &self,
        coefficients: &[Rational],
    ) -> Result<Vec<Rational>, MkError> {
        self.check_coefficients(coefficients)?;
        let mut expanded = vec![Rational::from((0, 1)); self.full_monomial_dimension()];
        for (coefficient, orbit) in coefficients.iter().zip(&self.orbits) {
            for member in &orbit.members {
                let position = self.monomial_positions.get(member).ok_or_else(|| {
                    MkError::InvalidProblem(format!(
                        "orbit member {:?} is absent from the monomial embedding",
                        member.0
                    ))
                })?;
                expanded[*position] += coefficient;
            }
        }
        Ok(expanded)
    }

    pub fn quadratic_i(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        let expanded = self.expand_coefficients_exact(coefficients)?;
        self.monomial.quadratic_i(&expanded)
    }

    pub fn quadratic_j_total(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        let expanded = self.expand_coefficients_exact(coefficients)?;
        self.monomial.quadratic_j_total(&expanded)
    }

    pub fn rayleigh_quotient(&self, coefficients: &[Rational]) -> Result<Rational, MkError> {
        let denominator = self.quadratic_i(coefficients)?;
        if denominator <= 0 {
            return Err(MkError::NonPositiveDenominator);
        }
        Ok(self.quadratic_j_total(coefficients)? / denominator)
    }

    pub fn certificate(
        &self,
        coefficients: &[Rational],
    ) -> Result<MkSymmetricRayleighCertificate, MkError> {
        let numerator = self.quadratic_j_total(coefficients)?;
        let denominator = self.quadratic_i(coefficients)?;
        if denominator <= 0 {
            return Err(MkError::NonPositiveDenominator);
        }
        let quotient = numerator.clone() / denominator.clone();
        Ok(MkSymmetricRayleighCertificate {
            k: self.k(),
            degree: self.degree(),
            symmetric_dimension: self.dimension(),
            monomial_embedding_dimension: self.full_monomial_dimension(),
            search_space: Self::SEARCH_SPACE.to_owned(),
            basis_normalization: "one coefficient on every distinct monomial in the orbit"
                .to_owned(),
            numerator: exact_record(&numerator),
            denominator: exact_record(&denominator),
            quotient: exact_record(&quotient),
        })
    }

    pub fn apply_i_f64(&self, x: &[f64], y: &mut [f64]) -> Result<(), MkError> {
        self.apply_f64(x, y, |row, column| self.i_entry(row, column))
    }

    pub fn apply_j_total_f64(&self, x: &[f64], y: &mut [f64]) -> Result<(), MkError> {
        self.apply_f64(x, y, |row, column| self.j_total_entry(row, column))
    }

    /// Stream the exact symmetric `I` entries into an MPFR matrix-vector
    /// product at `precision_bits` without storing the dense form.
    pub fn apply_i_hp(
        &self,
        x: &[Float],
        y: &mut [Float],
        precision_bits: u32,
    ) -> Result<(), MkError> {
        self.apply_hp(x, y, precision_bits, |row, column| {
            self.i_entry(row, column)
        })
    }

    /// Stream the exact symmetric total-`J` entries into an MPFR
    /// matrix-vector product at `precision_bits` without dense storage.
    pub fn apply_j_total_hp(
        &self,
        x: &[Float],
        y: &mut [Float],
        precision_bits: u32,
    ) -> Result<(), MkError> {
        self.apply_hp(x, y, precision_bits, |row, column| {
            self.j_total_entry(row, column)
        })
    }

    pub fn feasibility(&self) -> MkSymmetricFeasibility {
        let dimension = self.dimension() as u64;
        let embedding_dimension = self.full_monomial_dimension() as u64;
        MkSymmetricFeasibility {
            k: self.k(),
            degree: self.degree(),
            dimension: self.dimension(),
            basis_count: self.dimension(),
            monomial_embedding_basis_count: self.full_monomial_dimension(),
            total_orbit_members: self.orbits.iter().map(|orbit| orbit.members.len()).sum(),
            f64_vector_bytes: dimension.saturating_mul(8),
            streamed_operator_workspace_bytes: dimension.saturating_mul(16),
            dense_i_and_j_f64_bytes: dimension.saturating_mul(dimension).saturating_mul(16),
            dense_entry_count: dimension.saturating_mul(dimension),
            exact_entry_evaluations_per_streamed_form_application: dimension
                .saturating_mul(dimension),
            exact_certificate_term_upper_bound: embedding_dimension
                .saturating_mul(embedding_dimension),
            exact_certificate_multiply_add_upper_bound: embedding_dimension
                .saturating_mul(embedding_dimension)
                .saturating_mul(4),
            symmetry_restriction: Self::SEARCH_SPACE.to_owned(),
        }
    }

    fn dense_exact<F>(&self, mut entry: F) -> Result<Vec<Rational>, MkError>
    where
        F: FnMut(usize, usize) -> Result<Rational, MkError>,
    {
        let mut matrix = Vec::with_capacity(self.dimension() * self.dimension());
        for row in 0..self.dimension() {
            for column in 0..self.dimension() {
                matrix.push(entry(row, column)?);
            }
        }
        Ok(matrix)
    }

    fn apply_f64<F>(&self, x: &[f64], y: &mut [f64], mut entry: F) -> Result<(), MkError>
    where
        F: FnMut(usize, usize) -> Result<Rational, MkError>,
    {
        if x.len() != self.dimension() || y.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len().min(y.len()),
            });
        }
        for (row, output) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (column, coefficient) in x.iter().enumerate() {
                sum += entry(row, column)?.to_f64() * coefficient;
            }
            *output = sum;
        }
        Ok(())
    }

    fn apply_hp<F>(
        &self,
        x: &[Float],
        y: &mut [Float],
        precision_bits: u32,
        mut entry: F,
    ) -> Result<(), MkError>
    where
        F: FnMut(usize, usize) -> Result<Rational, MkError>,
    {
        if precision_bits <= 32 {
            return Err(MkError::InvalidProblem(
                "MPFR streamed actions require precision above 32 bits".to_owned(),
            ));
        }
        if x.len() != self.dimension() || y.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len().min(y.len()),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(MkError::InvalidProblem(
                "MPFR streamed action input contains a nonfinite value".to_owned(),
            ));
        }
        for (row, output) in y.iter_mut().enumerate() {
            let mut sum = Float::with_val(precision_bits, 0);
            for (column, coefficient) in x.iter().enumerate() {
                let mut term = Float::with_val(precision_bits, entry(row, column)?);
                term *= coefficient;
                sum += term;
            }
            *output = sum;
        }
        Ok(())
    }

    fn check_entry_indices(&self, row: usize, column: usize) -> Result<(), MkError> {
        if row >= self.dimension() || column >= self.dimension() {
            return Err(MkError::InvalidProblem(format!(
                "symmetric basis entry ({row}, {column}) is outside dimension {}",
                self.dimension()
            )));
        }
        Ok(())
    }

    fn check_coefficients(&self, coefficients: &[Rational]) -> Result<(), MkError> {
        if coefficients.len() != self.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: coefficients.len(),
            });
        }
        Ok(())
    }

    fn row_sum_norm_bound<F>(&self, mut entry: F) -> Result<f64, MkError>
    where
        F: FnMut(usize, usize) -> Result<Rational, MkError>,
    {
        let mut bound: f64 = 0.0;
        for row in 0..self.dimension() {
            let mut row_sum = 0.0;
            for column in 0..self.dimension() {
                let rounded = entry(row, column)?.to_f64().abs();
                row_sum = next_up_f64(row_sum + next_up_f64(rounded));
            }
            bound = bound.max(row_sum);
        }
        Ok(bound)
    }

    fn row_sum_norm_bound_exact<F>(&self, mut entry: F) -> Result<Rational, MkError>
    where
        F: FnMut(usize, usize) -> Result<Rational, MkError>,
    {
        let mut bound = Rational::from((0, 1));
        for row in 0..self.dimension() {
            let mut row_sum = Rational::from((0, 1));
            for column in 0..self.dimension() {
                let mut value = entry(row, column)?;
                if value < 0 {
                    value = -value;
                }
                row_sum += value;
            }
            if row_sum > bound {
                bound = row_sum;
            }
        }
        Ok(bound)
    }
}

fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value >= 0.0 { bits + 1 } else { bits - 1 })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkSymmetricRayleighCertificate {
    pub k: usize,
    pub degree: usize,
    pub symmetric_dimension: usize,
    pub monomial_embedding_dimension: usize,
    pub search_space: String,
    pub basis_normalization: String,
    pub numerator: ExactRationalRecord,
    pub denominator: ExactRationalRecord,
    pub quotient: ExactRationalRecord,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkSymmetricFeasibility {
    pub k: usize,
    pub degree: usize,
    pub dimension: usize,
    pub basis_count: usize,
    pub monomial_embedding_basis_count: usize,
    pub total_orbit_members: usize,
    pub f64_vector_bytes: u64,
    pub streamed_operator_workspace_bytes: u64,
    pub dense_i_and_j_f64_bytes: u64,
    pub dense_entry_count: u64,
    pub exact_entry_evaluations_per_streamed_form_application: u64,
    pub exact_certificate_term_upper_bound: u64,
    pub exact_certificate_multiply_add_upper_bound: u64,
    pub symmetry_restriction: String,
}

/// Controls and acceptance bounds for the v0.13.0 three-route `M_k`
/// milestone. All source forms are exact rational forms; both numerical routes
/// consume MPFR projections of those same forms.
#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MkThreeRouteAcceptanceOptions {
    pub k: usize,
    pub degree: usize,
    pub precision_bits: u32,
    pub initial_precision_bits: u32,
    pub absolute_residual_tolerance: xc_core::DecimalLiteral,
    pub scaled_backward_error_tolerance: xc_core::DecimalLiteral,
    pub ritz_value_stability_tolerance: xc_core::DecimalLiteral,
    pub eigenvalue_agreement_tolerance: xc_core::DecimalLiteral,
    pub overlap_tolerance: xc_core::DecimalLiteral,
    pub candidate_quotient_agreement_tolerance: xc_core::DecimalLiteral,
    pub maximum_iterations: usize,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MkThreeRouteAcceptanceRecord {
    pub schema_version: u32,
    pub k: usize,
    pub degree: usize,
    pub symmetric_dimension: usize,
    pub precision_bits: u32,
    pub matrix_free_eigenvalue: String,
    pub dense_eigenvalue: String,
    pub eigenvalue_absolute_difference: String,
    pub one_minus_metric_overlap_squared: String,
    pub matrix_free_residual_norm: String,
    pub dense_residual_norm: String,
    pub adaptive_attempt_precisions: Vec<u32>,
    pub candidate_coefficients: Vec<ExactRationalRecord>,
    pub candidate_certificate: MkSymmetricRayleighCertificate,
    pub candidate_absolute_difference: String,
    pub exact_source_forms: bool,
    pub matrix_free_route: String,
    pub dense_route: String,
    pub certification_route: String,
}

#[cfg(feature = "hp")]
fn rational_from_record(record: &ExactRationalRecord) -> Result<Rational, MkError> {
    record
        .validate_syntax()
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let numerator = Integer::from_str_radix(&record.numerator, 10)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let denominator = Integer::from_str_radix(&record.denominator, 10)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    Ok(Rational::from((numerator, denominator)))
}

/// Independently replay the exact candidate certificate and validate every
/// serialized numerical acceptance bound. This verifier does not trust the
/// producer's stored quotient fields.
#[cfg(feature = "hp")]
pub fn verify_mk_three_route_acceptance(
    record: &MkThreeRouteAcceptanceRecord,
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<(), MkError> {
    if record.schema_version != 1
        || record.k != options.k
        || record.degree != options.degree
        || record.precision_bits != options.precision_bits
        || !record.exact_source_forms
        || record.matrix_free_route != "adaptive_matrix_free_generalized_rayleigh_ritz_hp"
        || record.dense_route != "dense_generalized_cholesky_whitening_hp"
        || record.certification_route != "exact_rational_candidate_rayleigh_quotient"
    {
        return Err(MkError::InvalidProblem(
            "three-route M_k record identity or route declaration is invalid".to_owned(),
        ));
    }
    let reference = MkSymmetricReference::new(record.k, record.degree)?;
    if record.symmetric_dimension != reference.dimension()
        || record.candidate_coefficients.len() != reference.dimension()
    {
        return Err(MkError::DimensionMismatch {
            expected: reference.dimension(),
            actual: record.candidate_coefficients.len(),
        });
    }
    let coefficients = record
        .candidate_coefficients
        .iter()
        .map(rational_from_record)
        .collect::<Result<Vec<_>, _>>()?;
    let recomputed = reference.certificate(&coefficients)?;
    if recomputed != record.candidate_certificate {
        return Err(MkError::InvalidProblem(
            "stored M_k candidate certificate does not match exact replay".to_owned(),
        ));
    }
    for (value, bound, name) in [
        (
            &record.eigenvalue_absolute_difference,
            &options.eigenvalue_agreement_tolerance,
            "dense/matrix-free eigenvalue difference",
        ),
        (
            &record.one_minus_metric_overlap_squared,
            &options.overlap_tolerance,
            "metric overlap difference",
        ),
        (
            &record.candidate_absolute_difference,
            &options.candidate_quotient_agreement_tolerance,
            "candidate quotient difference",
        ),
    ] {
        let value = xc_core::DecimalLiteral::new(value.clone())
            .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
        if value
            .cmp_numeric(bound)
            .map_err(|error| MkError::InvalidProblem(error.to_string()))?
            == std::cmp::Ordering::Greater
        {
            return Err(MkError::InvalidProblem(format!(
                "{name} exceeds its acceptance bound"
            )));
        }
    }
    Ok(())
}

/// Execute exact-source dense HP, streamed matrix-free HP, and exact rational
/// candidate-quotient routes as one fail-closed acceptance workflow.
#[cfg(feature = "hp")]
pub fn run_mk_three_route_acceptance(
    options: &MkThreeRouteAcceptanceOptions,
) -> Result<MkThreeRouteAcceptanceRecord, MkError> {
    use xc_core::{EigenTarget, PrecisionEscalation, PrecisionPolicy};
    use xc_operator::GeneralizedEigenProblem;
    use xc_solver::{
        cross_check_generalized_hp_reports, solve_dense_generalized_whitening_hp,
        solve_matrix_free_generalized_adaptive_hp, AdaptiveGeneralizedExtremeOptionsHp,
        AdaptiveGeneralizedExtremeResultHp, DenseGeneralizedProblemHp, GeneralizedExtremeConfigHp,
        HpCrossCheckTolerance,
    };

    if options.precision_bits <= 64
        || options.initial_precision_bits <= 32
        || options.initial_precision_bits > options.precision_bits
        || options.maximum_iterations == 0
    {
        return Err(MkError::InvalidProblem(
            "three-route M_k acceptance requires HP precision, a valid initial precision, and positive iteration limit"
                .to_owned(),
        ));
    }
    let reference = MkSymmetricReference::new(options.k, options.degree)?;
    let dense_j: Vec<Float> = reference
        .dense_j_total_exact()?
        .iter()
        .map(|value| Float::with_val(options.precision_bits, value))
        .collect();
    let dense_i: Vec<Float> = reference
        .dense_i_exact()?
        .iter()
        .map(|value| Float::with_val(options.precision_bits, value))
        .collect();
    let operator = MkSymmetricJOperatorHp::new(&reference, options.precision_bits)?;
    let metric = MkSymmetricIMetricHp::new(&reference, options.precision_bits)?;
    let problem = GeneralizedEigenProblem::new(&operator, &metric)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let escalation = if options.initial_precision_bits == options.precision_bits {
        PrecisionEscalation::Fixed
    } else {
        PrecisionEscalation::AddBits(options.precision_bits - options.initial_precision_bits)
    };
    let adaptive = solve_matrix_free_generalized_adaptive_hp(
        &problem,
        &AdaptiveGeneralizedExtremeOptionsHp {
            target: EigenTarget::AlgebraicLargest,
            absolute_residual_tolerance: options.absolute_residual_tolerance.clone(),
            scaled_backward_error_tolerance: options.scaled_backward_error_tolerance.clone(),
            ritz_value_stability_tolerance: options.ritz_value_stability_tolerance.clone(),
            maximum_iterations: options.maximum_iterations,
            minimum_iterations: 2,
            precision: PrecisionPolicy {
                initial_bits: options.initial_precision_bits,
                maximum_bits: options.precision_bits,
                guard_bits: 0,
                escalation,
            },
        },
    )
    .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let (matrix_free, attempts) = match adaptive {
        AdaptiveGeneralizedExtremeResultHp::Converged { result, attempts } => (*result, attempts),
        AdaptiveGeneralizedExtremeResultHp::Inconclusive {
            attempts, reason, ..
        } => {
            return Err(MkError::InvalidProblem(format!(
                "matrix-free M_k route remained inconclusive: {reason}; attempts={attempts:?}"
            )))
        }
    };
    let dense_problem = DenseGeneralizedProblemHp::new(&dense_j, &dense_i, reference.dimension())
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let dense = solve_dense_generalized_whitening_hp(
        &dense_problem,
        &GeneralizedExtremeConfigHp {
            target: EigenTarget::AlgebraicLargest,
            precision_bits: options.precision_bits,
            absolute_residual_tolerance: options.absolute_residual_tolerance.clone(),
            scaled_backward_error_tolerance: options.scaled_backward_error_tolerance.clone(),
            ritz_value_stability_tolerance: options.ritz_value_stability_tolerance.clone(),
            maximum_iterations: options.maximum_iterations,
            minimum_iterations: 2,
        },
    )
    .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let checked = cross_check_generalized_hp_reports(
        &problem,
        &matrix_free,
        &dense,
        &HpCrossCheckTolerance {
            eigenvalue_absolute: options.eigenvalue_agreement_tolerance.clone(),
            one_minus_overlap_squared: options.overlap_tolerance.clone(),
        },
    )
    .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let coefficients = matrix_free
        .eigenvector
        .iter()
        .map(|value| {
            value.to_rational().ok_or_else(|| {
                MkError::InvalidProblem(
                    "MPFR candidate coefficient could not be converted to an exact rational"
                        .to_owned(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let candidate_certificate = reference.certificate(&coefficients)?;
    let candidate_quotient = rational_from_record(&candidate_certificate.quotient)?;
    let mut candidate_difference = matrix_free.eigenvalue.clone();
    candidate_difference -= Float::with_val(options.precision_bits, &candidate_quotient);
    candidate_difference.abs_mut();
    let candidate_bound = Float::with_val(
        options.precision_bits,
        Float::parse(options.candidate_quotient_agreement_tolerance.as_str())
            .map_err(|error| MkError::InvalidProblem(error.to_string()))?,
    );
    if candidate_difference > candidate_bound {
        return Err(MkError::InvalidProblem(format!(
            "exact candidate quotient differs from the HP optimum by {candidate_difference}"
        )));
    }
    let record = MkThreeRouteAcceptanceRecord {
        schema_version: 1,
        k: options.k,
        degree: options.degree,
        symmetric_dimension: reference.dimension(),
        precision_bits: options.precision_bits,
        matrix_free_eigenvalue: matrix_free.eigenvalue.to_string(),
        dense_eigenvalue: dense.eigenvalue.to_string(),
        eigenvalue_absolute_difference: checked.eigenvalue_absolute_difference.to_string(),
        one_minus_metric_overlap_squared: checked.one_minus_metric_overlap_squared.to_string(),
        matrix_free_residual_norm: matrix_free.residual_norm.to_string(),
        dense_residual_norm: dense.residual_norm.to_string(),
        adaptive_attempt_precisions: attempts
            .iter()
            .map(|attempt| attempt.precision_bits)
            .collect(),
        candidate_coefficients: coefficients.iter().map(exact_record).collect(),
        candidate_certificate,
        candidate_absolute_difference: candidate_difference.to_string(),
        exact_source_forms: true,
        matrix_free_route: "adaptive_matrix_free_generalized_rayleigh_ritz_hp".to_owned(),
        dense_route: "dense_generalized_cholesky_whitening_hp".to_owned(),
        certification_route: "exact_rational_candidate_rayleigh_quotient".to_owned(),
    };
    verify_mk_three_route_acceptance(&record, options)?;
    Ok(record)
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MkScaleAcceptanceOptions {
    pub historical_dense_degree_limit: usize,
    pub target_degree: usize,
    pub precision_bits: u32,
    pub minimum_exact_lower_bound: ExactRationalRecord,
    pub quotient_agreement_tolerance: xc_core::DecimalLiteral,
}

#[cfg(feature = "hp")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MkScaleAcceptanceRecord {
    pub schema_version: u32,
    pub k: usize,
    pub source_degree: usize,
    pub historical_dense_degree_limit: usize,
    pub target_degree: usize,
    pub symmetric_dimension: usize,
    pub full_monomial_embedding_dimension: usize,
    pub precision_bits: u32,
    pub matrix_free_i_applications: usize,
    pub matrix_free_j_applications: usize,
    pub matrix_free_quotient: String,
    pub exact_candidate_coefficients: Vec<ExactRationalRecord>,
    pub exact_certificate: MkSymmetricRayleighCertificate,
    pub quotient_absolute_difference: String,
    pub streamed_working_vector_bytes: u64,
    pub equivalent_dense_forms_bytes: u64,
    pub operator_representation: String,
    pub metric_representation: String,
}

#[cfg(feature = "hp")]
fn maynard_m5_symmetric_candidate(
    reference: &MkSymmetricReference,
) -> Result<Vec<Rational>, MkError> {
    if reference.k() != 5 || reference.degree() < 3 {
        return Err(MkError::InvalidProblem(
            "the permanent M5 scale witness requires k=5 and degree at least 3".to_owned(),
        ));
    }
    reference
        .orbits()
        .iter()
        .map(|orbit| {
            let degree = orbit.partition.total_degree();
            let nonzero = orbit.partition.length();
            let maximum = orbit.partition.0.first().copied().unwrap_or(0) as usize;
            Ok(match (degree, nonzero, maximum) {
                (0, 0, 0) => Rational::from((17, 35)),
                (1, 1, 1) => Rational::from((-83, 70)),
                (2, 1, 2) => Rational::from((62, 35)),
                (2, 2, 1) => Rational::from((7, 5)),
                (3, 1, 3) | (3, 2, 2) => Rational::from((-1, 1)),
                _ => Rational::from((0, 1)),
            })
        })
        .collect()
}

/// Verify the permanent larger-space `M_5` matrix-free run by replaying its
/// exact candidate quotient and all resource/representation invariants.
#[cfg(feature = "hp")]
pub fn verify_mk_scale_acceptance(
    record: &MkScaleAcceptanceRecord,
    options: &MkScaleAcceptanceOptions,
) -> Result<(), MkError> {
    if record.schema_version != 1
        || record.k != 5
        || record.source_degree != 3
        || record.historical_dense_degree_limit != options.historical_dense_degree_limit
        || record.target_degree != options.target_degree
        || record.precision_bits != options.precision_bits
        || record.target_degree <= record.historical_dense_degree_limit
        || record.matrix_free_i_applications == 0
        || record.matrix_free_j_applications == 0
        || record.operator_representation != "matrix_free"
        || record.metric_representation != "matrix_free"
        || record.streamed_working_vector_bytes >= record.equivalent_dense_forms_bytes
    {
        return Err(MkError::InvalidProblem(
            "M_k scale record violates its route, degree, or bounded-memory contract".to_owned(),
        ));
    }
    let reference = MkSymmetricReference::new(5, record.target_degree)?;
    if record.symmetric_dimension != reference.dimension()
        || record.full_monomial_embedding_dimension != reference.full_monomial_dimension()
        || record.exact_candidate_coefficients.len() != reference.dimension()
    {
        return Err(MkError::DimensionMismatch {
            expected: reference.dimension(),
            actual: record.exact_candidate_coefficients.len(),
        });
    }
    let coefficients = record
        .exact_candidate_coefficients
        .iter()
        .map(rational_from_record)
        .collect::<Result<Vec<_>, _>>()?;
    let expected_coefficients = maynard_m5_symmetric_candidate(&reference)?;
    if coefficients != expected_coefficients {
        return Err(MkError::InvalidProblem(
            "scale record candidate is not the exact prolonged M5 witness".to_owned(),
        ));
    }
    let certificate = reference.certificate(&coefficients)?;
    if certificate != record.exact_certificate {
        return Err(MkError::InvalidProblem(
            "scale record exact certificate failed replay".to_owned(),
        ));
    }
    let quotient = rational_from_record(&certificate.quotient)?;
    let minimum = rational_from_record(&options.minimum_exact_lower_bound)?;
    if quotient <= minimum {
        return Err(MkError::InvalidProblem(
            "scale candidate does not exceed the required exact lower bound".to_owned(),
        ));
    }
    let difference = xc_core::DecimalLiteral::new(record.quotient_absolute_difference.clone())
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    if difference
        .cmp_numeric(&options.quotient_agreement_tolerance)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?
        == std::cmp::Ordering::Greater
    {
        return Err(MkError::InvalidProblem(
            "matrix-free quotient disagrees with the exact certificate".to_owned(),
        ));
    }
    Ok(())
}

/// Run the published exact `M_5` witness in a strictly larger symmetric degree
/// space using only streamed HP form applications, then attach its exact
/// rational lower-bound certificate.
#[cfg(feature = "hp")]
pub fn run_mk_scale_acceptance(
    options: &MkScaleAcceptanceOptions,
) -> Result<MkScaleAcceptanceRecord, MkError> {
    if options.target_degree <= options.historical_dense_degree_limit
        || options.historical_dense_degree_limit < 3
        || options.precision_bits <= 64
    {
        return Err(MkError::InvalidProblem(
            "M_k scale acceptance requires target degree above an explicit historical limit of at least three and HP precision"
                .to_owned(),
        ));
    }
    let reference = MkSymmetricReference::new(5, options.target_degree)?;
    let coefficients = maynard_m5_symmetric_candidate(&reference)?;
    let input: Vec<Float> = coefficients
        .iter()
        .map(|value| Float::with_val(options.precision_bits, value))
        .collect();
    let operator = MkSymmetricJOperatorHp::new(&reference, options.precision_bits)?;
    let metric = MkSymmetricIMetricHp::new(&reference, options.precision_bits)?;
    let mut applied_j = vec![Float::with_val(options.precision_bits, 0); reference.dimension()];
    let mut applied_i = vec![Float::with_val(options.precision_bits, 0); reference.dimension()];
    operator
        .apply(&input, &mut applied_j)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    metric
        .apply(&input, &mut applied_i)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    let dot_hp = |left: &[Float], right: &[Float]| {
        let mut sum = Float::with_val(options.precision_bits, 0);
        for (left, right) in left.iter().zip(right) {
            let mut product = Float::with_val(options.precision_bits, left);
            product *= right;
            sum += product;
        }
        sum
    };
    let numerator = dot_hp(&input, &applied_j);
    let denominator = dot_hp(&input, &applied_i);
    if denominator <= 0 {
        return Err(MkError::NonPositiveDenominator);
    }
    let mut matrix_free_quotient = numerator;
    matrix_free_quotient /= denominator;
    let exact_certificate = reference.certificate(&coefficients)?;
    let exact_quotient = rational_from_record(&exact_certificate.quotient)?;
    let mut difference = matrix_free_quotient.clone();
    difference -= Float::with_val(options.precision_bits, exact_quotient);
    difference.abs_mut();
    let scalar_bytes = u64::from(options.precision_bits).div_ceil(8);
    let dimension = reference.dimension() as u64;
    let record = MkScaleAcceptanceRecord {
        schema_version: 1,
        k: 5,
        source_degree: 3,
        historical_dense_degree_limit: options.historical_dense_degree_limit,
        target_degree: options.target_degree,
        symmetric_dimension: reference.dimension(),
        full_monomial_embedding_dimension: reference.full_monomial_dimension(),
        precision_bits: options.precision_bits,
        matrix_free_i_applications: 1,
        matrix_free_j_applications: 1,
        matrix_free_quotient: matrix_free_quotient.to_string(),
        exact_candidate_coefficients: coefficients.iter().map(exact_record).collect(),
        exact_certificate,
        quotient_absolute_difference: difference.to_string(),
        streamed_working_vector_bytes: 3u64.saturating_mul(dimension).saturating_mul(scalar_bytes),
        equivalent_dense_forms_bytes: 2u64
            .saturating_mul(dimension)
            .saturating_mul(dimension)
            .saturating_mul(scalar_bytes),
        operator_representation: match operator.metadata().structure {
            MatrixStructure::MatrixFree => "matrix_free".to_owned(),
            other => format!("{other:?}"),
        },
        metric_representation: match metric.metadata().structure {
            MatrixStructure::MatrixFree => "matrix_free".to_owned(),
            other => format!("{other:?}"),
        },
    };
    verify_mk_scale_acceptance(&record, options)?;
    Ok(record)
}

fn mk_operator_error(error: MkError) -> OperatorError {
    match error {
        MkError::DimensionMismatch { expected, actual } => {
            OperatorError::DimensionMismatch { expected, actual }
        }
        other => OperatorError::ApplicationFailed(other.to_string()),
    }
}

/// Matrix-free f64 discovery adapter for the symmetric `J` form. Exact
/// rational entries remain the source of every streamed action.
pub struct MkSymmetricJOperatorF64<'a> {
    reference: &'a MkSymmetricReference,
    norm_bound: f64,
}

impl<'a> MkSymmetricJOperatorF64<'a> {
    pub fn new(reference: &'a MkSymmetricReference) -> Result<Self, MkError> {
        let norm_bound =
            reference.row_sum_norm_bound(|row, column| reference.j_total_entry(row, column))?;
        if !norm_bound.is_finite() {
            return Err(MkError::InvalidProblem(
                "symmetric J row-sum norm bound is outside the finite f64 range".to_owned(),
            ));
        }
        Ok(Self {
            reference,
            norm_bound,
        })
    }
}

impl LinearOperator<f64> for MkSymmetricJOperatorF64<'_> {
    fn dimension(&self) -> usize {
        self.reference.dimension()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.reference
            .apply_j_total_f64(x, y)
            .map_err(mk_operator_error)
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            "maynard_tao_mk_symmetric_j",
            self.dimension(),
            MatrixStructure::MatrixFree,
            "f64_from_exact_rational_entries",
        );
        metadata.symmetric = true;
        metadata.exact_action = false;
        metadata.tags = vec![
            "mk".to_owned(),
            "fully_symmetric_orbit_sum".to_owned(),
            "discovery_only".to_owned(),
        ];
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.norm_bound)
    }
}

impl SymmetricOperator<f64> for MkSymmetricJOperatorF64<'_> {}

/// Matrix-free f64 discovery adapter for the positive-definite symmetric `I`
/// metric on the declared finite symmetric polynomial space.
pub struct MkSymmetricIMetricF64<'a> {
    reference: &'a MkSymmetricReference,
    norm_bound: f64,
}

impl<'a> MkSymmetricIMetricF64<'a> {
    pub fn new(reference: &'a MkSymmetricReference) -> Result<Self, MkError> {
        let norm_bound =
            reference.row_sum_norm_bound(|row, column| reference.i_entry(row, column))?;
        if !norm_bound.is_finite() {
            return Err(MkError::InvalidProblem(
                "symmetric I row-sum norm bound is outside the finite f64 range".to_owned(),
            ));
        }
        Ok(Self {
            reference,
            norm_bound,
        })
    }
}

impl LinearOperator<f64> for MkSymmetricIMetricF64<'_> {
    fn dimension(&self) -> usize {
        self.reference.dimension()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        self.reference.apply_i_f64(x, y).map_err(mk_operator_error)
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            "maynard_tao_mk_symmetric_i_metric",
            self.dimension(),
            MatrixStructure::MatrixFree,
            "f64_from_exact_rational_entries",
        );
        metadata.symmetric = true;
        metadata.exact_action = false;
        metadata.tags = vec![
            "mk".to_owned(),
            "fully_symmetric_orbit_sum".to_owned(),
            "positive_definite_metric".to_owned(),
            "discovery_only".to_owned(),
        ];
        metadata
    }

    fn norm_bound(&self) -> Option<f64> {
        Some(self.norm_bound)
    }
}

impl SymmetricOperator<f64> for MkSymmetricIMetricF64<'_> {}
impl PositiveDefiniteMetric<f64> for MkSymmetricIMetricF64<'_> {}

/// Matrix-free MPFR adapter for the symmetric `J` form. Every entry is
/// generated as an exact rational and rounded only when accumulated into the
/// requested MPFR working precision.
pub struct MkSymmetricJOperatorHp<'a> {
    reference: &'a MkSymmetricReference,
    precision_bits: u32,
    norm_bound: Float,
}

impl<'a> MkSymmetricJOperatorHp<'a> {
    pub fn new(reference: &'a MkSymmetricReference, precision_bits: u32) -> Result<Self, MkError> {
        if precision_bits <= 32 {
            return Err(MkError::InvalidProblem(
                "MPFR Mk operators require precision above 32 bits".to_owned(),
            ));
        }
        let exact_norm_bound = reference
            .row_sum_norm_bound_exact(|row, column| reference.j_total_entry(row, column))?;
        let norm_bound = Float::with_val_round(precision_bits, exact_norm_bound, Round::Up).0;
        Ok(Self {
            reference,
            precision_bits,
            norm_bound,
        })
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
}

impl LinearOperator<Float> for MkSymmetricJOperatorHp<'_> {
    fn dimension(&self) -> usize {
        self.reference.dimension()
    }

    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        self.reference
            .apply_j_total_hp(x, y, self.precision_bits)
            .map_err(mk_operator_error)
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            "maynard_tao_mk_symmetric_j_hp",
            self.dimension(),
            MatrixStructure::MatrixFree,
            "rug_mpfr_from_exact_rational_entries",
        );
        metadata.symmetric = true;
        metadata.exact_action = false;
        metadata.tags = vec![
            "mk".to_owned(),
            "fully_symmetric_orbit_sum".to_owned(),
            "hp".to_owned(),
            "exact_entry_stream".to_owned(),
        ];
        metadata
    }

    fn norm_bound(&self) -> Option<Float> {
        Some(self.norm_bound.clone())
    }
}

impl SymmetricOperator<Float> for MkSymmetricJOperatorHp<'_> {}

/// Matrix-free MPFR adapter for the positive-definite symmetric `I` metric.
/// Exact rational entries remain the source for every streamed application.
pub struct MkSymmetricIMetricHp<'a> {
    reference: &'a MkSymmetricReference,
    precision_bits: u32,
    norm_bound: Float,
}

impl<'a> MkSymmetricIMetricHp<'a> {
    pub fn new(reference: &'a MkSymmetricReference, precision_bits: u32) -> Result<Self, MkError> {
        if precision_bits <= 32 {
            return Err(MkError::InvalidProblem(
                "MPFR Mk metrics require precision above 32 bits".to_owned(),
            ));
        }
        let exact_norm_bound =
            reference.row_sum_norm_bound_exact(|row, column| reference.i_entry(row, column))?;
        let norm_bound = Float::with_val_round(precision_bits, exact_norm_bound, Round::Up).0;
        Ok(Self {
            reference,
            precision_bits,
            norm_bound,
        })
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
}

impl LinearOperator<Float> for MkSymmetricIMetricHp<'_> {
    fn dimension(&self) -> usize {
        self.reference.dimension()
    }

    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        self.reference
            .apply_i_hp(x, y, self.precision_bits)
            .map_err(mk_operator_error)
    }

    fn metadata(&self) -> OperatorMetadata {
        let mut metadata = OperatorMetadata::new(
            "maynard_tao_mk_symmetric_i_metric_hp",
            self.dimension(),
            MatrixStructure::MatrixFree,
            "rug_mpfr_from_exact_rational_entries",
        );
        metadata.symmetric = true;
        metadata.exact_action = false;
        metadata.tags = vec![
            "mk".to_owned(),
            "fully_symmetric_orbit_sum".to_owned(),
            "positive_definite_metric".to_owned(),
            "hp".to_owned(),
            "exact_entry_stream".to_owned(),
        ];
        metadata
    }

    fn norm_bound(&self) -> Option<Float> {
        Some(self.norm_bound.clone())
    }
}

impl SymmetricOperator<Float> for MkSymmetricIMetricHp<'_> {}
impl PositiveDefiniteMetric<Float> for MkSymmetricIMetricHp<'_> {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkRayleighCertificate {
    pub k: usize,
    pub degree: usize,
    pub dimension: usize,
    pub search_space: String,
    pub numerator: ExactRationalRecord,
    pub denominator: ExactRationalRecord,
    pub quotient: ExactRationalRecord,
}

pub fn exact_record(value: &Rational) -> ExactRationalRecord {
    ExactRationalRecord {
        numerator: value.numer().to_string(),
        denominator: value.denom().to_string(),
    }
}

/// One exact polynomial vector whose normalized basis element is
/// `coefficients / sqrt(squared_norm)`.  Keeping the square root symbolic
/// preserves exact arithmetic even when the normalization is irrational.
#[derive(Clone, Debug)]
pub struct ExactOrthonormalVector {
    pub coefficients: Vec<Rational>,
    pub squared_norm: Rational,
}

/// An exact Gram-Schmidt simplex-polynomial basis, orthonormal for the
/// Maynard-Tao `I` inner product.  It is intentionally a validation-scale
/// construction: production recurrences may be faster, but must reproduce
/// this basis's exact Gram identities at moderate degree.
#[derive(Clone, Debug)]
pub struct ExactSimplexOrthonormalBasis {
    k: usize,
    degree: usize,
    vectors: Vec<ExactOrthonormalVector>,
}

impl ExactSimplexOrthonormalBasis {
    pub const FAMILY: &'static str = "exact Gram-Schmidt simplex orthonormal basis";

    pub fn new(reference: &MkMonomialReference) -> Result<Self, MkError> {
        let dimension = reference.dimension();
        let gram = reference.dense_i_exact()?;
        let mut vectors: Vec<ExactOrthonormalVector> = Vec::with_capacity(dimension);
        for column in 0..dimension {
            let mut coefficients = vec![Rational::from((0, 1)); dimension];
            coefficients[column] = Rational::from((1, 1));
            for previous in &vectors {
                let projection_numerator =
                    exact_dense_inner_product(&coefficients, &previous.coefficients, &gram)?;
                let projection = projection_numerator / previous.squared_norm.clone();
                for (coefficient, previous_coefficient) in
                    coefficients.iter_mut().zip(&previous.coefficients)
                {
                    let mut correction = projection.clone();
                    correction *= previous_coefficient;
                    *coefficient -= correction;
                }
            }
            let squared_norm = exact_dense_inner_product(&coefficients, &coefficients, &gram)?;
            if squared_norm <= 0 {
                return Err(MkError::InvalidProblem(format!(
                    "simplex Gram-Schmidt produced a nonpositive norm at column {column}"
                )));
            }
            vectors.push(ExactOrthonormalVector {
                coefficients,
                squared_norm,
            });
        }
        let basis = Self {
            k: reference.k(),
            degree: reference.degree(),
            vectors,
        };
        basis.verify_exact(reference)?;
        Ok(basis)
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn dimension(&self) -> usize {
        self.vectors.len()
    }

    pub fn vectors(&self) -> &[ExactOrthonormalVector] {
        &self.vectors
    }

    /// Verify the symbolic normalized Gram matrix exactly: off-diagonal
    /// numerators are zero and each diagonal numerator equals its stored
    /// squared norm, hence division by the two symbolic square roots gives 1.
    pub fn verify_exact(&self, reference: &MkMonomialReference) -> Result<(), MkError> {
        if self.k != reference.k()
            || self.degree != reference.degree()
            || self.dimension() != reference.dimension()
        {
            return Err(MkError::InvalidProblem(
                "orthonormal basis and monomial reference semantics differ".to_owned(),
            ));
        }
        let gram = reference.dense_i_exact()?;
        for row in 0..self.dimension() {
            for column in 0..self.dimension() {
                let inner = exact_dense_inner_product(
                    &self.vectors[row].coefficients,
                    &self.vectors[column].coefficients,
                    &gram,
                )?;
                if row == column {
                    if inner != self.vectors[row].squared_norm {
                        return Err(MkError::InvalidProblem(format!(
                            "orthonormal diagonal identity failed at {row}"
                        )));
                    }
                } else if inner != 0 {
                    return Err(MkError::InvalidProblem(format!(
                        "orthonormal cross-moment ({row}, {column}) is nonzero"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Exact Proriol-Koornwinder-Dubiner simplex basis generated from shifted
/// Jacobi three-term recurrences. Each vector is stored as a rational
/// polynomial divided symbolically by the square root of its exact norm.
#[derive(Clone, Debug)]
pub struct ExactPkdSimplexOrthonormalBasis {
    k: usize,
    degree: usize,
    indices: Vec<MultiIndex>,
    vectors: Vec<ExactOrthonormalVector>,
}

impl ExactPkdSimplexOrthonormalBasis {
    pub const FAMILY: &'static str =
        "exact Proriol-Koornwinder-Dubiner shifted-Jacobi recurrence basis";

    pub fn new(reference: &MkMonomialReference) -> Result<Self, MkError> {
        let positions = reference
            .indices()
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, index)| (index, position))
            .collect::<BTreeMap<_, _>>();
        let mut vectors = Vec::with_capacity(reference.dimension());
        for index in reference.indices() {
            let polynomial = pkd_polynomial(reference.k(), index)?;
            let mut coefficients = vec![Rational::from((0, 1)); reference.dimension()];
            for (monomial, coefficient) in polynomial {
                let position = positions.get(&monomial).ok_or_else(|| {
                    MkError::InvalidProblem(
                        "PKD recurrence produced a monomial outside the finite space".to_owned(),
                    )
                })?;
                coefficients[*position] = coefficient;
            }
            let squared_norm = pkd_squared_norm(reference.k(), index)?;
            vectors.push(ExactOrthonormalVector {
                coefficients,
                squared_norm,
            });
        }
        let basis = Self {
            k: reference.k(),
            degree: reference.degree(),
            indices: reference.indices().to_vec(),
            vectors,
        };
        basis.verify_exact(reference)?;
        Ok(basis)
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn dimension(&self) -> usize {
        self.vectors.len()
    }

    pub fn indices(&self) -> &[MultiIndex] {
        &self.indices
    }

    pub fn vectors(&self) -> &[ExactOrthonormalVector] {
        &self.vectors
    }

    pub fn verify_exact(&self, reference: &MkMonomialReference) -> Result<(), MkError> {
        if self.k != reference.k()
            || self.degree != reference.degree()
            || self.indices != reference.indices()
            || self.dimension() != reference.dimension()
        {
            return Err(MkError::InvalidProblem(
                "PKD basis and monomial reference semantics differ".to_owned(),
            ));
        }
        let gram = reference.dense_i_exact()?;
        for row in 0..self.dimension() {
            for column in 0..self.dimension() {
                let inner = exact_dense_inner_product(
                    &self.vectors[row].coefficients,
                    &self.vectors[column].coefficients,
                    &gram,
                )?;
                if row == column {
                    if inner != self.vectors[row].squared_norm {
                        return Err(MkError::InvalidProblem(format!(
                            "PKD recurrence norm disagrees with exact simplex moments at {row}"
                        )));
                    }
                } else if inner != 0 {
                    return Err(MkError::InvalidProblem(format!(
                        "PKD cross-moment ({row}, {column}) is nonzero"
                    )));
                }
            }
        }
        Ok(())
    }
}

type ExactPolynomial = BTreeMap<MultiIndex, Rational>;

fn exact_polynomial_product(
    left: &ExactPolynomial,
    right: &ExactPolynomial,
    k: usize,
) -> Result<ExactPolynomial, MkError> {
    let mut product = BTreeMap::new();
    for (left_index, left_coefficient) in left {
        for (right_index, right_coefficient) in right {
            if left_index.dimension() != k || right_index.dimension() != k {
                return Err(MkError::InvalidProblem(
                    "PKD polynomial factor has the wrong variable dimension".to_owned(),
                ));
            }
            let mut exponent = Vec::with_capacity(k);
            for coordinate in 0..k {
                exponent.push(
                    left_index.0[coordinate]
                        .checked_add(right_index.0[coordinate])
                        .ok_or_else(|| {
                            MkError::InvalidProblem(
                                "PKD monomial exponent overflowed u32".to_owned(),
                            )
                        })?,
                );
            }
            let mut term = left_coefficient.clone();
            term *= right_coefficient;
            *product
                .entry(MultiIndex(exponent))
                .or_insert_with(|| Rational::from((0, 1))) += term;
        }
    }
    product.retain(|_, coefficient| coefficient != &0);
    Ok(product)
}

fn affine_remainder_power(
    k: usize,
    preceding: usize,
    power: usize,
) -> Result<ExactPolynomial, MkError> {
    let zero_index = MultiIndex(vec![0; k]);
    let mut result = BTreeMap::from([(zero_index.clone(), Rational::from((1, 1)))]);
    if power == 0 {
        return Ok(result);
    }
    let mut remainder = BTreeMap::from([(zero_index, Rational::from((1, 1)))]);
    for variable in 0..preceding {
        let mut exponent = vec![0; k];
        exponent[variable] = 1;
        remainder.insert(MultiIndex(exponent), Rational::from((-1, 1)));
    }
    for _ in 0..power {
        result = exact_polynomial_product(&result, &remainder, k)?;
    }
    Ok(result)
}

fn rational_from_i128(value: i128) -> Rational {
    Rational::from((Integer::from(value), Integer::from(1)))
}

fn shifted_jacobi_alpha_zero_coefficients(
    degree: usize,
    alpha: usize,
) -> Result<Vec<Rational>, MkError> {
    let mut previous_previous = vec![Rational::from((1, 1))];
    if degree == 0 {
        return Ok(previous_previous);
    }
    // P_1^(alpha,0)(z) = (alpha + (alpha+2) z) / 2.
    let mut previous = vec![
        Rational::from((Integer::from(alpha), Integer::from(2))),
        Rational::from((Integer::from(alpha + 2), Integer::from(2))),
    ];
    for n in 2..=degree {
        let n_i = i128::try_from(n)
            .map_err(|_| MkError::InvalidProblem("Jacobi degree does not fit i128".to_owned()))?;
        let alpha_i = i128::try_from(alpha)
            .map_err(|_| MkError::InvalidProblem("Jacobi alpha does not fit i128".to_owned()))?;
        let twice_n_alpha = 2 * n_i + alpha_i;
        let denominator = 2 * n_i * (n_i + alpha_i) * (twice_n_alpha - 2);
        let outer = twice_n_alpha - 1;
        let z_factor = twice_n_alpha * (twice_n_alpha - 2);
        let constant_factor = alpha_i * alpha_i;
        let prior_factor = 2 * (n_i + alpha_i - 1) * (n_i - 1) * twice_n_alpha;
        if denominator == 0 {
            return Err(MkError::InvalidProblem(
                "Jacobi recurrence denominator vanished".to_owned(),
            ));
        }
        let mut current = vec![Rational::from((0, 1)); n + 1];
        for (power, coefficient) in previous.iter().enumerate() {
            let mut constant = coefficient.clone();
            constant *= rational_from_i128(outer * constant_factor);
            current[power] += constant;
            let mut shifted = coefficient.clone();
            shifted *= rational_from_i128(outer * z_factor);
            current[power + 1] += shifted;
        }
        for (power, coefficient) in previous_previous.iter().enumerate() {
            let mut correction = coefficient.clone();
            correction *= rational_from_i128(prior_factor);
            current[power] -= correction;
        }
        let denominator = rational_from_i128(denominator);
        for coefficient in &mut current {
            *coefficient /= &denominator;
        }
        previous_previous = previous;
        previous = current;
    }

    // Substitute z=2t-1 exactly, returning ascending coefficients in t.
    let mut shifted = vec![Rational::from((0, 1)); degree + 1];
    for (z_power, coefficient) in previous.iter().enumerate() {
        for (t_power, binomial) in integer_binomial_row(z_power).into_iter().enumerate() {
            let mut term = coefficient.clone();
            term *= Rational::from((binomial, Integer::from(1)));
            term *= Rational::from((Integer::from(1) << t_power, Integer::from(1)));
            if (z_power - t_power) % 2 == 1 {
                term = -term;
            }
            shifted[t_power] += term;
        }
    }
    Ok(shifted)
}

fn integer_binomial_row(n: usize) -> Vec<Integer> {
    let mut row = Vec::with_capacity(n + 1);
    let mut coefficient = Integer::from(1);
    for k in 0..=n {
        if k > 0 {
            coefficient *= n + 1 - k;
            coefficient /= k;
        }
        row.push(coefficient.clone());
    }
    row
}

fn pkd_polynomial(k: usize, index: &MultiIndex) -> Result<ExactPolynomial, MkError> {
    if index.dimension() != k {
        return Err(MkError::InvalidProblem(
            "PKD index has the wrong variable dimension".to_owned(),
        ));
    }
    let mut polynomial = BTreeMap::from([(MultiIndex(vec![0; k]), Rational::from((1, 1)))]);
    for variable in 0..k {
        let degree = index.0[variable] as usize;
        let later_degree = index.0[variable + 1..]
            .iter()
            .map(|&value| value as usize)
            .sum::<usize>();
        let alpha = 2 * later_degree + k - variable - 1;
        let jacobi = shifted_jacobi_alpha_zero_coefficients(degree, alpha)?;
        let mut factor = BTreeMap::<MultiIndex, Rational>::new();
        for (monomial_degree, jacobi_coefficient) in jacobi.into_iter().enumerate() {
            let mut term = affine_remainder_power(k, variable, degree - monomial_degree)?;
            if monomial_degree > 0 {
                let mut monomial = vec![0; k];
                monomial[variable] = u32::try_from(monomial_degree).map_err(|_| {
                    MkError::InvalidProblem("PKD monomial degree does not fit u32".to_owned())
                })?;
                term = exact_polynomial_product(
                    &term,
                    &BTreeMap::from([(MultiIndex(monomial), Rational::from((1, 1)))]),
                    k,
                )?;
            }
            for (term_index, mut coefficient) in term {
                coefficient *= &jacobi_coefficient;
                *factor
                    .entry(term_index)
                    .or_insert_with(|| Rational::from((0, 1))) += coefficient;
            }
        }
        factor.retain(|_, coefficient| coefficient != &0);
        polynomial = exact_polynomial_product(&polynomial, &factor, k)?;
    }
    Ok(polynomial)
}

fn pkd_squared_norm(k: usize, index: &MultiIndex) -> Result<Rational, MkError> {
    if index.dimension() != k {
        return Err(MkError::InvalidProblem(
            "PKD norm index has the wrong variable dimension".to_owned(),
        ));
    }
    let mut norm = Rational::from((1, 1));
    for variable in 0..k {
        let degree = index.0[variable] as usize;
        let later_degree = index.0[variable + 1..]
            .iter()
            .map(|&value| value as usize)
            .sum::<usize>();
        let alpha = 2 * later_degree + k - variable - 1;
        norm /= Integer::from(2 * degree + alpha + 1);
    }
    Ok(norm)
}

fn exact_dense_inner_product(
    left: &[Rational],
    right: &[Rational],
    matrix: &[Rational],
) -> Result<Rational, MkError> {
    if left.len() != right.len() || matrix.len() != left.len().saturating_mul(left.len()) {
        return Err(MkError::InvalidProblem(
            "exact inner-product dimensions are inconsistent".to_owned(),
        ));
    }
    let dimension = left.len();
    let mut total = Rational::from((0, 1));
    for row in 0..dimension {
        for column in 0..dimension {
            let mut term = left[row].clone();
            term *= &matrix[row * dimension + column];
            term *= &right[column];
            total += term;
        }
    }
    Ok(total)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MkAssuranceMode {
    Exploratory,
    CrossChecked,
    Certified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MkApproximationMetric {
    EuclideanOperatorNorm,
    IMetricOperatorNorm,
}

/// A portable bound for an approximate `M_k` operator representation.
/// A certificate applies only to the exact finite semantics named here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkApproximationCertificate {
    pub construction: String,
    pub k: usize,
    pub degree: usize,
    pub metric: MkApproximationMetric,
    pub rigorous_operator_error_bound: ExactRationalRecord,
    pub rounding_contribution: ExactRationalRecord,
    pub parameter_range: String,
    pub validation_digest: String,
}

impl MkApproximationCertificate {
    pub fn validate_for(&self, k: usize, degree: usize) -> Result<(), MkError> {
        if self.k != k || self.degree != degree {
            return Err(MkError::InvalidProblem(
                "approximation certificate does not match k and degree".to_owned(),
            ));
        }
        if self.construction.trim().is_empty()
            || self.parameter_range.trim().is_empty()
            || self.validation_digest.trim().is_empty()
        {
            return Err(MkError::InvalidProblem(
                "approximation certificate metadata must be nonempty".to_owned(),
            ));
        }
        let error = rational_record_value(&self.rigorous_operator_error_bound)?;
        let rounding = rational_record_value(&self.rounding_contribution)?;
        if error < 0 || rounding < 0 || rounding > error {
            return Err(MkError::InvalidProblem(
                "approximation bounds must satisfy 0 <= rounding <= total error".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MkOperatorAcceleration {
    ExactStreamed,
    ExactSumFactorized,
    DegreeDifferenceBand {
        half_width: usize,
        certificate: Option<MkApproximationCertificate>,
    },
    LowRank {
        rank: usize,
        certificate: Option<MkApproximationCertificate>,
    },
}

/// Fail-closed assurance preflight for structured `M_k` actions.  In
/// particular, factorial decay by itself is never accepted as a bound for a
/// `|d-d'|` band.
pub fn validate_mk_acceleration(
    acceleration: &MkOperatorAcceleration,
    assurance: MkAssuranceMode,
    k: usize,
    degree: usize,
) -> Result<(), MkError> {
    match acceleration {
        MkOperatorAcceleration::ExactStreamed | MkOperatorAcceleration::ExactSumFactorized => {
            Ok(())
        }
        MkOperatorAcceleration::DegreeDifferenceBand {
            half_width,
            certificate,
        } => {
            if *half_width > degree {
                return Err(MkError::InvalidProblem(
                    "degree-difference half-width exceeds the finite degree".to_owned(),
                ));
            }
            validate_optional_approximation_certificate(
                certificate.as_ref(),
                assurance,
                k,
                degree,
                "degree-difference band",
            )
        }
        MkOperatorAcceleration::LowRank { rank, certificate } => {
            if *rank == 0 {
                return Err(MkError::InvalidProblem(
                    "low-rank acceleration rank must be positive".to_owned(),
                ));
            }
            validate_optional_approximation_certificate(
                certificate.as_ref(),
                assurance,
                k,
                degree,
                "low-rank approximation",
            )
        }
    }
}

fn validate_optional_approximation_certificate(
    certificate: Option<&MkApproximationCertificate>,
    assurance: MkAssuranceMode,
    k: usize,
    degree: usize,
    description: &str,
) -> Result<(), MkError> {
    match certificate {
        Some(certificate) => certificate.validate_for(k, degree),
        None if assurance == MkAssuranceMode::Certified => Err(MkError::InvalidProblem(format!(
            "Certified mode rejects an unbounded {description}"
        ))),
        None => Ok(()),
    }
}

fn rational_record_value(record: &ExactRationalRecord) -> Result<Rational, MkError> {
    let numerator = record
        .numerator
        .parse::<Integer>()
        .map_err(|_| MkError::InvalidProblem("invalid approximation-bound numerator".to_owned()))?;
    let denominator = record.denominator.parse::<Integer>().map_err(|_| {
        MkError::InvalidProblem("invalid approximation-bound denominator".to_owned())
    })?;
    if denominator <= 0 {
        return Err(MkError::InvalidProblem(
            "approximation-bound denominator must be positive".to_owned(),
        ));
    }
    Ok(rational(numerator, denominator))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MkSymmetricForm {
    IMetric,
    JTotal,
}

/// A reusable exact degree-band action with a rigorously constructed error
/// certificate. The action skips entries outside the declared band; its
/// Euclidean operator-norm error is bounded by the exact maximum absolute row
/// sum of the omitted symmetric matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkCertifiedDegreeBandAction {
    pub k: usize,
    pub degree: usize,
    pub form: MkSymmetricForm,
    pub half_width: usize,
    pub retained_entries: u64,
    pub omitted_entries: u64,
    pub certificate: MkApproximationCertificate,
}

impl MkCertifiedDegreeBandAction {
    pub fn construct(
        reference: &MkSymmetricReference,
        form: MkSymmetricForm,
        half_width: usize,
    ) -> Result<Self, MkError> {
        if half_width > reference.degree() {
            return Err(MkError::InvalidProblem(
                "degree-band half-width exceeds the finite degree".to_owned(),
            ));
        }
        let mut maximum_omitted_row_sum = Rational::from((0, 1));
        let mut retained_entries = 0_u64;
        let mut omitted_entries = 0_u64;
        let mut hasher = Sha256::new();
        digest_field(&mut hasher, b"mk-certified-degree-band-v1");
        digest_field(&mut hasher, &reference.k().to_le_bytes());
        digest_field(&mut hasher, &reference.degree().to_le_bytes());
        digest_field(&mut hasher, &half_width.to_le_bytes());
        digest_field(
            &mut hasher,
            match form {
                MkSymmetricForm::IMetric => b"i_metric",
                MkSymmetricForm::JTotal => b"j_total",
            },
        );
        for row in 0..reference.dimension() {
            let row_degree = reference.orbits()[row].partition.total_degree();
            let mut omitted_row_sum = Rational::from((0, 1));
            for column in 0..reference.dimension() {
                let column_degree = reference.orbits()[column].partition.total_degree();
                let entry = symmetric_form_entry(reference, form, row, column)?;
                if row_degree.abs_diff(column_degree) <= half_width {
                    retained_entries = retained_entries.saturating_add(1);
                } else {
                    omitted_entries = omitted_entries.saturating_add(1);
                    let absolute = rational_absolute(&entry);
                    omitted_row_sum += &absolute;
                    digest_field(&mut hasher, &row.to_le_bytes());
                    digest_field(&mut hasher, &column.to_le_bytes());
                    digest_field(&mut hasher, entry.numer().to_string().as_bytes());
                    digest_field(&mut hasher, entry.denom().to_string().as_bytes());
                }
            }
            if omitted_row_sum > maximum_omitted_row_sum {
                maximum_omitted_row_sum = omitted_row_sum;
            }
        }
        let digest = format!("sha256:{:x}", hasher.finalize());
        let certificate = MkApproximationCertificate {
            construction:
                "exact omitted-entry maximum absolute row-sum bound for a symmetric degree band"
                    .to_owned(),
            k: reference.k(),
            degree: reference.degree(),
            metric: MkApproximationMetric::EuclideanOperatorNorm,
            rigorous_operator_error_bound: exact_record(&maximum_omitted_row_sum),
            rounding_contribution: exact_record(&Rational::from((0, 1))),
            parameter_range: format!(
                "{} form; retain |total_degree(row)-total_degree(column)| <= {half_width}",
                match form {
                    MkSymmetricForm::IMetric => "I",
                    MkSymmetricForm::JTotal => "J_total",
                }
            ),
            validation_digest: digest,
        };
        certificate.validate_for(reference.k(), reference.degree())?;
        Ok(Self {
            k: reference.k(),
            degree: reference.degree(),
            form,
            half_width,
            retained_entries,
            omitted_entries,
            certificate,
        })
    }

    pub fn verify<'a>(
        &'a self,
        reference: &'a MkSymmetricReference,
    ) -> Result<VerifiedMkDegreeBandAction<'a>, MkError> {
        let replay = Self::construct(reference, self.form, self.half_width)?;
        if &replay != self {
            return Err(MkError::InvalidProblem(
                "degree-band action does not match its exact omitted-entry replay".to_owned(),
            ));
        }
        let mut retained_rows = Vec::with_capacity(reference.dimension());
        for row in 0..reference.dimension() {
            let row_degree = reference.orbits()[row].partition.total_degree();
            let mut retained = Vec::new();
            for column in 0..reference.dimension() {
                let column_degree = reference.orbits()[column].partition.total_degree();
                if row_degree.abs_diff(column_degree) <= self.half_width {
                    retained.push((
                        column,
                        symmetric_form_entry(reference, self.form, row, column)?,
                    ));
                }
            }
            retained_rows.push(retained);
        }
        Ok(VerifiedMkDegreeBandAction {
            reference,
            retained_rows,
        })
    }

    pub fn acceleration(&self) -> MkOperatorAcceleration {
        MkOperatorAcceleration::DegreeDifferenceBand {
            half_width: self.half_width,
            certificate: Some(self.certificate.clone()),
        }
    }
}

/// Runtime capability produced only by exact certificate replay. Repeated
/// operator applications use the retained band directly and do not rebuild
/// the O(N^2) omitted-entry proof on every iteration.
pub struct VerifiedMkDegreeBandAction<'a> {
    reference: &'a MkSymmetricReference,
    retained_rows: Vec<Vec<(usize, Rational)>>,
}

impl VerifiedMkDegreeBandAction<'_> {
    pub fn apply_exact(&self, input: &[Rational]) -> Result<Vec<Rational>, MkError> {
        if input.len() != self.reference.dimension() {
            return Err(MkError::DimensionMismatch {
                expected: self.reference.dimension(),
                actual: input.len(),
            });
        }
        let mut output = vec![Rational::from((0, 1)); self.reference.dimension()];
        for (output_entry, retained) in output.iter_mut().zip(&self.retained_rows) {
            for (column, entry) in retained {
                let mut term = entry.clone();
                term *= &input[*column];
                *output_entry += term;
            }
        }
        Ok(output)
    }
}

fn symmetric_form_entry(
    reference: &MkSymmetricReference,
    form: MkSymmetricForm,
    row: usize,
    column: usize,
) -> Result<Rational, MkError> {
    match form {
        MkSymmetricForm::IMetric => reference.i_entry(row, column),
        MkSymmetricForm::JTotal => reference.j_total_entry(row, column),
    }
}

fn rational_absolute(value: &Rational) -> Rational {
    if value < &0 {
        -value.clone()
    } else {
        value.clone()
    }
}

fn digest_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(value.len().to_le_bytes());
    hasher.update(value);
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveEnrichmentRule {
    CompleteDegreeShell,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveSpacePolicy {
    pub k: usize,
    pub initial_degree: usize,
    pub maximum_degree: usize,
    pub maximum_generations: usize,
    pub enrichment_rule: AdaptiveEnrichmentRule,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveSpaceGeneration {
    pub generation: usize,
    pub parent_generation: Option<usize>,
    pub degree: usize,
    pub candidates_considered: Vec<IntegerPartition>,
    pub accepted_block: Vec<IntegerPartition>,
    pub realized_basis: Vec<IntegerPartition>,
    pub stopping_reason: Option<String>,
}

/// Build a deterministic nested sequence of symmetric partition spaces.  Each
/// generation adds one complete degree shell, so prolongation is exact and a
/// lower-bound candidate from any parent remains valid in every descendant.
pub fn build_adaptive_symmetric_spaces(
    policy: &AdaptiveSpacePolicy,
) -> Result<Vec<AdaptiveSpaceGeneration>, MkError> {
    if policy.k == 0 {
        return Err(MkError::InvalidProblem("k must be positive".to_owned()));
    }
    if policy.initial_degree > policy.maximum_degree || policy.maximum_generations == 0 {
        return Err(MkError::InvalidProblem(
            "adaptive-space degree range or generation budget is invalid".to_owned(),
        ));
    }
    let mut history = Vec::new();
    let initial_basis = enumerate_integer_partitions(policy.k, policy.initial_degree)?;
    history.push(AdaptiveSpaceGeneration {
        generation: 0,
        parent_generation: None,
        degree: policy.initial_degree,
        candidates_considered: initial_basis.clone(),
        accepted_block: initial_basis.clone(),
        realized_basis: initial_basis,
        stopping_reason: None,
    });

    for degree in policy.initial_degree.saturating_add(1)..=policy.maximum_degree {
        if history.len() >= policy.maximum_generations {
            if let Some(last) = history.last_mut() {
                last.stopping_reason = Some("maximum_generations".to_owned());
            }
            break;
        }
        let complete = enumerate_integer_partitions(policy.k, degree)?;
        let accepted_block = complete
            .iter()
            .filter(|partition| partition.total_degree() == degree)
            .cloned()
            .collect::<Vec<_>>();
        let generation = history.len();
        history.push(AdaptiveSpaceGeneration {
            generation,
            parent_generation: Some(generation - 1),
            degree,
            candidates_considered: accepted_block.clone(),
            accepted_block,
            realized_basis: complete,
            stopping_reason: None,
        });
    }
    if let Some(last) = history.last_mut() {
        if last.stopping_reason.is_none() {
            last.stopping_reason = Some(
                if last.degree == policy.maximum_degree {
                    "maximum_degree"
                } else {
                    "generation_budget"
                }
                .to_owned(),
            );
        }
    }
    Ok(history)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MkWarmStart {
    pub coefficients: Vec<f64>,
    pub source_generation: Option<usize>,
    pub target_generation: usize,
    pub exploratory_seed: Option<u64>,
    pub seed_generator: xc_core::SeedGeneratorKind,
    pub certification_dependency: bool,
}

/// Deterministically prolong a vector between nested canonical partition
/// bases.  New coordinates are exactly zero.
pub fn prolong_symmetric_warm_start(
    parent_basis: &[IntegerPartition],
    parent_coefficients: &[f64],
    child_basis: &[IntegerPartition],
    source_generation: usize,
    target_generation: usize,
) -> Result<MkWarmStart, MkError> {
    if parent_basis.len() != parent_coefficients.len() {
        return Err(MkError::DimensionMismatch {
            expected: parent_basis.len(),
            actual: parent_coefficients.len(),
        });
    }
    let child_positions = child_basis
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, partition)| (partition, index))
        .collect::<BTreeMap<_, _>>();
    let mut coefficients = vec![0.0; child_basis.len()];
    for (partition, coefficient) in parent_basis.iter().zip(parent_coefficients) {
        let position = child_positions.get(partition).ok_or_else(|| {
            MkError::InvalidProblem("warm-start target is not a nested basis".to_owned())
        })?;
        coefficients[*position] = *coefficient;
    }
    Ok(MkWarmStart {
        coefficients,
        source_generation: Some(source_generation),
        target_generation,
        exploratory_seed: None,
        seed_generator: xc_core::SeedGeneratorKind::CachedHp,
        certification_dependency: false,
    })
}

/// Produce a reproducible exploratory seed vector.  The record is explicitly
/// excluded from certification dependencies; exact or interval verification
/// consumes only the final submitted candidate.
pub fn exploratory_random_warm_start(
    dimension: usize,
    seed: u64,
    solver_config: &xc_core::SolverConfig,
) -> Result<MkWarmStart, MkError> {
    solver_config
        .authorize_seed_generator(xc_core::SeedGeneratorKind::Randomized)
        .map_err(|error| MkError::InvalidProblem(error.to_string()))?;
    if dimension == 0 {
        return Err(MkError::InvalidProblem(
            "warm-start dimension must be positive".to_owned(),
        ));
    }
    let mut state = seed.max(1);
    let mut coefficients = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state as f64) / (u64::MAX as f64);
        coefficients.push(2.0 * unit - 1.0);
    }
    let norm = coefficients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(MkError::InvalidProblem(
            "randomized warm start produced an invalid norm".to_owned(),
        ));
    }
    for coefficient in &mut coefficients {
        *coefficient /= norm;
    }
    Ok(MkWarmStart {
        coefficients,
        source_generation: None,
        target_generation: 0,
        exploratory_seed: Some(seed),
        seed_generator: xc_core::SeedGeneratorKind::Randomized,
        certification_dependency: false,
    })
}

/// Exact isotypic sectors for the permutation action on polynomial variables.
/// The named variants are stable conveniences for the three common shapes;
/// `Partition` accepts every other irreducible `S_k` shape.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MkPermutationSector {
    Trivial,
    Standard,
    Alternating,
    Partition(IntegerPartition),
}

impl MkPermutationSector {
    fn partition(&self, k: usize) -> Result<IntegerPartition, MkError> {
        if k == 0 || k > 8 {
            return Err(MkError::InvalidProblem(
                "exact sector projectors require 1 <= k <= 8".to_owned(),
            ));
        }
        let partition = match self {
            Self::Trivial => IntegerPartition(vec![u32::try_from(k).map_err(|_| {
                MkError::InvalidProblem("k does not fit the partition schema".to_owned())
            })?]),
            Self::Standard if k >= 3 => IntegerPartition(vec![
                u32::try_from(k - 1).map_err(|_| {
                    MkError::InvalidProblem("k does not fit the partition schema".to_owned())
                })?,
                1,
            ]),
            Self::Standard => {
                return Err(MkError::InvalidProblem(
                    "the standard sector is requested separately only for k >= 3".to_owned(),
                ));
            }
            Self::Alternating if k >= 2 => IntegerPartition(vec![1; k]),
            Self::Alternating => {
                return Err(MkError::InvalidProblem(
                    "the alternating sector is nontrivial only for k >= 2".to_owned(),
                ));
            }
            Self::Partition(partition) => partition.clone(),
        };
        partition.validate(k)?;
        if partition.total_degree() != k {
            return Err(MkError::InvalidProblem(format!(
                "isotypic partition must have total size k={k}"
            )));
        }
        Ok(partition)
    }
}

fn representation_dimension(partition: &IntegerPartition) -> Result<i64, MkError> {
    let k = partition.total_degree();
    partition.validate(k)?;
    let numerator = (2..=k).try_fold(1_u128, |product, factor| {
        product.checked_mul(factor as u128).ok_or_else(|| {
            MkError::InvalidProblem("irreducible representation dimension overflows".to_owned())
        })
    })?;
    let mut denominator = 1_u128;
    for (row, &row_length) in partition.0.iter().enumerate() {
        for column in 0..row_length as usize {
            let below = partition
                .0
                .iter()
                .skip(row + 1)
                .filter(|&&length| length as usize > column)
                .count();
            let hook = row_length as usize - column + below;
            denominator = denominator.checked_mul(hook as u128).ok_or_else(|| {
                MkError::InvalidProblem(
                    "irreducible representation hook product overflows".to_owned(),
                )
            })?;
        }
    }
    if numerator % denominator != 0 {
        return Err(MkError::InvalidProblem(
            "hook-length calculation lost integrality".to_owned(),
        ));
    }
    i64::try_from(numerator / denominator).map_err(|_| {
        MkError::InvalidProblem("irreducible representation dimension does not fit i64".to_owned())
    })
}

fn permutation_cycle_type(permutation: &[usize]) -> Result<Vec<usize>, MkError> {
    validate_permutation(permutation, permutation.len())?;
    let mut visited = vec![false; permutation.len()];
    let mut cycles = Vec::new();
    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut length = 0;
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            length += 1;
            current = permutation[current];
        }
        cycles.push(length);
    }
    cycles.sort_unstable_by(|left, right| right.cmp(left));
    Ok(cycles)
}

/// Frobenius character formula, evaluated as a bounded sparse coefficient
/// extraction from `Vandermonde(x) * product(power_sum_cycle(x))`.
fn irreducible_character(
    partition: &IntegerPartition,
    cycle_type: &[usize],
) -> Result<i64, MkError> {
    let k = partition.total_degree();
    partition.validate(k)?;
    if k == 0 || k > 8 || cycle_type.contains(&0) || cycle_type.iter().sum::<usize>() != k {
        return Err(MkError::InvalidProblem(
            "character evaluation requires a nonempty size-matched partition and cycle type with k <= 8"
                .to_owned(),
        ));
    }
    let variables = partition.length();
    let target = partition
        .0
        .iter()
        .enumerate()
        .map(|(index, &part)| part as usize + variables - 1 - index)
        .collect::<Vec<_>>();
    let mut vandermonde_indices = (0..variables).collect::<Vec<_>>();
    let mut states = BTreeMap::<Vec<usize>, i128>::new();
    loop {
        let exponents = vandermonde_indices
            .iter()
            .map(|&column| variables - 1 - column)
            .collect::<Vec<_>>();
        states.insert(
            exponents,
            i128::from(permutation_sign(&vandermonde_indices)),
        );
        if !next_permutation_usize(&mut vandermonde_indices) {
            break;
        }
    }
    for &cycle_length in cycle_type {
        let mut next = BTreeMap::<Vec<usize>, i128>::new();
        for (exponents, coefficient) in states {
            for variable in 0..variables {
                let Some(updated) = exponents[variable].checked_add(cycle_length) else {
                    continue;
                };
                if updated > target[variable] {
                    continue;
                }
                let mut image = exponents.clone();
                image[variable] = updated;
                *next.entry(image).or_insert(0) += coefficient;
            }
        }
        states = next;
    }
    let character = states.get(&target).copied().unwrap_or(0);
    i64::try_from(character)
        .map_err(|_| MkError::InvalidProblem("irreducible character does not fit i64".to_owned()))
}

/// An exact central character projector onto a declared `S_k` isotypic
/// component of the complete finite monomial space.
#[derive(Clone, Debug)]
pub struct MkSectorProjector {
    sector: MkPermutationSector,
    k: usize,
    degree: usize,
    matrix: Vec<Rational>,
}

impl MkSectorProjector {
    pub fn new(
        reference: &MkMonomialReference,
        sector: MkPermutationSector,
    ) -> Result<Self, MkError> {
        let partition = sector.partition(reference.k())?;
        let permutations = enumerate_permutations(reference.k())?;
        let group_order = permutations.len();
        let representation_dimension = representation_dimension(&partition)?;
        let mut characters = BTreeMap::<Vec<usize>, i64>::new();
        let positions = reference
            .indices()
            .iter()
            .cloned()
            .enumerate()
            .map(|(position, index)| (index, position))
            .collect::<BTreeMap<_, _>>();
        let dimension = reference.dimension();
        let mut matrix = vec![Rational::from((0, 1)); dimension * dimension];
        for permutation in &permutations {
            let cycle_type = permutation_cycle_type(permutation)?;
            let character = if let Some(character) = characters.get(&cycle_type) {
                *character
            } else {
                let character = irreducible_character(&partition, &cycle_type)?;
                characters.insert(cycle_type, character);
                character
            };
            if character == 0 {
                continue;
            }
            let weight = Rational::from((
                Integer::from(representation_dimension * character),
                Integer::from(group_order),
            ));
            for (column, index) in reference.indices().iter().enumerate() {
                let image = permute_multi_index(index, permutation)?;
                let row = positions.get(&image).ok_or_else(|| {
                    MkError::InvalidProblem(
                        "permutation image is absent from the complete monomial space".to_owned(),
                    )
                })?;
                matrix[*row * dimension + column] += &weight;
            }
        }
        let projector = Self {
            sector,
            k: reference.k(),
            degree: reference.degree(),
            matrix,
        };
        projector.verify_exact()?;
        Ok(projector)
    }

    pub fn sector(&self) -> MkPermutationSector {
        self.sector.clone()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn ambient_dimension(&self) -> usize {
        integer_square_dimension(self.matrix.len())
    }

    pub fn matrix(&self) -> &[Rational] {
        &self.matrix
    }

    pub fn sector_dimension(&self) -> Result<usize, MkError> {
        let dimension = self.ambient_dimension();
        let mut trace = Rational::from((0, 1));
        for index in 0..dimension {
            trace += &self.matrix[index * dimension + index];
        }
        if trace < 0 || trace.denom() != &Integer::from(1) {
            return Err(MkError::InvalidProblem(
                "exact sector-projector trace is not a nonnegative integer".to_owned(),
            ));
        }
        trace.numer().to_usize().ok_or_else(|| {
            MkError::InvalidProblem("sector dimension does not fit usize".to_owned())
        })
    }

    pub fn apply_exact(&self, input: &[Rational]) -> Result<Vec<Rational>, MkError> {
        let dimension = self.ambient_dimension();
        if input.len() != dimension {
            return Err(MkError::DimensionMismatch {
                expected: dimension,
                actual: input.len(),
            });
        }
        let mut output = vec![Rational::from((0, 1)); dimension];
        for (row, output_entry) in output.iter_mut().enumerate() {
            for (column, input_entry) in input.iter().enumerate() {
                let mut term = self.matrix[row * dimension + column].clone();
                term *= input_entry;
                *output_entry += term;
            }
        }
        Ok(output)
    }

    pub fn verify_exact(&self) -> Result<(), MkError> {
        let dimension = self.ambient_dimension();
        for row in 0..dimension {
            for column in 0..dimension {
                if self.matrix[row * dimension + column] != self.matrix[column * dimension + row] {
                    return Err(MkError::InvalidProblem(format!(
                        "sector projector is not self-adjoint at ({row}, {column})"
                    )));
                }
                let mut squared = Rational::from((0, 1));
                for inner in 0..dimension {
                    let mut term = self.matrix[row * dimension + inner].clone();
                    term *= &self.matrix[inner * dimension + column];
                    squared += term;
                }
                if squared != self.matrix[row * dimension + column] {
                    return Err(MkError::InvalidProblem(format!(
                        "sector projector is not idempotent at ({row}, {column})"
                    )));
                }
            }
        }
        let _ = self.sector_dimension()?;
        Ok(())
    }

    pub fn is_orthogonal_to(&self, other: &Self) -> Result<bool, MkError> {
        if self.k != other.k
            || self.degree != other.degree
            || self.matrix.len() != other.matrix.len()
        {
            return Err(MkError::InvalidProblem(
                "sector projectors have incompatible finite semantics".to_owned(),
            ));
        }
        let dimension = self.ambient_dimension();
        for row in 0..dimension {
            for column in 0..dimension {
                let mut product = Rational::from((0, 1));
                for inner in 0..dimension {
                    let mut term = self.matrix[row * dimension + inner].clone();
                    term *= &other.matrix[inner * dimension + column];
                    product += term;
                }
                if product != 0 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Verify that the central projector commutes with an explicit group
    /// action.  This is the defining invariance check for an isotypic sector.
    pub fn verify_commutes_with(
        &self,
        reference: &MkMonomialReference,
        permutation: &[usize],
    ) -> Result<(), MkError> {
        if reference.k() != self.k || reference.degree() != self.degree {
            return Err(MkError::InvalidProblem(
                "group-action check uses different finite semantics".to_owned(),
            ));
        }
        validate_permutation(permutation, self.k)?;
        for column in 0..self.ambient_dimension() {
            let mut basis = vec![Rational::from((0, 1)); self.ambient_dimension()];
            basis[column] = Rational::from((1, 1));
            let projected_then_permuted =
                permute_coefficients(reference, &self.apply_exact(&basis)?, permutation)?;
            let permuted_then_projected =
                self.apply_exact(&permute_coefficients(reference, &basis, permutation)?)?;
            if projected_then_permuted != permuted_then_projected {
                return Err(MkError::InvalidProblem(format!(
                    "sector projector does not commute with the group action on column {column}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkSectorDimension {
    pub sector: MkPermutationSector,
    pub dimension: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MkSectorCoverageReport {
    pub k: usize,
    pub degree: usize,
    pub ambient_dimension: usize,
    pub sector_dimensions: Vec<MkSectorDimension>,
    pub pairwise_orthogonal: bool,
    pub reconstructs_full_space: bool,
    pub claim_scope: String,
}

impl MkSectorCoverageReport {
    pub fn sector_dimension(&self, sector: &MkPermutationSector) -> Option<usize> {
        self.sector_dimensions
            .iter()
            .find(|entry| &entry.sector == sector)
            .map(|entry| entry.dimension)
    }
}

/// Construct every exact partition-labelled central sector and verify that
/// they exhaust the complete monomial representation.
pub fn exact_sector_coverage(
    reference: &MkMonomialReference,
) -> Result<(Vec<MkSectorProjector>, MkSectorCoverageReport), MkError> {
    if reference.k() > 8 {
        return Err(MkError::InvalidProblem(
            "exact sector coverage is limited to k <= 8".to_owned(),
        ));
    }
    let sectors = enumerate_integer_partitions(reference.k(), reference.k())?
        .into_iter()
        .filter(|partition| partition.total_degree() == reference.k())
        .map(|partition| {
            if partition.0 == vec![reference.k() as u32] {
                MkPermutationSector::Trivial
            } else if reference.k() >= 3 && partition.0 == vec![(reference.k() - 1) as u32, 1] {
                MkPermutationSector::Standard
            } else if reference.k() >= 2 && partition.0 == vec![1; reference.k()] {
                MkPermutationSector::Alternating
            } else {
                MkPermutationSector::Partition(partition)
            }
        })
        .collect::<Vec<_>>();
    let projectors = sectors
        .into_iter()
        .map(|sector| MkSectorProjector::new(reference, sector))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pairwise_orthogonal = true;
    for left in 0..projectors.len() {
        for right in left + 1..projectors.len() {
            pairwise_orthogonal &= projectors[left].is_orthogonal_to(&projectors[right])?;
        }
    }
    let ambient_dimension = reference.dimension();
    let mut sum = vec![Rational::from((0, 1)); ambient_dimension * ambient_dimension];
    let mut sector_dimensions = Vec::with_capacity(projectors.len());
    for projector in &projectors {
        sector_dimensions.push(MkSectorDimension {
            sector: projector.sector(),
            dimension: projector.sector_dimension()?,
        });
        for (entry, value) in sum.iter_mut().zip(projector.matrix()) {
            *entry += value;
        }
    }
    let reconstructs_full_space = pairwise_orthogonal
        && (0..ambient_dimension).all(|row| {
            (0..ambient_dimension).all(|column| {
                let expected = if row == column {
                    Rational::from((1, 1))
                } else {
                    Rational::from((0, 1))
                };
                sum[row * ambient_dimension + column] == expected
            })
        });
    let claim_scope = if reconstructs_full_space {
        "all partition-labelled irreducible permutation-symmetry sectors in this finite polynomial representation"
    } else {
        "partition-labelled sector coverage failed exact full-space reconstruction"
    }
    .to_owned();
    Ok((
        projectors,
        MkSectorCoverageReport {
            k: reference.k(),
            degree: reference.degree(),
            ambient_dimension,
            sector_dimensions,
            pairwise_orthogonal,
            reconstructs_full_space,
            claim_scope,
        },
    ))
}

fn enumerate_permutations(k: usize) -> Result<Vec<Vec<usize>>, MkError> {
    let mut group_order = 1usize;
    for factor in 2..=k {
        group_order = group_order.checked_mul(factor).ok_or_else(|| {
            MkError::InvalidProblem("permutation group order overflows usize".to_owned())
        })?;
    }
    if group_order > 40_320 {
        return Err(MkError::InvalidProblem(
            "exact sector projector is limited to k <= 8".to_owned(),
        ));
    }
    fn recurse(prefix: &mut Vec<usize>, remaining: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if remaining.is_empty() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..remaining.len() {
            let value = remaining.remove(index);
            prefix.push(value);
            recurse(prefix, remaining, output);
            prefix.pop();
            remaining.insert(index, value);
        }
    }
    let mut output = Vec::with_capacity(group_order);
    recurse(
        &mut Vec::with_capacity(k),
        &mut (0..k).collect(),
        &mut output,
    );
    Ok(output)
}

fn validate_permutation(permutation: &[usize], k: usize) -> Result<(), MkError> {
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    if sorted != (0..k).collect::<Vec<_>>() {
        return Err(MkError::InvalidProblem(
            "variable action is not a permutation of 0..k".to_owned(),
        ));
    }
    Ok(())
}

fn permutation_sign(permutation: &[usize]) -> i64 {
    let mut inversions = 0usize;
    for left in 0..permutation.len() {
        for right in left + 1..permutation.len() {
            inversions += usize::from(permutation[left] > permutation[right]);
        }
    }
    if inversions.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

fn next_permutation_usize(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .expect("a lexicographic successor exists after the pivot");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

fn permute_multi_index(index: &MultiIndex, permutation: &[usize]) -> Result<MultiIndex, MkError> {
    validate_permutation(permutation, index.dimension())?;
    let mut image = vec![0; index.dimension()];
    for (source, &target) in permutation.iter().enumerate() {
        image[target] = index.0[source];
    }
    Ok(MultiIndex(image))
}

fn permute_coefficients(
    reference: &MkMonomialReference,
    coefficients: &[Rational],
    permutation: &[usize],
) -> Result<Vec<Rational>, MkError> {
    if coefficients.len() != reference.dimension() {
        return Err(MkError::DimensionMismatch {
            expected: reference.dimension(),
            actual: coefficients.len(),
        });
    }
    let positions = reference
        .indices()
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, index)| (index, position))
        .collect::<BTreeMap<_, _>>();
    let mut output = vec![Rational::from((0, 1)); coefficients.len()];
    for (column, index) in reference.indices().iter().enumerate() {
        let image = permute_multi_index(index, permutation)?;
        let row = positions.get(&image).ok_or_else(|| {
            MkError::InvalidProblem("permuted coefficient index is absent".to_owned())
        })?;
        output[*row] = coefficients[column].clone();
    }
    Ok(output)
}

fn integer_square_dimension(length: usize) -> usize {
    let mut dimension = 0usize;
    while dimension.saturating_mul(dimension) < length {
        dimension += 1;
    }
    debug_assert_eq!(dimension.saturating_mul(dimension), length);
    dimension
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mk_reuse_plan_separates_forms_solver_and_certificate() {
        let plan = mk_artifact_reuse_plan();
        plan.validate().unwrap();
        for kind in [
            "exact_forms",
            "solver_candidate",
            "exact_candidate_certificate",
        ] {
            assert!(plan.artifacts.iter().any(|node| node.kind == kind));
        }
    }

    #[test]
    fn mk_domain_planner_selects_generalized_hp_maximization_route() {
        let request = MkSolverPlanningRequest {
            basis_dimension: 12,
            requested_candidates: 1,
            assurance: xc_core::AssuranceLevel::Computed,
            precision: xc_core::PrecisionPolicy::fixed(192),
            matrix_materialized: false,
        };
        let plan = xc_solver::DomainSolverPlanner::plan(&MkSolverPlanner, &request).unwrap();
        assert_eq!(plan.domain_id, "maynard_tao_mk_symmetric");
        assert_eq!(plan.input.target, xc_core::EigenTarget::AlgebraicLargest);
        assert!(plan.input.generalized);
        assert_eq!(
            plan.solver_plan.primary,
            xc_solver::SolverRoute::HpMatrixFreeGeneralizedRayleighRitz
        );
    }

    #[test]
    fn multi_index_count_matches_binomial_dimension() {
        // C(D+k, k) = C(5, 2) = 10.
        assert_eq!(enumerate_multi_indices(2, 3).unwrap().len(), 10);
    }

    #[test]
    fn constant_function_k1_has_quotient_one() {
        let engine = MkMonomialReference::new(1, 0).unwrap();
        let coefficients = vec![Rational::from((1, 1))];
        assert_eq!(
            engine.quadratic_i(&coefficients).unwrap(),
            Rational::from((1, 1))
        );
        assert_eq!(
            engine.quadratic_j_total(&coefficients).unwrap(),
            Rational::from((1, 1))
        );
        assert_eq!(
            engine.rayleigh_quotient(&coefficients).unwrap(),
            Rational::from((1, 1))
        );
    }

    #[test]
    fn constant_function_k2_has_quotient_four_thirds() {
        let engine = MkMonomialReference::new(2, 0).unwrap();
        let coefficients = vec![Rational::from((1, 1))];
        assert_eq!(
            engine.quadratic_i(&coefficients).unwrap(),
            Rational::from((1, 2))
        );
        assert_eq!(
            engine.quadratic_j_total(&coefficients).unwrap(),
            Rational::from((2, 3))
        );
        assert_eq!(
            engine.rayleigh_quotient(&coefficients).unwrap(),
            Rational::from((4, 3))
        );
    }

    #[test]
    fn published_maynard_m5_witness_reproduces_exact_historical_bound() {
        let certificate = maynard_2015_m5_certificate().unwrap();
        assert_eq!(certificate.k, 5);
        assert_eq!(certificate.degree, 3);
        assert_eq!(certificate.quotient.numerator, "1417255");
        assert_eq!(certificate.quotient.denominator, "708216");
        xc_certify::exact::verify_rayleigh_lower_bound(
            &certificate.numerator,
            &certificate.denominator,
            &ExactRationalRecord {
                numerator: "2".to_owned(),
                denominator: "1".to_owned(),
            },
        )
        .unwrap();

        // The same polynomial embedded into a larger degree space must keep
        // the identical quotient because all added coordinates are zero.
        let larger = MkMonomialReference::new(5, 5).unwrap();
        let candidate = maynard_2015_m5_candidate(&larger).unwrap();
        assert_eq!(
            larger.rayleigh_quotient(&candidate).unwrap(),
            Rational::from((1_417_255, 708_216))
        );
    }

    #[test]
    fn dense_reference_matrices_are_symmetric() {
        let engine = MkMonomialReference::new(2, 2).unwrap();
        for matrix in [
            engine.dense_i_exact().unwrap(),
            engine.dense_j_total_exact().unwrap(),
        ] {
            let n = engine.dimension();
            for row in 0..n {
                for column in 0..n {
                    assert_eq!(matrix[row * n + column], matrix[column * n + row]);
                }
            }
        }
    }

    #[test]
    fn entries_match_direct_low_degree_values() {
        let engine = MkMonomialReference::new(2, 1).unwrap();
        let a = MultiIndex(vec![1, 0]);
        let b = MultiIndex(vec![0, 1]);
        // Integral x1*x2 over the 2-simplex = 1!1!/(2+2)! = 1/24.
        assert_eq!(engine.i_entry(&a, &b).unwrap(), Rational::from((1, 24)));
    }

    #[test]
    fn symmetric_partition_basis_has_canonical_dimension_and_orbits() {
        let partitions = enumerate_integer_partitions(3, 4).unwrap();
        // Sum of p_k(d) for d=0..4 and partitions with at most three parts:
        // 1 + 1 + 2 + 3 + 4 = 11.
        assert_eq!(partitions.len(), 11);
        assert_eq!(partitions[0], IntegerPartition(Vec::new()));
        assert!(partitions.windows(2).all(|pair| {
            pair[0].total_degree() < pair[1].total_degree()
                || (pair[0].total_degree() == pair[1].total_degree() && pair[0].0 < pair[1].0)
        }));

        let engine = MkSymmetricReference::new(3, 4).unwrap();
        assert_eq!(engine.dimension(), 11);
        assert_eq!(engine.full_monomial_dimension(), 35);
        let orbit = engine
            .orbits()
            .iter()
            .find(|orbit| orbit.partition == IntegerPartition(vec![2, 1]))
            .unwrap();
        assert_eq!(orbit.members.len(), 6);
        assert!(orbit.members.iter().all(|index| index.total_degree() == 3));
    }

    #[test]
    fn symmetric_exact_forms_match_full_monomial_embedding() {
        let engine = MkSymmetricReference::new(3, 3).unwrap();
        let coefficients: Vec<Rational> = (0..engine.dimension())
            .map(|index| Rational::from(((index + 1) as i32, (index + 2) as i32)))
            .collect();
        let expanded = engine.expand_coefficients_exact(&coefficients).unwrap();

        let dense_i = engine.dense_i_exact().unwrap();
        let dense_j = engine.dense_j_total_exact().unwrap();
        let quadratic = |matrix: &[Rational]| {
            let mut total = Rational::from((0, 1));
            for row in 0..engine.dimension() {
                for column in 0..engine.dimension() {
                    let mut term = coefficients[row].clone();
                    term *= &matrix[row * engine.dimension() + column];
                    term *= &coefficients[column];
                    total += term;
                }
            }
            total
        };

        assert_eq!(
            quadratic(&dense_i),
            engine.monomial.quadratic_i(&expanded).unwrap()
        );
        assert_eq!(
            quadratic(&dense_j),
            engine.monomial.quadratic_j_total(&expanded).unwrap()
        );
        assert_eq!(
            engine.quadratic_i(&coefficients).unwrap(),
            quadratic(&dense_i)
        );
        assert_eq!(
            engine.quadratic_j_total(&coefficients).unwrap(),
            quadratic(&dense_j)
        );
    }

    #[test]
    fn symmetric_streamed_actions_match_exact_dense_matrices() {
        let engine = MkSymmetricReference::new(3, 3).unwrap();
        let x: Vec<f64> = (0..engine.dimension())
            .map(|index| (index as f64 + 1.0) / 7.0)
            .collect();
        for (dense, apply) in [
            (
                engine.dense_i_f64().unwrap(),
                MkSymmetricReference::apply_i_f64
                    as fn(&MkSymmetricReference, &[f64], &mut [f64]) -> Result<(), MkError>,
            ),
            (
                engine.dense_j_total_f64().unwrap(),
                MkSymmetricReference::apply_j_total_f64,
            ),
        ] {
            let mut streamed = vec![0.0; engine.dimension()];
            apply(&engine, &x, &mut streamed).unwrap();
            for row in 0..engine.dimension() {
                let expected = (0..engine.dimension())
                    .map(|column| dense[row * engine.dimension() + column] * x[column])
                    .sum::<f64>();
                assert!((streamed[row] - expected).abs() < 2e-14);
            }
        }
    }

    #[test]
    fn symmetric_certificate_declares_restriction_and_exact_embedding() {
        let engine = MkSymmetricReference::new(3, 2).unwrap();
        let mut coefficients = vec![Rational::from((0, 1)); engine.dimension()];
        coefficients[0] = Rational::from((1, 1));
        coefficients[2] = Rational::from((1, 3));
        let certificate = engine.certificate(&coefficients).unwrap();

        assert_eq!(certificate.symmetric_dimension, engine.dimension());
        assert_eq!(
            certificate.monomial_embedding_dimension,
            engine.full_monomial_dimension()
        );
        assert_eq!(certificate.search_space, MkSymmetricReference::SEARCH_SPACE);
        assert!(!certificate.search_space.contains("global optimum"));
        assert!(certificate.denominator.numerator.parse::<i128>().unwrap() > 0);
    }

    #[test]
    fn symmetric_dense_generalized_problem_uses_i_as_metric() {
        use xc_core::EigenTarget;
        use xc_solver::{DenseGeneralizedProblemF64, DenseGeneralizedReferenceSolverF64};

        let engine = MkSymmetricReference::new(2, 2).unwrap();
        let j = engine.dense_j_total_f64().unwrap();
        let i = engine.dense_i_f64().unwrap();
        let problem = DenseGeneralizedProblemF64::new(&j, &i, engine.dimension(), 1e-13)
            .expect("the exact Mk forms are symmetric");
        let result = DenseGeneralizedReferenceSolverF64::default()
            .solve(&problem, &EigenTarget::AlgebraicLargest)
            .unwrap();
        assert!(result.eigenvalue.is_finite() && result.eigenvalue > 0.0);
        assert!(result.residual_norm < 1e-9);

        let rational_candidate: Vec<Rational> = result
            .eigenvector
            .iter()
            .map(|value| {
                let scaled = (value * 1_000_000.0).round() as i64;
                Rational::from((scaled, 1_000_000))
            })
            .collect();
        let certificate = engine.certificate(&rational_candidate).unwrap();
        assert!(certificate.quotient.numerator.parse::<i128>().unwrap() > 0);
    }

    #[test]
    fn symmetric_largest_generalized_eigenvalue_agrees_dense_and_matrix_free() {
        use xc_core::EigenTarget;
        use xc_operator::GeneralizedEigenProblem;
        use xc_solver::{
            DenseGeneralizedProblemF64, DenseGeneralizedReferenceSolverF64,
            GeneralizedExtremeConfigF64, MatrixFreeLobpcgF64,
        };

        let engine = MkSymmetricReference::new(2, 2).unwrap();
        let dense_j = engine.dense_j_total_f64().unwrap();
        let dense_i = engine.dense_i_f64().unwrap();
        let dense_problem =
            DenseGeneralizedProblemF64::new(&dense_j, &dense_i, engine.dimension(), 1e-13).unwrap();
        let dense = DenseGeneralizedReferenceSolverF64::default()
            .solve(&dense_problem, &EigenTarget::AlgebraicLargest)
            .unwrap();

        let operator = MkSymmetricJOperatorF64::new(&engine).unwrap();
        let metric = MkSymmetricIMetricF64::new(&engine).unwrap();
        let matrix_free_problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let matrix_free = MatrixFreeLobpcgF64
            .solve(
                &matrix_free_problem,
                &GeneralizedExtremeConfigF64 {
                    target: EigenTarget::AlgebraicLargest,
                    absolute_residual_tolerance: 1e-11,
                    scaled_backward_error_tolerance: 1e-11,
                    ritz_value_stability_tolerance: 1e-12,
                    maximum_iterations: 500,
                    minimum_iterations: 2,
                },
            )
            .unwrap();
        let difference = (dense.eigenvalue - matrix_free.eigenvalue).abs();
        assert!(
            difference < 1e-9,
            "largest generalized eigenvalue differs by {difference:e}"
        );
        assert!(dense.residual_norm < 1e-9);
        assert!(
            matrix_free.scaled_backward_error < 1e-9,
            "matrix-free backward error is {:e} (absolute residual {:e})",
            matrix_free.scaled_backward_error,
            matrix_free.residual_norm
        );
        assert_eq!(matrix_free.status, xc_core::ResultStatus::Converged);
    }

    #[test]
    fn symmetric_hp_streamed_actions_match_exact_rational_products() {
        let precision = 192;
        let engine = MkSymmetricReference::new(3, 2).unwrap();
        let exact_input: Vec<Rational> = (1..=engine.dimension())
            .map(|value| Rational::from((value as i32, 3)))
            .collect();
        let input: Vec<Float> = exact_input
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();

        for (apply, entry) in [
            (
                MkSymmetricReference::apply_i_hp
                    as fn(
                        &MkSymmetricReference,
                        &[Float],
                        &mut [Float],
                        u32,
                    ) -> Result<(), MkError>,
                MkSymmetricReference::i_entry
                    as fn(&MkSymmetricReference, usize, usize) -> Result<Rational, MkError>,
            ),
            (
                MkSymmetricReference::apply_j_total_hp,
                MkSymmetricReference::j_total_entry,
            ),
        ] {
            let mut actual = vec![Float::with_val(precision, 0); engine.dimension()];
            apply(&engine, &input, &mut actual, precision).unwrap();
            for (row, value) in actual.iter().enumerate() {
                let mut exact = Rational::from((0, 1));
                for (column, coefficient) in exact_input.iter().enumerate() {
                    exact += entry(&engine, row, column).unwrap() * coefficient;
                }
                let mut difference = value.clone();
                difference -= Float::with_val(precision, exact);
                difference.abs_mut();
                assert!(difference < Float::with_val(precision, 1e-50));
            }
        }
    }

    #[test]
    fn symmetric_mk_exact_forms_request_both_generalized_extremes_in_mpfr() {
        use xc_core::{DecimalLiteral, EigenTarget, ResultStatus};
        use xc_solver::{
            solve_dense_generalized_whitening_hp, DenseGeneralizedProblemHp,
            GeneralizedExtremeConfigHp,
        };

        let precision = 192;
        let engine = MkSymmetricReference::new(2, 2).unwrap();
        let stiffness: Vec<Float> = engine
            .dense_j_total_exact()
            .unwrap()
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();
        let gram: Vec<Float> = engine
            .dense_i_exact()
            .unwrap()
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();
        let problem =
            DenseGeneralizedProblemHp::new(&stiffness, &gram, engine.dimension()).unwrap();

        let solve = |target| {
            solve_dense_generalized_whitening_hp(
                &problem,
                &GeneralizedExtremeConfigHp {
                    target,
                    precision_bits: precision,
                    absolute_residual_tolerance: DecimalLiteral::new("1e-35").unwrap(),
                    scaled_backward_error_tolerance: DecimalLiteral::new("1e-35").unwrap(),
                    ritz_value_stability_tolerance: DecimalLiteral::new("1e-35").unwrap(),
                    maximum_iterations: 5_000,
                    minimum_iterations: 2,
                },
            )
            .unwrap()
        };
        let smallest = solve(EigenTarget::AlgebraicSmallest);
        let largest = solve(EigenTarget::AlgebraicLargest);

        assert_eq!(smallest.status, ResultStatus::Converged);
        assert_eq!(largest.status, ResultStatus::Converged);
        assert!(smallest.eigenvalue < largest.eigenvalue);
        assert!(smallest.residual_norm < Float::with_val(precision, 1e-30));
        assert!(largest.residual_norm < Float::with_val(precision, 1e-30));
        assert!(smallest.metric_normalization_error < Float::with_val(precision, 1e-30));
        assert!(largest.metric_normalization_error < Float::with_val(precision, 1e-30));
    }

    #[test]
    fn symmetric_mk_largest_generalized_eigenvalue_runs_matrix_free_in_mpfr() {
        use xc_core::{
            DecimalLiteral, EigenTarget, PrecisionEscalation, PrecisionPolicy, ResultStatus,
        };
        use xc_operator::GeneralizedEigenProblem;
        use xc_solver::{
            cross_check_generalized_hp_reports, solve_dense_generalized_whitening_hp,
            solve_matrix_free_generalized_adaptive_hp, AdaptiveGeneralizedExtremeOptionsHp,
            AdaptiveGeneralizedExtremeResultHp, DenseGeneralizedProblemHp,
            GeneralizedExtremeConfigHp, HpCrossCheckTolerance,
        };

        let precision = 256;
        let engine = MkSymmetricReference::new(2, 2).unwrap();
        let dense_j: Vec<Float> = engine
            .dense_j_total_exact()
            .unwrap()
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();
        let dense_i: Vec<Float> = engine
            .dense_i_exact()
            .unwrap()
            .iter()
            .map(|value| Float::with_val(precision, value))
            .collect();

        let operator = MkSymmetricJOperatorHp::new(&engine, precision).unwrap();
        let metric = MkSymmetricIMetricHp::new(&engine, precision).unwrap();
        let problem = GeneralizedEigenProblem::new(&operator, &metric).unwrap();
        let adaptive = solve_matrix_free_generalized_adaptive_hp(
            &problem,
            &AdaptiveGeneralizedExtremeOptionsHp {
                target: EigenTarget::AlgebraicLargest,
                absolute_residual_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                scaled_backward_error_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                ritz_value_stability_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                maximum_iterations: 5_000,
                minimum_iterations: 2,
                precision: PrecisionPolicy {
                    initial_bits: 64,
                    maximum_bits: precision,
                    guard_bits: 0,
                    escalation: PrecisionEscalation::AddBits(192),
                },
            },
        )
        .unwrap();
        let (hp, attempts) = match adaptive {
            AdaptiveGeneralizedExtremeResultHp::Converged { result, attempts } => {
                (*result, attempts)
            }
            AdaptiveGeneralizedExtremeResultHp::Inconclusive {
                attempts, reason, ..
            } => panic!("adaptive HP Mk solve remained inconclusive: {reason}; {attempts:?}"),
        };

        assert_eq!(
            hp.status,
            ResultStatus::Converged,
            "iterations={}, residual={}, backward={}, stability={}, value={}",
            hp.iterations,
            hp.residual_norm,
            hp.scaled_backward_error,
            hp.ritz_value_stability,
            hp.eigenvalue
        );
        assert!(hp.residual_norm < Float::with_val(precision, 1e-40));
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].precision_bits, 64);
        assert_eq!(attempts[0].status, ResultStatus::Approximate);
        assert_eq!(attempts[1].precision_bits, precision);
        assert_eq!(attempts[1].status, ResultStatus::Converged);
        assert_eq!(operator.precision_bits(), precision);
        assert_eq!(metric.precision_bits(), precision);
        assert_eq!(operator.metadata().structure, MatrixStructure::MatrixFree);
        assert_eq!(metric.metadata().structure, MatrixStructure::MatrixFree);
        assert!(!operator.metadata().exact_action);
        assert!(!metric.metadata().exact_action);

        let dense_problem =
            DenseGeneralizedProblemHp::new(&dense_j, &dense_i, engine.dimension()).unwrap();
        let dense = solve_dense_generalized_whitening_hp(
            &dense_problem,
            &GeneralizedExtremeConfigHp {
                target: EigenTarget::AlgebraicLargest,
                precision_bits: precision,
                absolute_residual_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                scaled_backward_error_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                ritz_value_stability_tolerance: DecimalLiteral::new("1e-45").unwrap(),
                maximum_iterations: 5_000,
                minimum_iterations: 2,
            },
        )
        .unwrap();
        let checked = cross_check_generalized_hp_reports(
            &problem,
            &hp,
            &dense,
            &HpCrossCheckTolerance {
                eigenvalue_absolute: DecimalLiteral::new("1e-35").unwrap(),
                one_minus_overlap_squared: DecimalLiteral::new("1e-35").unwrap(),
            },
        )
        .unwrap();
        assert_eq!(checked.assurance, xc_core::AssuranceLevel::CrossChecked);
        assert!(checked.eigenvalue_absolute_difference < Float::with_val(precision, 1e-35));
        assert!(checked.one_minus_metric_overlap_squared < Float::with_val(precision, 1e-35));
    }

    #[test]
    fn symmetric_mk_three_route_acceptance_round_trips_and_rejects_tampering() {
        use xc_core::DecimalLiteral;

        let options = MkThreeRouteAcceptanceOptions {
            k: 2,
            degree: 2,
            precision_bits: 256,
            initial_precision_bits: 64,
            absolute_residual_tolerance: DecimalLiteral::new("1e-45").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-45").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-45").unwrap(),
            eigenvalue_agreement_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            overlap_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            candidate_quotient_agreement_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            maximum_iterations: 5_000,
        };
        let record = run_mk_three_route_acceptance(&options).unwrap();
        verify_mk_three_route_acceptance(&record, &options).unwrap();
        assert_eq!(record.adaptive_attempt_precisions, vec![64, 256]);
        assert_eq!(record.symmetric_dimension, 4);
        assert!(record.exact_source_forms);

        let encoded = serde_json::to_vec(&record).unwrap();
        let decoded: MkThreeRouteAcceptanceRecord = serde_json::from_slice(&encoded).unwrap();
        verify_mk_three_route_acceptance(&decoded, &options).unwrap();

        let mut tampered = decoded;
        tampered.candidate_certificate.quotient.numerator = "0".to_owned();
        assert!(verify_mk_three_route_acceptance(&tampered, &options).is_err());
    }

    #[test]
    fn symmetric_mk_scale_acceptance_runs_beyond_declared_dense_degree_limit() {
        use xc_core::DecimalLiteral;

        let options = MkScaleAcceptanceOptions {
            historical_dense_degree_limit: 3,
            target_degree: 6,
            precision_bits: 192,
            minimum_exact_lower_bound: ExactRationalRecord {
                numerator: "2".to_owned(),
                denominator: "1".to_owned(),
            },
            quotient_agreement_tolerance: DecimalLiteral::new("1e-45").unwrap(),
        };
        let record = run_mk_scale_acceptance(&options).unwrap();
        verify_mk_scale_acceptance(&record, &options).unwrap();
        assert!(record.target_degree > record.historical_dense_degree_limit);
        assert!(record.symmetric_dimension > 0);
        assert!(record.full_monomial_embedding_dimension > record.symmetric_dimension);
        assert!(record.streamed_working_vector_bytes < record.equivalent_dense_forms_bytes);
        assert_eq!(
            record.exact_certificate.quotient,
            ExactRationalRecord {
                numerator: "1417255".to_owned(),
                denominator: "708216".to_owned(),
            }
        );

        let mut tampered = record;
        tampered.operator_representation = "dense".to_owned();
        assert!(verify_mk_scale_acceptance(&tampered, &options).is_err());
    }

    #[test]
    fn symmetric_operator_adapters_report_matrix_free_semantics() {
        let engine = MkSymmetricReference::new(3, 2).unwrap();
        let operator = MkSymmetricJOperatorF64::new(&engine).unwrap();
        let metric = MkSymmetricIMetricF64::new(&engine).unwrap();
        assert_eq!(operator.dimension(), engine.dimension());
        assert_eq!(metric.dimension(), engine.dimension());
        assert_eq!(operator.metadata().structure, MatrixStructure::MatrixFree);
        assert_eq!(metric.metadata().structure, MatrixStructure::MatrixFree);
        assert!(operator.norm_bound().unwrap().is_finite());
        assert!(metric.norm_bound().unwrap().is_finite());
        let feasibility = engine.feasibility();
        assert_eq!(feasibility.dimension, engine.dimension());
        assert_eq!(feasibility.basis_count, engine.dimension());
        assert_eq!(
            feasibility.total_orbit_members,
            engine.full_monomial_dimension()
        );
        assert_eq!(
            feasibility.exact_entry_evaluations_per_streamed_form_application,
            (engine.dimension() * engine.dimension()) as u64
        );
        assert!(feasibility.dense_i_and_j_f64_bytes > feasibility.f64_vector_bytes);
        assert!(feasibility.exact_certificate_multiply_add_upper_bound > 0);
        assert_eq!(
            feasibility.symmetry_restriction,
            MkSymmetricReference::SEARCH_SPACE
        );
    }

    #[test]
    fn exact_simplex_basis_is_symbolically_orthonormal() {
        let reference = MkMonomialReference::new(2, 3).unwrap();
        let basis = ExactSimplexOrthonormalBasis::new(&reference).unwrap();
        assert_eq!(basis.k(), 2);
        assert_eq!(basis.degree(), 3);
        assert_eq!(basis.dimension(), reference.dimension());
        assert!(basis.vectors().iter().all(|vector| vector.squared_norm > 0));
        basis.verify_exact(&reference).unwrap();
    }

    #[test]
    fn pkd_shifted_jacobi_recurrence_and_simplex_norms_are_exact() {
        assert_eq!(
            shifted_jacobi_alpha_zero_coefficients(2, 0).unwrap(),
            vec![
                Rational::from((1, 1)),
                Rational::from((-6, 1)),
                Rational::from((6, 1)),
            ]
        );
        assert_eq!(
            shifted_jacobi_alpha_zero_coefficients(1, 1).unwrap(),
            vec![Rational::from((-1, 1)), Rational::from((3, 1))]
        );

        let reference_2d = MkMonomialReference::new(2, 3).unwrap();
        let basis_2d = ExactPkdSimplexOrthonormalBasis::new(&reference_2d).unwrap();
        assert_eq!(basis_2d.k(), 2);
        assert_eq!(basis_2d.degree(), 3);
        assert_eq!(basis_2d.dimension(), reference_2d.dimension());
        assert_eq!(basis_2d.vectors()[0].squared_norm, Rational::from((1, 2)));
        basis_2d.verify_exact(&reference_2d).unwrap();

        let reference_3d = MkMonomialReference::new(3, 2).unwrap();
        let basis_3d = ExactPkdSimplexOrthonormalBasis::new(&reference_3d).unwrap();
        assert_eq!(basis_3d.vectors()[0].squared_norm, Rational::from((1, 6)));
        basis_3d.verify_exact(&reference_3d).unwrap();
    }

    #[test]
    fn certified_acceleration_rejects_unproved_degree_band() {
        let band = MkOperatorAcceleration::DegreeDifferenceBand {
            half_width: 2,
            certificate: None,
        };
        assert!(validate_mk_acceleration(&band, MkAssuranceMode::Exploratory, 4, 6).is_ok());
        let error = validate_mk_acceleration(&band, MkAssuranceMode::Certified, 4, 6).unwrap_err();
        assert!(error
            .to_string()
            .contains("unbounded degree-difference band"));

        let bound = Rational::from((1, 1_000_000));
        let rounding = Rational::from((1, 10_000_000));
        let certified_band = MkOperatorAcceleration::DegreeDifferenceBand {
            half_width: 2,
            certificate: Some(MkApproximationCertificate {
                construction: "validated shell-tail enclosure".to_owned(),
                k: 4,
                degree: 6,
                metric: MkApproximationMetric::IMetricOperatorNorm,
                rigorous_operator_error_bound: exact_record(&bound),
                rounding_contribution: exact_record(&rounding),
                parameter_range: "degree 0..=6".to_owned(),
                validation_digest: "sha256:fixture".to_owned(),
            }),
        };
        validate_mk_acceleration(&certified_band, MkAssuranceMode::Certified, 4, 6).unwrap();
    }

    #[test]
    fn constructed_degree_band_action_has_replayable_exact_norm_bound() {
        let reference = MkSymmetricReference::new(3, 4).unwrap();
        let action =
            MkCertifiedDegreeBandAction::construct(&reference, MkSymmetricForm::JTotal, 0).unwrap();
        assert!(action.retained_entries > 0);
        assert!(action.omitted_entries > 0);
        assert_eq!(
            action.retained_entries + action.omitted_entries,
            (reference.dimension() * reference.dimension()) as u64
        );
        assert_eq!(
            action.certificate.metric,
            MkApproximationMetric::EuclideanOperatorNorm
        );
        validate_mk_acceleration(
            &action.acceleration(),
            MkAssuranceMode::Certified,
            reference.k(),
            reference.degree(),
        )
        .unwrap();

        let input = (1..=reference.dimension())
            .map(|value| Rational::from((value, reference.dimension())))
            .collect::<Vec<_>>();
        let approximate = action
            .verify(&reference)
            .unwrap()
            .apply_exact(&input)
            .unwrap();
        let mut on_the_fly = vec![Rational::from((0, 1)); reference.dimension()];
        for (row, output) in on_the_fly.iter_mut().enumerate() {
            let row_degree = reference.orbits()[row].partition.total_degree();
            for (column, coefficient) in input.iter().enumerate() {
                let column_degree = reference.orbits()[column].partition.total_degree();
                if row_degree.abs_diff(column_degree) <= action.half_width {
                    let mut term =
                        symmetric_form_entry(&reference, action.form, row, column).unwrap();
                    term *= coefficient;
                    *output += term;
                }
            }
        }
        assert_eq!(approximate, on_the_fly);
        let dense = reference.dense_j_total_exact().unwrap();
        let mut exact = vec![Rational::from((0, 1)); reference.dimension()];
        for row in 0..reference.dimension() {
            for column in 0..reference.dimension() {
                let mut term = dense[row * reference.dimension() + column].clone();
                term *= &input[column];
                exact[row] += term;
            }
        }
        let error_squared = exact
            .iter()
            .zip(&approximate)
            .map(|(exact, approximate)| {
                let difference = exact.clone() - approximate;
                difference.clone() * difference
            })
            .sum::<Rational>();
        let input_squared = input
            .iter()
            .map(|value| value.clone() * value)
            .sum::<Rational>();
        let bound =
            rational_record_value(&action.certificate.rigorous_operator_error_bound).unwrap();
        assert!(error_squared <= bound.clone() * bound * input_squared);

        let decoded: MkCertifiedDegreeBandAction =
            serde_json::from_slice(&serde_json::to_vec(&action).unwrap()).unwrap();
        decoded.verify(&reference).unwrap();
        let mut tampered = decoded;
        tampered.certificate.rigorous_operator_error_bound = exact_record(&Rational::from((0, 1)));
        assert!(tampered.verify(&reference).is_err());

        let metric_action =
            MkCertifiedDegreeBandAction::construct(&reference, MkSymmetricForm::IMetric, 1)
                .unwrap();
        metric_action.verify(&reference).unwrap();
    }

    #[test]
    fn adaptive_spaces_are_nested_and_warm_start_is_exactly_prolonged() {
        let history = build_adaptive_symmetric_spaces(&AdaptiveSpacePolicy {
            k: 3,
            initial_degree: 1,
            maximum_degree: 4,
            maximum_generations: 4,
            enrichment_rule: AdaptiveEnrichmentRule::CompleteDegreeShell,
        })
        .unwrap();
        assert_eq!(history.len(), 4);
        assert_eq!(
            history.last().unwrap().stopping_reason.as_deref(),
            Some("maximum_degree")
        );
        for pair in history.windows(2) {
            assert!(pair[0]
                .realized_basis
                .iter()
                .all(|partition| pair[1].realized_basis.contains(partition)));
            assert!(pair[1]
                .accepted_block
                .iter()
                .all(|partition| partition.total_degree() == pair[1].degree));
        }

        let parent = &history[1];
        let child = &history[2];
        let parent_coefficients = (0..parent.realized_basis.len())
            .map(|index| index as f64 + 0.5)
            .collect::<Vec<_>>();
        let warm = prolong_symmetric_warm_start(
            &parent.realized_basis,
            &parent_coefficients,
            &child.realized_basis,
            parent.generation,
            child.generation,
        )
        .unwrap();
        assert!(!warm.certification_dependency);
        for (partition, coefficient) in parent.realized_basis.iter().zip(&parent_coefficients) {
            let position = child
                .realized_basis
                .iter()
                .position(|candidate| candidate == partition)
                .unwrap();
            assert_eq!(warm.coefficients[position], *coefficient);
        }
    }

    #[test]
    fn randomized_warm_starts_are_reproducible_but_never_certification_evidence() {
        let mut config = xc_core::SolverConfig {
            target: xc_core::EigenTarget::AlgebraicLargest,
            subspace: xc_core::Subspace::Full,
            assurance: xc_core::AssuranceLevel::Certified,
            precision: xc_core::PrecisionPolicy::fixed(192),
            stopping: xc_core::StoppingPolicy::default(),
            reproducibility: xc_core::Reproducibility::Deterministic,
            algorithm_preferences: Vec::new(),
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        };
        assert!(exploratory_random_warm_start(8, 42, &config).is_err());
        config.allow_randomized_seed = true;

        let first = exploratory_random_warm_start(8, 42, &config).unwrap();
        let repeat = exploratory_random_warm_start(8, 42, &config).unwrap();
        let other = exploratory_random_warm_start(8, 43, &config).unwrap();
        assert_eq!(first, repeat);
        assert_ne!(first.coefficients, other.coefficients);
        assert_eq!(first.exploratory_seed, Some(42));
        assert_eq!(first.seed_generator, xc_core::SeedGeneratorKind::Randomized);
        assert!(!first.certification_dependency);
        config
            .validate_seed_use(&xc_core::SeedUseEvidence {
                generator: first.seed_generator,
                used_only_as_initial_guess: true,
                final_hp_verification: true,
                acceptance_depended_on_seed: first.certification_dependency,
            })
            .unwrap();
        let norm = first
            .coefficients
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-14);
    }

    #[test]
    fn exact_s3_sector_projectors_are_complete_orthogonal_and_equivariant() {
        let reference = MkMonomialReference::new(3, 3).unwrap();
        let (projectors, report) = exact_sector_coverage(&reference).unwrap();
        assert_eq!(projectors.len(), 3);
        assert!(report.pairwise_orthogonal);
        assert!(report.reconstructs_full_space);
        assert_eq!(
            report
                .sector_dimensions
                .iter()
                .map(|entry| entry.dimension)
                .sum::<usize>(),
            reference.dimension()
        );
        assert!(report
            .sector_dimension(&MkPermutationSector::Standard)
            .is_some_and(|dimension| dimension > 0));
        assert!(report
            .sector_dimension(&MkPermutationSector::Alternating)
            .is_some_and(|dimension| dimension > 0));
        for projector in &projectors {
            projector.verify_exact().unwrap();
            projector
                .verify_commutes_with(&reference, &[1, 2, 0])
                .unwrap();
        }
    }

    #[test]
    fn s4_partition_characters_match_the_exact_table() {
        let partition = IntegerPartition(vec![2, 2]);
        assert_eq!(representation_dimension(&partition).unwrap(), 2);
        assert_eq!(irreducible_character(&partition, &[1, 1, 1, 1]).unwrap(), 2);
        assert_eq!(irreducible_character(&partition, &[2, 1, 1]).unwrap(), 0);
        assert_eq!(irreducible_character(&partition, &[2, 2]).unwrap(), 2);
        assert_eq!(irreducible_character(&partition, &[3, 1]).unwrap(), -1);
        assert_eq!(irreducible_character(&partition, &[4]).unwrap(), 0);
    }

    #[test]
    fn exact_s4_partition_projectors_are_complete_and_requestable() {
        let reference = MkMonomialReference::new(4, 3).unwrap();
        let (projectors, report) = exact_sector_coverage(&reference).unwrap();
        assert_eq!(projectors.len(), 5);
        assert!(report.pairwise_orthogonal);
        assert!(report.reconstructs_full_space);
        assert_eq!(
            report
                .sector_dimensions
                .iter()
                .map(|entry| entry.dimension)
                .sum::<usize>(),
            reference.dimension()
        );
        assert!(report.claim_scope.contains("partition-labelled"));
        let decoded: MkSectorCoverageReport =
            serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(decoded, report);
        let requested = MkPermutationSector::Partition(IntegerPartition(vec![2, 2]));
        let projector = MkSectorProjector::new(&reference, requested.clone()).unwrap();
        assert_eq!(projector.sector(), requested);
        assert!(projector.sector_dimension().unwrap() > 0);
        projector
            .verify_commutes_with(&reference, &[1, 2, 3, 0])
            .unwrap();
    }

    #[test]
    fn exact_s5_partition_projectors_reconstruct_the_full_space() {
        let reference = MkMonomialReference::new(5, 2).unwrap();
        let (projectors, report) = exact_sector_coverage(&reference).unwrap();
        assert_eq!(projectors.len(), 7);
        assert!(report.pairwise_orthogonal);
        assert!(report.reconstructs_full_space);
        assert_eq!(
            report
                .sector_dimensions
                .iter()
                .map(|entry| entry.dimension)
                .sum::<usize>(),
            reference.dimension()
        );
    }
}
