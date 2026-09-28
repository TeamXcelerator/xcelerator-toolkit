//! Validation-scale exact I-orthogonal coordinates for a streamed J action.
//! The symbolic columns satisfy B^T I B=D exactly. Their inverse square-root
//! normalization is computed at guarded precision; acceptance always replays
//! the lifted state against the original exact I and J forms.
use super::*;

pub(super) fn budget<'a>(values: impl Iterator<Item = &'a Rational>) -> Result<(), MkError> {
    let mut bits = 0u128;
    for x in values {
        bits += u128::from(x.numer().significant_bits()) + u128::from(x.denom().significant_bits());
        if bits > 67_108_864 {
            return Err(MkError::InvalidProblem(
                "exact Maynard coordinate algebra exceeds 64 Mbit".into(),
            ));
        }
    }
    Ok(())
}
pub(super) struct Coordinates<'a> {
    reference: &'a MkSymmetricReference,
    columns: Vec<Vec<Float>>,
    precision: u32,
}
impl<'a> Coordinates<'a> {
    pub(super) fn new(reference: &'a MkSymmetricReference, p: u32) -> Result<Self, MkError> {
        let n = reference.dimension();
        if n == 0 || n > 128 {
            return Err(MkError::InvalidProblem(
                "exact I-orthogonal acceptance is limited to 128 symmetric directions".into(),
            ));
        }
        let gram = reference.dense_i_exact()?;
        budget(gram.iter())?;
        let mut basis: Vec<ExactOrthonormalVector> = Vec::with_capacity(n);
        for index in 0..n {
            let mut v = vec![Rational::from(0); n];
            v[index] = Rational::from(1);
            for previous in &basis {
                let projection = exact_dense_inner_product(&v, &previous.coefficients, &gram)?
                    / &previous.squared_norm;
                for (a, b) in v.iter_mut().zip(&previous.coefficients) {
                    *a -= projection.clone() * b;
                }
                budget(v.iter())?;
            }
            let norm = exact_dense_inner_product(&v, &v, &gram)?;
            if norm <= 0 {
                return Err(MkError::NonPositiveDenominator);
            }
            budget(
                basis
                    .iter()
                    .flat_map(|v| v.coefficients.iter())
                    .chain(&v)
                    .chain(std::iter::once(&norm)),
            )?;
            basis.push(ExactOrthonormalVector {
                coefficients: v,
                squared_norm: norm,
            });
        }
        let work = p
            .checked_add(64)
            .ok_or_else(|| MkError::InvalidProblem("coordinate precision overflow".into()))?;
        let mut columns = Vec::with_capacity(n);
        for vector in basis {
            let norm = Float::with_val(work, vector.squared_norm).sqrt();
            if !norm.is_finite() || norm <= 0 {
                return Err(MkError::NonPositiveDenominator);
            }
            let column = vector
                .coefficients
                .into_iter()
                .map(|x| {
                    let v = Float::with_val(work, &x) / &norm;
                    if !v.is_finite() || (v == 0 && x != 0) {
                        return Err(MkError::InvalidProblem(
                            "I-coordinate normalization exceeds MPFR range".into(),
                        ));
                    }
                    Ok(v)
                })
                .collect::<Result<Vec<_>, _>>()?;
            columns.push(column);
        }
        Ok(Self {
            reference,
            columns,
            precision: work,
        })
    }
    pub(super) fn lift(&self, x: &[Float], p: u32) -> Result<Vec<Float>, MkError> {
        let n = self.reference.dimension();
        if x.len() != n {
            return Err(MkError::DimensionMismatch {
                expected: n,
                actual: x.len(),
            });
        }
        let mut out = vec![Float::with_val(self.precision, 0); n];
        for (c, coefficient) in self.columns.iter().zip(x) {
            for (y, b) in out.iter_mut().zip(c) {
                *y += Float::with_val(self.precision, b * coefficient);
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(MkError::InvalidProblem(
                "I-coordinate lift exceeds MPFR range".into(),
            ));
        }
        Ok(out.into_iter().map(|x| Float::with_val(p, x)).collect())
    }
    pub(super) fn lift_unit(&self, x: &[Float], p: u32) -> Result<Vec<Float>, MkError> {
        let mut out = self.lift(x, self.precision)?;
        let mut image = vec![Float::with_val(self.precision, 0); out.len()];
        self.reference
            .apply_i_hp(&out, &mut image, self.precision)?;
        let terms = out
            .iter()
            .zip(&image)
            .map(|(x, y)| Float::with_val(self.precision, x * y))
            .collect::<Vec<_>>();
        let norm = Float::with_val(self.precision, Float::sum(terms.iter())).sqrt();
        if !norm.is_finite() || norm <= 0 {
            return Err(MkError::NonPositiveDenominator);
        }
        for x in &mut out {
            *x /= &norm;
        }
        Ok(out.into_iter().map(|x| Float::with_val(p, x)).collect())
    }
}
impl LinearOperator<Float> for Coordinates<'_> {
    fn dimension(&self) -> usize {
        self.reference.dimension()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        if y.len() != self.dimension() {
            return Err(mk_operator_error(MkError::DimensionMismatch {
                expected: self.dimension(),
                actual: y.len(),
            }));
        }
        let lifted = self.lift(x, self.precision).map_err(mk_operator_error)?;
        let mut action = vec![Float::with_val(self.precision, 0); self.dimension()];
        self.reference
            .apply_j_total_hp(&lifted, &mut action, self.precision)
            .map_err(mk_operator_error)?;
        for (out, column) in y.iter_mut().zip(&self.columns) {
            let terms = column
                .iter()
                .zip(&action)
                .map(|(a, b)| Float::with_val(self.precision, a * b))
                .collect::<Vec<_>>();
            *out = Float::with_val(self.precision, Float::sum(terms.iter()));
            if !out.is_finite() {
                return Err(mk_operator_error(MkError::InvalidProblem(
                    "transformed J action exceeds MPFR range".into(),
                )));
            }
        }
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        let mut m = OperatorMetadata::new(
            "mk_exact_i_orthogonal_streamed_j",
            self.dimension(),
            MatrixStructure::MatrixFree,
            "rug_mpfr_guarded_symbolic_i_coordinates",
        );
        m.symmetric = true;
        m.exact_action = false;
        m
    }
}
impl SymmetricOperator<Float> for Coordinates<'_> {}
pub(super) struct Identity(pub usize);
impl LinearOperator<Float> for Identity {
    fn dimension(&self) -> usize {
        self.0
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        if x.len() != self.0 || y.len() != self.0 {
            return Err(mk_operator_error(MkError::DimensionMismatch {
                expected: self.0,
                actual: x.len(),
            }));
        }
        y.clone_from_slice(x);
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        let mut m = OperatorMetadata::new(
            "identity_in_symbolic_i_coordinates",
            self.0,
            MatrixStructure::MatrixFree,
            "exact_identity",
        );
        m.symmetric = true;
        m.exact_action = true;
        m
    }
}
impl SymmetricOperator<Float> for Identity {}
impl PositiveDefiniteMetric<Float> for Identity {}
