#![cfg(feature = "hp-reference")]

use rug::{
    float::{Constant, Round},
    ops::{AddAssignRound, DivAssignRound, MulAssignRound},
    Float, Rational,
};
use xc_core::{DecimalLiteral, EigenTarget, ResultStatus};
use xc_operator::{
    GeneralizedEigenProblem, LinearOperator, MatrixStructure, OperatorError, OperatorMetadata,
    PositiveDefiniteMetric, SymmetricOperator,
};
use xc_solver::{
    GeneralizedExtremeConfigF64, GeneralizedExtremeConfigHp, MatrixFreeGeneralizedRayleighRitzHp,
    MatrixFreeLobpcgF64,
};

#[derive(Clone)]
struct IntegerTridiagonal {
    diagonal: Vec<i32>,
    off: Vec<i32>,
}
impl IntegerTridiagonal {
    fn uniform(n: usize, diagonal: i32, off: i32) -> Self {
        Self {
            diagonal: vec![diagonal; n],
            off: vec![off; n - 1],
        }
    }
    fn exact_action(&self, x: &[Rational]) -> Vec<Rational> {
        (0..x.len())
            .map(|i| {
                let mut y = Rational::from(&x[i] * self.diagonal[i]);
                if i > 0 {
                    y += Rational::from(&x[i - 1] * self.off[i - 1]);
                }
                if i + 1 < x.len() {
                    y += Rational::from(&x[i + 1] * self.off[i]);
                }
                y
            })
            .collect()
    }
    fn metadata(&self, backend: &str) -> OperatorMetadata {
        let mut m = OperatorMetadata::new(
            "integer tridiagonal oracle pencil",
            self.diagonal.len(),
            MatrixStructure::Tridiagonal,
            backend,
        );
        m.symmetric = true;
        m
    }
}
impl LinearOperator<f64> for IntegerTridiagonal {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), OperatorError> {
        for i in 0..x.len() {
            y[i] = f64::from(self.diagonal[i]) * x[i];
            if i > 0 {
                y[i] += f64::from(self.off[i - 1]) * x[i - 1];
            }
            if i + 1 < x.len() {
                y[i] += f64::from(self.off[i]) * x[i + 1];
            }
        }
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        self.metadata("f64")
    }
}
impl SymmetricOperator<f64> for IntegerTridiagonal {}
impl LinearOperator<Float> for IntegerTridiagonal {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }
    fn apply(&self, x: &[Float], y: &mut [Float]) -> Result<(), OperatorError> {
        for i in 0..x.len() {
            let p = y[i].prec();
            y[i] = Float::with_val(p, &x[i] * self.diagonal[i]);
            if i > 0 {
                y[i] += Float::with_val(p, &x[i - 1] * self.off[i - 1]);
            }
            if i + 1 < x.len() {
                y[i] += Float::with_val(p, &x[i + 1] * self.off[i]);
            }
        }
        Ok(())
    }
    fn metadata(&self) -> OperatorMetadata {
        self.metadata("rug_mpfr")
    }
}
impl SymmetricOperator<Float> for IntegerTridiagonal {}
struct Metric(IntegerTridiagonal);
impl<T> LinearOperator<T> for Metric
where
    IntegerTridiagonal: LinearOperator<T>,
{
    fn dimension(&self) -> usize {
        self.0.diagonal.len()
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> Result<(), OperatorError> {
        self.0.apply(x, y)
    }
    fn metadata(&self) -> OperatorMetadata {
        <IntegerTridiagonal as LinearOperator<T>>::metadata(&self.0)
    }
}
impl SymmetricOperator<f64> for Metric {}
impl PositiveDefiniteMetric<f64> for Metric {}
impl SymmetricOperator<Float> for Metric {}
impl PositiveDefiniteMetric<Float> for Metric {}

fn norm_bound(x: &[Rational], round: Round) -> Float {
    let squared = x.iter().fold(Rational::new(), |mut sum, v| {
        sum += Rational::from(v * v);
        sum
    });
    let mut norm = Float::with_val_round(512, squared, round).0;
    norm.sqrt_round(round);
    norm
}
fn exact_backward_error_upper(
    a: &IntegerTridiagonal,
    b: &IntegerTridiagonal,
    value: &Rational,
    x: &[Rational],
) -> Float {
    let ax = a.exact_action(x);
    let bx = b.exact_action(x);
    let residual: Vec<_> = ax
        .iter()
        .zip(&bx)
        .map(|(av, bv)| av - Rational::from(value * bv))
        .collect();
    let numerator = norm_bound(&residual, Round::Up);
    let mut denominator = norm_bound(&bx, Round::Down);
    denominator.mul_assign_round(
        Float::with_val_round(512, value.clone().abs(), Round::Down).0,
        Round::Down,
    );
    denominator.add_assign_round(norm_bound(&ax, Round::Down), Round::Down);
    {
        let mut result = numerator;
        result.div_assign_round(denominator, Round::Up);
        result
    }
}
fn closed_form(n: usize, metric_diagonal: i32) -> Float {
    let angle = Float::with_val(512, Constant::Pi) / (n + 1);
    let c = angle.cos();
    let numerator = Float::with_val(512, 2) - Float::with_val(512, 2 * &c);
    if metric_diagonal == 1 {
        numerator
    } else {
        numerator / (Float::with_val(512, 4) + Float::with_val(512, 2 * c))
    }
}

#[test]
fn native_generalized_fresh_images_match_exact_integer_pencil_residuals() {
    for (n, metric_diagonal) in [(48, 1), (32, 4)] {
        let a = IntegerTridiagonal::uniform(n, 2, -1);
        let b = Metric(IntegerTridiagonal::uniform(
            n,
            metric_diagonal,
            if metric_diagonal == 1 { 0 } else { 1 },
        ));
        let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
        let cfg = GeneralizedExtremeConfigF64 {
            target: EigenTarget::AlgebraicSmallest,
            absolute_residual_tolerance: 1e-300,
            scaled_backward_error_tolerance: 1e-12,
            ritz_value_stability_tolerance: 1e-15,
            maximum_iterations: 30000,
            minimum_iterations: 2,
        };
        let r = MatrixFreeLobpcgF64.solve(&problem, &cfg).unwrap();
        let x: Vec<_> = r
            .eigenvector
            .iter()
            .map(|v| Rational::from_f64(*v).unwrap())
            .collect();
        let actual =
            exact_backward_error_upper(&a, &b.0, &Rational::from_f64(r.eigenvalue).unwrap(), &x);
        eprintln!(
            "native n={n} status={:?} iterations={} reported={} exact_upper={}",
            r.status, r.iterations, r.scaled_backward_error, actual
        );
        assert_eq!(r.status, ResultStatus::Converged);
        assert!(actual <= Float::with_val(512, cfg.scaled_backward_error_tolerance));
        assert!(
            (Float::with_val(512, r.eigenvalue) - closed_form(n, metric_diagonal)).abs()
                < Float::with_val(512, 1e-13)
        );
        assert_eq!(
            r.algorithm,
            "matrix_free_lobpcg_fresh_images_real_stationary_updates_f64_v4"
        );
    }
}

#[test]
fn hp_generalized_fresh_images_match_exact_integer_pencil_residuals() {
    for (n, metric_diagonal) in [(24, 4), (32, 1)] {
        let a = IntegerTridiagonal::uniform(n, 2, -1);
        let b = Metric(IntegerTridiagonal::uniform(
            n,
            metric_diagonal,
            if metric_diagonal == 1 { 0 } else { 1 },
        ));
        let problem = GeneralizedEigenProblem::new(&a, &b).unwrap();
        let cfg = GeneralizedExtremeConfigHp {
            target: EigenTarget::AlgebraicSmallest,
            precision_bits: 128,
            absolute_residual_tolerance: DecimalLiteral::new("1e-100").unwrap(),
            scaled_backward_error_tolerance: DecimalLiteral::new("1e-35").unwrap(),
            ritz_value_stability_tolerance: DecimalLiteral::new("1e-30").unwrap(),
            maximum_iterations: 30000,
            minimum_iterations: 2,
        };
        let r = MatrixFreeGeneralizedRayleighRitzHp
            .solve(&problem, &cfg)
            .unwrap();
        let x: Vec<_> = r
            .eigenvector
            .iter()
            .map(|v| v.to_rational().unwrap())
            .collect();
        let actual = exact_backward_error_upper(&a, &b.0, &r.eigenvalue.to_rational().unwrap(), &x);
        eprintln!(
            "HP n={n} status={:?} iterations={} reported={} exact_upper={}",
            r.status, r.iterations, r.scaled_backward_error, actual
        );
        assert_eq!(r.status, ResultStatus::Converged);
        assert!(actual <= Float::with_val(512, Float::parse("1e-35").unwrap()));
        assert!(
            (Float::with_val(512, &r.eigenvalue) - closed_form(n, metric_diagonal)).abs()
                < Float::with_val(512, Float::parse("1e-35").unwrap())
        );
        assert_eq!(
            r.algorithm,
            "matrix_free_generalized_b_orthogonal_rayleigh_ritz_fresh_images_rounding_margin_hp_v4"
        );
    }
}
