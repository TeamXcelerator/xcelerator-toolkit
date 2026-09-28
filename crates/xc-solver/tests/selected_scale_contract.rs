#![cfg(feature = "hp-reference")]
use rug::{Float, Rational};
use xc_core::{DecimalLiteral, PrecisionEscalation, PrecisionPolicy, ResultStatus};
use xc_numerics::eigen::TridiagSolver;
use xc_solver::*;

fn verify_tiny_pair(result: &HpSelectedTridiagonalEigenpairs, diagonal: &[Float], off: &[Float]) {
    assert_eq!(result.items.len(), 1);
    let HpSelectedTridiagonalItem::SimpleEigenpair(pair) = &result.items[0] else {
        panic!("the exact two-dimensional spectrum has two separated simple eigenvalues");
    };
    assert_eq!(pair.enclosure.index, 1);
    let scale = Float::with_val(512, &diagonal[1]);
    // Independently, [-s,b;b,s] has eigenvalues +/-sqrt(s^2+b^2).
    let mut square = Float::with_val(512, &scale * &scale);
    square += Float::with_val(512, &off[0] * &off[0]);
    let expected = square.sqrt();
    let relative_error = (Float::with_val(512, &pair.eigenvalue) - &expected).abs() / &expected;
    assert!(relative_error < Float::with_val(512, 1) >> 100u32);
    let x: Vec<_> = pair
        .eigenvector
        .iter()
        .map(|v| v.to_rational().unwrap())
        .collect();
    let lambda = pair.eigenvalue.to_rational().unwrap();
    let b = off[0].to_rational().unwrap();
    let mut residual_sq = Rational::new();
    let mut norm_sq = Rational::new();
    for i in 0..2 {
        let mut residual = (diagonal[i].to_rational().unwrap() - &lambda) * &x[i];
        residual += Rational::from(&b * &x[1 - i]);
        residual_sq += Rational::from(&residual * &residual);
        norm_sq += Rational::from(&x[i] * &x[i]);
    }
    assert!(
        (norm_sq.clone() - rug::Rational::from(1)).abs()
            < Rational::from((1, rug::Integer::from(1) << 110))
    );
    // The exact gap is at least 2s, so this checks the returned-vector angle
    // scale independently of the very loose caller absolute value tolerance.
    let gap_lower = diagonal[1].to_rational().unwrap() - diagonal[0].to_rational().unwrap();
    let limit = Rational::from((1, rug::Integer::from(1) << 200));
    assert!(residual_sq < gap_lower.clone() * gap_lower * norm_sq * limit);
}

#[test]
fn selected_adapter_does_not_promote_an_inaccurate_tiny_state() {
    let p = 128;
    for exponent in [400_i32, 800_i32] {
        let scale = Float::with_val(p, 1) >> exponent;
        let diagonal = [-scale.clone(), scale.clone()];
        for off_value in [Float::with_val(p, 0), scale.clone()] {
            let off = [off_value];
            let problem = TridiagonalProblemHp::new(&diagonal, &off).unwrap();
            let vector_options = TridiagEigvecOptions {
                solver: TridiagSolver::BandedInterleaved,
                max_steps: 20,
                early_termination: false,
            };
            let request = HpSelectedTridiagonalEigenpairOptions {
                first_index: 1,
                last_index: 1,
                absolute_tolerance: Float::with_val(p, 1) >> 70_i32,
                maximum_bisection_iterations: 1600,
                eigenvector_options: vector_options,
                precision_bits: p,
            };
            // Keep the original +/-2^-400 diagonal fixture and its loose value
            // tolerance. Fresh exact residual/gap evidence permits its repaired
            // recovery; the additional coupled spectrum is +/-sqrt(2)*scale.
            let fixed = solve_tridiagonal_selected_eigenpairs_hp(&problem, &request).unwrap();
            verify_tiny_pair(&fixed, &diagonal, &off);
            let adaptive = solve_tridiagonal_selected_eigenpairs_adaptive_hp(
                &problem,
                &HpAdaptiveSelectedTridiagonalOptions {
                    first_index: 1,
                    last_index: 1,
                    absolute_tolerance: DecimalLiteral::new("1e-21").unwrap(),
                    maximum_bisection_iterations: 1600,
                    eigenvector_options: vector_options,
                    precision: PrecisionPolicy {
                        initial_bits: p,
                        maximum_bits: p,
                        guard_bits: 0,
                        escalation: PrecisionEscalation::AddBits(64),
                    },
                },
            )
            .unwrap();
            let HpAdaptiveSelectedTridiagonalResult::Converged { result, attempts } = adaptive
            else {
                panic!("independently accurate tiny pairs should converge");
            };
            verify_tiny_pair(&result, &diagonal, &off);
            assert_eq!(attempts.len(), 1);
            assert_eq!(attempts[0].status, ResultStatus::Converged);
            assert_eq!(attempts[0].precision_bits, p);
            assert_eq!(attempts[0].selected_items, 1);
        }
    }
}

#[test]
fn selected_adapter_preserves_explicit_bisection_work_exhaustion() {
    let p = 128;
    let scale = Float::with_val(p, 1) >> 400_i32;
    let diagonal = [-scale.clone(), scale.clone()];
    let off = [scale.clone()];
    let problem = TridiagonalProblemHp::new(&diagonal, &off).unwrap();
    let request = HpSelectedTridiagonalEigenpairOptions {
        first_index: 1,
        last_index: 1,
        absolute_tolerance: scale >> 100u32,
        maximum_bisection_iterations: 1,
        eigenvector_options: TridiagEigvecOptions {
            solver: TridiagSolver::BandedInterleaved,
            max_steps: 20,
            early_termination: false,
        },
        precision_bits: p,
    };
    // The irrational sqrt(2) root cannot reach this width in one bisection.
    assert!(matches!(
        solve_tridiagonal_selected_eigenpairs_hp(&problem, &request),
        Err(SolverError::IterationBudgetExhausted(_))
    ));
}
