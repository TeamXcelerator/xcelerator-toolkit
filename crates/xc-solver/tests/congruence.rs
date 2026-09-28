#![cfg(feature = "hp-reference")]
use rug::Float;
use xc_core::{DecimalLiteral, EigenTarget, ResultStatus};
use xc_solver::{
    solve_dense_generalized_whitening_hp, DenseGeneralizedProblemHp, GeneralizedExtremeConfigHp,
};
#[test]
fn dense_generalized_extremes_match_exact_congruence_spectrum_across_scales() {
    let d = [-2, 1, 4, 7];
    let c = [[1, 2, 0, 1], [0, 1, 1, 0], [0, 0, 1, 2], [0, 0, 0, 1]];
    let mut cases = 0;
    for p in [96, 192] {
        for exponent in [-300, 0, 300] {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for i in 0..4 {
                for j in 0..4 {
                    let mut av =
                        Float::with_val(p, (0..4).map(|k| c[k][i] * d[k] * c[k][j]).sum::<i32>());
                    let mut bv = Float::with_val(p, (0..4).map(|k| c[k][i] * c[k][j]).sum::<i32>());
                    av <<= exponent;
                    bv <<= exponent;
                    a.push(av);
                    b.push(bv);
                }
            }
            // A=C^T D C and B=C^T C, so det(A-lambda B)=det(C)^2 product(d-lambda).
            let problem = DenseGeneralizedProblemHp::new(&a, &b, 4).unwrap();
            for (target, expected) in [
                (EigenTarget::AlgebraicSmallest, -2),
                (EigenTarget::AlgebraicLargest, 7),
            ] {
                let lit = || DecimalLiteral::new("1e-20").unwrap();
                let config = GeneralizedExtremeConfigHp {
                    target,
                    precision_bits: p,
                    absolute_residual_tolerance: lit(),
                    scaled_backward_error_tolerance: lit(),
                    ritz_value_stability_tolerance: lit(),
                    maximum_iterations: 128,
                    minimum_iterations: 2,
                };
                let report = solve_dense_generalized_whitening_hp(&problem, &config).unwrap();
                assert_eq!(report.status, ResultStatus::Converged);
                let error = (report.eigenvalue - Float::with_val(p, expected)).abs();
                assert!(
                    error < Float::with_val(p, Float::parse("1e-18").unwrap()),
                    "p={p} exponent={exponent} error={error}"
                );
                cases += 1;
            }
        }
    }
    eprintln!("{cases} manufactured generalized congruence extremes agree");
}
