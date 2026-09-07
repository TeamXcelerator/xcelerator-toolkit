#![cfg(feature = "hp")]
use rug::{ops::Pow, Float};
use xc_numerics::linalg::*;
fn f(x: i32) -> Float {
    Float::with_val(256, x)
}
fn check(d: &[i32], l: &[i32], u: &[i32]) -> bool {
    let n = d.len();
    let (mut before, mut determinant) = (1_i64, i64::from(d[0]));
    for i in 1..n {
        let next =
            i64::from(d[i]) * determinant - i64::from(l[i - 1]) * i64::from(u[i - 1]) * before;
        before = determinant;
        determinant = next;
    }
    if determinant == 0 {
        return false;
    }

    let diag = d.iter().map(|&x| f(x)).collect::<Vec<_>>();
    let lower = l.iter().map(|&x| f(x)).collect::<Vec<_>>();
    let upper = u.iter().map(|&x| f(x)).collect::<Vec<_>>();
    let mut a = vec![f(0); n * n];
    let exact = (0..n).map(|i| f((i as i32) % 5 - 2)).collect::<Vec<_>>();
    for i in 0..n {
        a[i * n + i] = diag[i].clone();
        if i + 1 < n {
            a[i * n + i + 1] = upper[i].clone();
            a[(i + 1) * n + i] = lower[i].clone();
        }
    }
    let b = (0..n)
        .map(|i| {
            let mut v = f(0);
            for j in 0..n {
                v += Float::with_val(256, &a[i * n + j] * &exact[j]);
            }
            v
        })
        .collect::<Vec<_>>();
    let Ok(factors) = tridiag_lu_factor_hp(&lower, &diag, &upper, 256) else {
        return false;
    };
    if factors.u_d.iter().any(Float::is_zero) {
        return false;
    }
    let x = tridiag_lu_solve_pivoted_hp(&factors, &b, 256).unwrap();
    let dense = lu_solve(&lu_factor(&a, n).unwrap(), &b, n, 256);
    let tol = Float::with_val(256, 2).pow(-180);
    for i in 0..n {
        assert!(
            Float::with_val(256, &x[i] - &exact[i]).abs() < tol,
            "solution case {d:?} {l:?} {u:?}"
        );
        assert!(Float::with_val(256, &x[i] - &dense[i]).abs() < tol);
    }
    true
}
#[test]
fn later_pivot_uses_interleaved_rhs_and_preserves_the_old_route() {
    let factors =
        tridiag_lu_factor_hp(&[f(1), f(1)], &[f(2), f(1), f(1)], &[f(1), f(1)], 256).unwrap();
    let b = vec![f(3), f(3), f(2)];
    let correct = tridiag_lu_solve_pivoted_hp(&factors, &b, 256).unwrap();
    assert_eq!(correct, vec![f(1), f(1), f(1)]);
    let historical = tridiag_lu_solve_hp(&factors, &b, 256).unwrap();
    assert_ne!(historical, correct);
}
#[test]
fn independent_dense_and_known_solution_checks_cover_pivot_patterns() {
    let mut tested = 0;
    for seed in 0..120 {
        let n = 2 + seed % 7;
        let d = (0..n)
            .map(|i| ((i * 11 + seed * 7) % 9) - 4)
            .collect::<Vec<_>>();
        let l = (0..n - 1)
            .map(|i| ((i * 5 + seed * 3) % 7) - 3)
            .collect::<Vec<_>>();
        let u = (0..n - 1)
            .map(|i| ((i * 3 + seed * 5) % 7) - 3)
            .collect::<Vec<_>>();
        tested += usize::from(check(&d, &l, &u));
    }
    assert!(tested > 80);
}
#[test]
fn invalid_factor_shapes_permutations_and_zero_pivots_fail() {
    let mut factors =
        tridiag_lu_factor_hp(&[f(1), f(1)], &[f(2), f(1), f(1)], &[f(1), f(1)], 256).unwrap();
    factors.perm = vec![2, 0, 1];
    assert!(tridiag_lu_solve_pivoted_hp(&factors, &vec![f(1); 3], 256).is_err());
    factors.perm = vec![0, 2, 1];
    factors.u_d[2] = f(0);
    assert!(tridiag_lu_solve_pivoted_hp(&factors, &vec![f(1); 3], 256).is_err());
    factors.u_d.clear();
    assert!(tridiag_lu_solve_pivoted_hp(&factors, &[], 256).is_err());
}
#[test]
fn thousand_dimension_solve_needs_only_banded_storage() {
    let n = 1000;
    let diag = vec![f(4); n];
    let off = vec![f(-1); n - 1];
    let factors = tridiag_lu_factor_hp(&off, &diag, &off, 256).unwrap();
    let mut b = vec![f(2); n];
    b[0] = f(3);
    b[n - 1] = f(3);
    let x = tridiag_lu_solve_pivoted_hp(&factors, &b, 256).unwrap();
    let tol = Float::with_val(256, 2).pow(-220);
    assert!(x.iter().all(|v| Float::with_val(256, v - 1).abs() < tol));
}

#[test]
fn explicit_eigenvector_route_matches_dense_and_has_distinct_identity() {
    use xc_numerics::eigen::*;
    let diag = vec![f(2), f(1), f(1)];
    let off = vec![f(1), f(1)];
    let eig = tridiag_eigenvalues_hp(&diag, &off, 256).unwrap();
    for lambda in &eig {
        let mut vectors = vec![];
        for solver in [TridiagSolver::Dense, TridiagSolver::BandedInterleaved] {
            vectors.push(
                tridiag_eigenvector_for_value_hp(
                    &diag,
                    &off,
                    lambda,
                    256,
                    TridiagEigvecOptions {
                        solver,
                        max_steps: 20,
                        early_termination: false,
                    },
                )
                .unwrap(),
            );
        }
        let mut overlap = f(0);
        for (a, b) in vectors[0].iter().zip(&vectors[1]) {
            overlap += Float::with_val(256, a * b);
        }
        assert!(Float::with_val(256, overlap.abs() - 1).abs() < Float::with_val(256, 2).pow(-180));
    }
    assert_ne!(
        TridiagSolver::Banded.semantics_id(),
        TridiagSolver::BandedInterleaved.semantics_id()
    );
}
