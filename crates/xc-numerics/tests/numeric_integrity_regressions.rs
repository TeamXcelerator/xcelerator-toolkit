//! Independent algebraic oracles. Expectations are algebraic identities, not replayed toolkit algorithms.
#![cfg(feature = "hp")]

use rug::{ops::Pow, Float, Rational};
use xc_numerics::{eigen, interval::*, linalg, mpfr_interval::MpfrInterval, prefix, quadrature};

fn hp(p: u32, value: &Rational) -> Float {
    Float::with_val(p, value)
}
fn q(n: i32, d: i32) -> Rational {
    Rational::from((n, d))
}
fn close(actual: &Float, expected: &Rational, p: u32, bits: u32) {
    let error = Float::with_val(p + 32, actual - hp(p + 32, expected)).abs();
    let scale = hp(p + 32, expected).abs().max(&Float::with_val(p + 32, 1));
    assert!(
        error <= scale * (Float::with_val(p + 32, 1) >> bits),
        "{actual} vs {expected}: {error}"
    );
}

#[test]
fn rational_and_mpfr_operations_contain_exact_cartesian_endpoints() {
    let points = [q(-13, 7), q(-1, 3), q(0, 1), q(1, 11), q(5, 3), q(37, 2)];
    for p in [2, 7, 32, 97] {
        for a in &points {
            for b in &points {
                let x = MpfrInterval::from_rational(a, p);
                let y = MpfrInterval::from_rational(b, p);
                for (result, exact) in [
                    (x.add(&y), a.clone() + b),
                    (x.sub(&y), a.clone() - b),
                    (x.mul(&y), a.clone() * b),
                ] {
                    assert!(
                        result.to_rational_interval().contains(&exact),
                        "p={p}, a={a}, b={b}"
                    );
                }
                if b != &0 {
                    assert!(x
                        .div(&y)
                        .unwrap()
                        .to_rational_interval()
                        .contains(&(a.clone() / b)));
                }
                let hull = RationalInterval::hull(a.clone(), b.clone());
                let square = hull.square();
                assert!(square.contains(&(a.clone() * a)) && square.contains(&(b.clone() * b)));
                if hull.contains_zero() {
                    assert_eq!(square.lower(), &q(0, 1));
                }
            }
        }
    }
}

fn real_polynomial_from_roots(roots: &[Rational]) -> Vec<Rational> {
    let mut coefficients = vec![q(1, 1)];
    for root in roots {
        let mut next = vec![q(0, 1); coefficients.len() + 1];
        for (i, c) in coefficients.iter().enumerate() {
            next[i] -= c.clone() * root;
            next[i + 1] += c;
        }
        coefficients = next;
    }
    coefficients
}

#[test]
fn exact_sturm_counts_known_roots_and_multiplicities() {
    let roots = [q(-5, 2), q(-1, 3), q(1, 7), q(2, 1), q(7, 2)];
    let coefficients = real_polynomial_from_roots(&roots);
    for lower in -4..4 {
        for upper in lower + 1..=5 {
            let (lo, hi) = (q(2 * lower - 1, 2), q(2 * upper - 1, 2));
            if roots.contains(&lo) || roots.contains(&hi) {
                continue;
            }
            let expected = roots.iter().filter(|r| **r > lo && **r < hi).count();
            let count = exact_sturm_root_count(&coefficients, lo, hi).unwrap();
            assert_eq!(count.distinct_real_roots, expected);
            assert!(count.square_free);
        }
    }
    let mut repeated = roots.to_vec();
    repeated.extend([roots[0].clone(), roots[3].clone()]);
    let count =
        exact_sturm_root_count(&real_polynomial_from_roots(&repeated), q(-4, 1), q(4, 1)).unwrap();
    assert_eq!(count.distinct_real_roots, roots.len());
    assert!(!count.square_free);
    let boxes =
        exact_sturm_isolate_roots(&coefficients, q(-4, 1), q(4, 1), q(1, 1024), 32).unwrap();
    assert_eq!(boxes.len(), roots.len());
    for (enclosure, root) in boxes.iter().zip(&roots) {
        assert!(enclosure.contains(root));
        assert!(enclosure.width() <= q(1, 1024));
    }
}

#[test]
fn contour_counts_use_known_factored_polynomials() {
    // z^3-z has three roots; the narrow rectangle contains only zero.
    let coefficients = [-0, -1, 0, 1]
        .into_iter()
        .map(|v| ComplexRational {
            real: q(v, 1),
            imaginary: q(0, 1),
        })
        .collect::<Vec<_>>();
    for (width, count) in [(q(1, 2), 1), (q(3, 2), 3)] {
        let rectangle =
            RationalContourRectangle::new(-width.clone(), width, q(-1, 2), q(1, 2)).unwrap();
        let result =
            certify_polynomial_zero_count_on_rectangle(&coefficients, rectangle, 16).unwrap();
        assert!(result.rigorous && result.boundary_excludes_zero);
        assert_eq!(result.zero_count, count);
        for cell in result.cells {
            assert!(cell.image_enclosure.excludes_zero());
        }
    }
}

#[test]
fn dense_spectra_and_prefix_moments_match_exact_hadamard_similarity() {
    let p = 256;
    let h = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]];
    for lambdas in [
        vec![q(1, 16), q(1, 1), q(3, 1), q(10, 1)],
        vec![q(-9, 1), q(-1, 4), q(1, 8), q(7, 1)],
    ] {
        // H/2 is exactly orthogonal; A=(H/2) diag(lambda) (H/2)^T.
        let mut matrix = vec![q(0, 1); 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    matrix[i * 4 + j] += lambdas[k].clone() * q(h[i][k] * h[j][k], 4);
                }
            }
        }
        let a = matrix.iter().map(|v| hp(p, v)).collect::<Vec<_>>();
        let qr = eigen::dense_symmetric_eigenvalues_hp_stable(&a, 4, p).unwrap();
        let jacobi = eigen::dense_symmetric_eigendecomposition_jacobi_hp(&a, 4, p, 100).unwrap();
        for (i, expected) in lambdas.iter().enumerate() {
            close(&qr[i], expected, p, 200);
            close(&jacobi.eigenvalues[i], expected, p, 200);
            for row in 0..4 {
                let mut residual = Float::with_val(p, 0);
                for column in 0..4 {
                    residual += Float::with_val(
                        p,
                        &a[row * 4 + column] * &jacobi.eigenvectors[column * 4 + i],
                    );
                }
                residual -= Float::with_val(p, hp(p, expected) * &jacobi.eigenvectors[row * 4 + i]);
                assert!(residual.abs() < Float::with_val(p, 1) >> 190u32);
            }
        }
        if lambdas[0] > 0 {
            let report = prefix::analyze_prefixes_with_policy(
                &a,
                4,
                p,
                32,
                &[4],
                &prefix::PrefixDiagnosticPolicy::full(),
            )
            .unwrap();
            let row = report.rows.last().unwrap();
            for (power, encoded) in [
                (1, &row.inverse_trace),
                (2, &row.inverse_square_trace),
                (
                    3,
                    &row.third_inverse_moment
                        .as_ref()
                        .unwrap()
                        .inverse_cube_trace,
                ),
            ] {
                let expected = lambdas
                    .iter()
                    .map(|v| q(1, 1) / v.clone().pow(power))
                    .fold(q(0, 1), |a, b| a + b);
                close(
                    &Float::with_val(p, Float::parse(encoded).unwrap()),
                    &expected,
                    p,
                    190,
                );
            }
        }
    }
}

#[test]
fn repeated_pivots_recover_manufactured_exact_solutions() {
    let p = 192;
    for n in 2..=12 {
        // Tiny diagonal relative to lower entries forces successive adjacent pivots.
        let lo = (0..n - 1).map(|i| q((i + 2) as i32, 1)).collect::<Vec<_>>();
        let di = (0..n).map(|i| q((i + 1) as i32, 32)).collect::<Vec<_>>();
        let up = (0..n - 1)
            .map(|i| q(-((i + 1) as i32), 2))
            .collect::<Vec<_>>();
        let solution = (0..n).map(|i| q(2 * i as i32 - 3, 4)).collect::<Vec<_>>();
        let mut matrix = vec![q(0, 1); n * n];
        for i in 0..n {
            matrix[i * n + i] = di[i].clone();
            if i + 1 < n {
                matrix[(i + 1) * n + i] = lo[i].clone();
                matrix[i * n + i + 1] = up[i].clone();
            }
        }
        let rhs = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| matrix[i * n + j].clone() * &solution[j])
                    .fold(q(0, 1), |a, b| a + b)
            })
            .collect::<Vec<_>>();
        let convert = |v: &[Rational]| v.iter().map(|x| hp(p, x)).collect::<Vec<_>>();
        let factors =
            linalg::tridiag_lu_factor_hp(&convert(&lo), &convert(&di), &convert(&up), p).unwrap();
        let got = linalg::tridiag_lu_solve_hp(&factors, &convert(&rhs), p).unwrap();
        let dense = linalg::lu_factor(&convert(&matrix), n).unwrap();
        let got_dense = linalg::try_lu_solve(&dense, &convert(&rhs), n, p).unwrap();
        for i in 0..n {
            close(&got[i], &solution[i], p, 145);
            close(&got_dense[i], &solution[i], p, 145);
        }
    }
}

#[test]
fn hp_sturm_strict_thresholds_match_block_eigenvalues() {
    let p = 160;
    for exponent in [-200, 0, 200] {
        let scale = Float::with_val(p, 1) << exponent;
        // Two 2x2 blocks with spectra {-3,1} and {1,5}, hence a repeated 1.
        let d = [-1, -1, 3, 3].map(|v| Float::with_val(p, v) * &scale);
        let e = [2, 0, 2].map(|v| Float::with_val(p, v) * &scale);
        for (threshold, expected) in [(-4, 0), (-3, 0), (-2, 1), (1, 1), (2, 3), (5, 3), (6, 4)] {
            assert_eq!(
                eigen::tridiag_sturm_count_below_hp(
                    &d,
                    &e,
                    &(Float::with_val(p, threshold) * &scale),
                    p
                )
                .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn gauss_legendre_all_exact_degree_moments_at_multiple_precisions() {
    for p in [64, 128, 256] {
        for n in [1, 2, 3, 5, 8, 12] {
            let (nodes, weights) =
                quadrature::try_gauss_legendre_nodes(n, p, quadrature::CacheMode::Off).unwrap();
            for degree in 0..2 * n {
                // Evaluate the moment in wider arithmetic, so p-bit accumulation cannot hide rule error.
                let guard = p + 64;
                let mut moment = Float::with_val(guard, 0);
                for (x, w) in nodes.iter().zip(&weights) {
                    moment += Float::with_val(guard, x).pow(degree as u32) * w;
                }
                let expected = if degree % 2 == 0 {
                    q(2, (degree + 1) as i32)
                } else {
                    q(0, 1)
                };
                close(&moment, &expected, guard, p - 10);
            }
        }
    }
}

#[test]
fn managed_quadrature_disabled_computes_without_requiring_a_manifest() {
    // Contract regression: Disabled promises fresh computation without cache I/O.
    let result = quadrature::gauss_legendre_nodes_via_cache(
        2,
        128,
        quadrature::QuadratureCacheRequest {
            resolver: None,
            reference_resolver: None,
            acceptance: None,
            ordered_overlays: vec!["workstation".into()],
            mode: xc_cache::ArtifactExecutionCacheMode::Disabled,
            write_on_miss: false,
            write_visibility: xc_cache::CacheVisibility::Local,
            requested_assurance: xc_core::AssuranceLevel::Computed,
            certification_failure_policy:
                xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
            production_sink: None,
        },
    );
    let rule = result.expect("valid Disabled quadrature must compute");
    assert!(rule.artifact_manifest.is_none());
    assert_eq!(rule.nodes.len(), 2);
    assert_eq!(rule.weights.len(), 2);
}
