#![cfg(feature = "hp")]

//! Fresh, independently manufactured mathematical controls. These test exact
//! input models; passing them is not an all-domain accuracy certificate.
use rug::{Float, Rational};
use xc_numerics::{eigen, interval::*, linalg, mpfr_interval::MpfrInterval, quadrature};

fn q(n: i32, d: i32) -> Rational {
    Rational::from((n, d))
}

fn encloses(interval: &MpfrInterval, exact: &Rational) {
    interval.validate().unwrap();
    assert!(interval.lower().to_rational().unwrap() <= *exact);
    assert!(interval.upper().to_rational().unwrap() >= *exact);
}

#[test]
fn rational_oracles_for_directed_interval_arithmetic() {
    for p in [32, 64, 127, 256] {
        for a in [q(-101, 7), q(-1, 3), q(0, 1), q(2, 11), q(37, 5)] {
            for b in [q(-13, 17), q(1, 29), q(41, 3)] {
                let ai = MpfrInterval::from_rational(&a, p);
                let bi = MpfrInterval::from_rational(&b, p);
                encloses(&ai.add(&bi), &Rational::from(&a + &b));
                encloses(&ai.sub(&bi), &Rational::from(&a - &b));
                encloses(&ai.mul(&bi), &Rational::from(&a * &b));
                encloses(&ai.div(&bi).unwrap(), &Rational::from(&a / &b));
                encloses(&ai.with_precision(32).unwrap(), &a);
            }
        }
        for value in [q(0, 1), q(1, 7), q(2, 1), q(10001, 3)] {
            let root = MpfrInterval::from_rational(&value, p).sqrt().unwrap();
            let lo = root.lower().to_rational().unwrap();
            let hi = root.upper().to_rational().unwrap();
            assert!(Rational::from(&lo * &lo) <= value);
            assert!(Rational::from(&hi * &hi) >= value);
        }
    }
}

fn polynomial_from_roots(roots: &[Rational]) -> Vec<Rational> {
    let mut coefficients = vec![q(1, 1)];
    for root in roots {
        let mut next = vec![q(0, 1); coefficients.len() + 1];
        for (i, c) in coefficients.iter().enumerate() {
            next[i] -= Rational::from(c * root);
            next[i + 1] += c;
        }
        coefficients = next;
    }
    coefficients
}

fn complex_real(x: Rational) -> ComplexRational {
    ComplexRational {
        real: x,
        imaginary: q(0, 1),
    }
}

#[test]
fn prescribed_polynomial_roots_distinguish_counts_and_multiplicity() {
    let roots = [q(-2, 1), q(-2, 1), q(1, 2), q(3, 1)];
    let coefficients = polynomial_from_roots(&roots);
    let all = exact_sturm_root_count(&coefficients, q(-4, 1), q(4, 1)).unwrap();
    assert_eq!(all.distinct_real_roots, 3);
    assert!(!all.square_free);
    assert_eq!(
        exact_sturm_root_count(&coefficients, q(0, 1), q(1, 1))
            .unwrap()
            .distinct_real_roots,
        1
    );
    assert!(exact_sturm_root_count(&coefficients, q(-2, 1), q(4, 1)).is_err());
    let complex: Vec<_> = coefficients.into_iter().map(complex_real).collect();
    let rectangle = RationalContourRectangle::new(q(-4, 1), q(4, 1), q(-1, 1), q(1, 1)).unwrap();
    assert_eq!(
        certify_polynomial_zero_count_on_rectangle(&complex, rectangle, 20)
            .unwrap()
            .zero_count,
        4
    );
    // z^3+1/8 has three roots of modulus 1/2, all inside the unit circle.
    let cubic = [q(1, 8), q(0, 1), q(0, 1), q(1, 1)].map(complex_real);
    assert_eq!(
        certify_polynomial_zero_count_on_circle(&cubic, q(1, 1))
            .unwrap()
            .zero_count,
        3
    );
}

#[test]
fn certified_sturm_counts_match_exact_manufactured_spectra() {
    for p in [64, 127, 256] {
        // Independent factorization: det(T-tI)=(2-t)*((2-t)^2-2).
        let d = vec![Float::with_val(p, 2); 3];
        let e = vec![Float::with_val(p, 1); 2];
        for (threshold, expected) in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)] {
            assert_eq!(
                eigen::tridiag_sturm_count_below_hp(&d, &e, &Float::with_val(p, threshold), p)
                    .unwrap(),
                expected
            );
        }
        // Three independent blocks, including a repeated exact eigenvalue.
        let d: Vec<_> = [-3, 0, 0, 7].map(|x| Float::with_val(p, x)).into();
        let e = vec![Float::with_val(p, 0); 3];
        assert_eq!(
            eigen::tridiag_sturm_count_below_hp(&d, &e, &Float::with_val(p, 0), p).unwrap(),
            1
        );
        let spectrum = eigen::tridiag_selected_eigenvalues_hp(
            &d,
            &e,
            0,
            3,
            &(Float::with_val(p, 1) >> 30),
            200,
            p,
        )
        .unwrap();
        for (enclosure, exact) in spectrum.enclosures.iter().zip([-3, 0, 0, 7]) {
            assert!(enclosure.lower <= exact && enclosure.upper >= exact);
            assert!(
                enclosure.lower_count <= enclosure.index && enclosure.index < enclosure.upper_count
            );
        }
    }
}

#[test]
fn dense_spectrum_matches_prescribed_rational_orthogonal_conjugation() {
    // Q=I-J/2 is exactly orthogonal in dimension four. A=Q D Q^T
    // is dyadic and has the prescribed spectrum without using any eigensolver.
    let lambda = [q(-3, 1), q(1, 2), q(5, 1), q(11, 1)];
    let matrix: Vec<_> = (0..16)
        .map(|ij| {
            let (i, j) = (ij / 4, ij % 4);
            (0..4)
                .map(|k| {
                    let qi = q(if i == k { 1 } else { -1 }, 2);
                    let qj = q(if j == k { 1 } else { -1 }, 2);
                    qi * &lambda[k] * qj
                })
                .fold(q(0, 1), |a, b| a + b)
        })
        .collect();
    for p in [64, 128, 257] {
        let a: Vec<_> = matrix.iter().map(|x| Float::with_val(p, x)).collect();
        let qr = eigen::dense_symmetric_eigenvalues_hp_stable(&a, 4, p).unwrap();
        let jacobi = eigen::dense_symmetric_eigendecomposition_jacobi_hp(&a, 4, p, 80).unwrap();
        let tolerance = Float::with_val(p, 1) >> (p - 24);
        for values in [&qr, &jacobi.eigenvalues] {
            for (actual, expected) in values.iter().zip(&lambda) {
                assert!(
                    Float::with_val(p, actual - Float::with_val(p, expected)).abs() < tolerance
                );
            }
        }
        let (d, e, q) = eigen::householder_tridiag_hp_stable(&a, 4, p).unwrap();
        // Replay AQ-QT and Q^TQ-I using exact rationals of returned points.
        let exact_q: Vec<_> = q.iter().map(|x| x.to_rational().unwrap()).collect();
        let mut largest = Rational::new();
        for i in 0..4 {
            for j in 0..4 {
                let aq = (0..4)
                    .map(|k| Rational::from(&matrix[i * 4 + k] * &exact_q[k * 4 + j]))
                    .fold(Rational::new(), |a, b| a + b);
                let qt = (0..4)
                    .map(|k| {
                        let t = if k == j {
                            d[k].to_rational().unwrap()
                        } else if k + 1 == j {
                            e[k].to_rational().unwrap()
                        } else if j + 1 == k {
                            e[j].to_rational().unwrap()
                        } else {
                            Rational::new()
                        };
                        &exact_q[i * 4 + k] * t
                    })
                    .fold(Rational::new(), |a, b| a + b);
                largest = largest.max((aq - qt).abs());
            }
        }
        assert!(largest < tolerance.to_rational().unwrap());
    }
}

#[test]
fn exact_manufactured_rhs_checks_lu_pivot_sequences() {
    let p = 192;
    for case in 0..24_i32 {
        let diagonal: Vec<i32> = (0..7).map(|i| (case + 3 * i) % 7 - 3).collect();
        let lower: Vec<i32> = (0..6).map(|i| 5 + (case + i) % 7).collect();
        let upper: Vec<i32> = (0..6).map(|i| 1 + (2 * case + i) % 5).collect();
        let solution: Vec<i32> = (0..7).map(|i| 2 * i - 5).collect();
        let mut a = vec![0_i32; 49];
        for i in 0..7 {
            a[i * 7 + i] = diagonal[i];
            if i < 6 {
                a[(i + 1) * 7 + i] = lower[i];
                a[i * 7 + i + 1] = upper[i];
            }
        }
        let rhs: Vec<_> = (0..7)
            .map(|i| Float::with_val(p, (0..7).map(|j| a[i * 7 + j] * solution[j]).sum::<i32>()))
            .collect();
        let hp = |v: &[i32]| v.iter().map(|x| Float::with_val(p, *x)).collect::<Vec<_>>();
        let band = linalg::tridiag_lu_factor_hp(&hp(&lower), &hp(&diagonal), &hp(&upper), p);
        let dense = linalg::lu_factor(&hp(&a), 7);
        if let (Ok(band), Ok(dense)) = (band, dense) {
            for solved in [
                linalg::tridiag_lu_solve_hp(&band, &rhs, p).unwrap(),
                linalg::try_lu_solve(&dense, &rhs, 7, p).unwrap(),
            ] {
                for (x, expected) in solved.iter().zip(&solution) {
                    assert!(
                        Float::with_val(p, x - *expected).abs() < (Float::with_val(p, 1) >> 140)
                    );
                }
            }
        }
    }
}

#[test]
fn quadrature_moments_use_exact_integrals_as_oracles() {
    for p in [64, 128, 257] {
        for n in [1, 2, 3, 5, 8, 16] {
            let (nodes, weights) =
                quadrature::try_gauss_legendre_nodes(n, p, quadrature::CacheMode::Off).unwrap();
            for degree in 0..2 * n {
                let mut moment = Rational::new();
                for (x, w) in nodes.iter().zip(&weights) {
                    let x = x.to_rational().unwrap();
                    let mut term = w.to_rational().unwrap();
                    for _ in 0..degree {
                        term *= &x;
                    }
                    moment += term;
                }
                let expected = if degree % 2 == 0 {
                    Rational::from((2, degree + 1))
                } else {
                    Rational::new()
                };
                assert!(
                    (moment - expected).abs()
                        < (Float::with_val(p, 1) >> (p - 12)).to_rational().unwrap(),
                    "p={p}, n={n}, degree={degree}"
                );
            }
        }
    }
}

#[test]
fn actual_odd_gl_center_points_are_recorded() {
    for p in [64, 128, 257] {
        for n in [3, 5, 7, 9, 11, 17, 31, 63] {
            let (nodes, _) =
                quadrature::try_gauss_legendre_nodes(n, p, quadrature::CacheMode::Off).unwrap();
            let center = &nodes[n / 2];
            println!(
                "GL_CENTER p={p} n={n} exact_zero={} stored={center}",
                center.is_zero()
            );
            assert!(center.clone().abs() < (Float::with_val(p, 1) >> (p - 8)));
        }
    }
}
