#![cfg(feature = "hp")]
use rug::{Float, Rational};
use xc_spectral::ccm::hp::*;

#[test]
fn every_state_criterion_matches_the_discrete_argmin_definition() {
    let targets = [
        CcmStateTarget::AlgebraicGround,
        CcmStateTarget::SmallestPositive,
        CcmStateTarget::NearestZero,
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Even,
            criterion: CcmStateCriterion::AlgebraicGround,
        },
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Even,
            criterion: CcmStateCriterion::SmallestPositive,
        },
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Even,
            criterion: CcmStateCriterion::NearestZero,
        },
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Odd,
            criterion: CcmStateCriterion::AlgebraicGround,
        },
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Odd,
            criterion: CcmStateCriterion::SmallestPositive,
        },
        CcmStateTarget::ParityRestricted {
            parity: CcmParity::Odd,
            criterion: CcmStateCriterion::NearestZero,
        },
    ];
    let mut checks = 0;
    for a in -2i32..=2 {
        for b in -2i32..=2 {
            for c in -2i32..=2 {
                for mask in 0..8 {
                    let values = [a, b, c];
                    let candidates = values
                        .iter()
                        .enumerate()
                        .map(|(j, v)| CcmStateCandidateHp {
                            algebraic_index: 11 + j,
                            eigenvalue: Float::with_val(128, *v),
                            eigenvector: vec![Float::with_val(128, j + 1), Float::with_val(128, 1)],
                            parity: if mask & (1 << j) == 0 {
                                CcmParity::Even
                            } else {
                                CcmParity::Odd
                            },
                        })
                        .collect::<Vec<_>>();
                    for target in targets {
                        let (parity, criterion) = match target {
                            CcmStateTarget::AlgebraicGround => (None, 0),
                            CcmStateTarget::SmallestPositive => (None, 1),
                            CcmStateTarget::NearestZero => (None, 2),
                            CcmStateTarget::ParityRestricted { parity, criterion } => (
                                Some(parity),
                                match criterion {
                                    CcmStateCriterion::AlgebraicGround => 0,
                                    CcmStateCriterion::SmallestPositive => 1,
                                    CcmStateCriterion::NearestZero => 2,
                                },
                            ),
                        };
                        let mut scores = (0..3)
                            .filter(|&j| {
                                parity.is_none_or(|p| candidates[j].parity == p)
                                    && (criterion != 1 || values[j] > 0)
                            })
                            .map(|j| {
                                (
                                    if criterion == 2 {
                                        values[j].abs()
                                    } else {
                                        values[j]
                                    },
                                    j,
                                )
                            })
                            .collect::<Vec<_>>();
                        scores.sort();
                        let unique =
                            !scores.is_empty() && (scores.len() == 1 || scores[0].0 != scores[1].0);
                        let selected = select_ccm_state_hp(target, &candidates);
                        assert_eq!(selected.is_ok(), unique);
                        if unique {
                            let result = selected.unwrap();
                            let expected = &candidates[scores[0].1];
                            assert_eq!(result.algebraic_index, expected.algebraic_index);
                            assert_eq!(result.eigenvalue, expected.eigenvalue);
                            assert_eq!(result.eigenvector, expected.eigenvector);
                            assert_eq!(result.parity, expected.parity);
                        }
                        checks += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checks, 9000);
}

#[test]
fn parity_embeddings_obey_reflection_and_requested_precision() {
    let mut checks = 0;
    for input_precision in [64, 128, 256] {
        for output_precision in [64, 128, 256] {
            for n in [0usize, 1, 2, 7, 16] {
                for shift in [-500_000_000i32, 0, 500_000_000] {
                    let even = (0..=n)
                        .map(|j| {
                            (Float::with_val(input_precision, (j as i32 % 7) - 3) / 7u32) << shift
                        })
                        .collect::<Vec<_>>();
                    let odd = even.iter().take(n).cloned().collect::<Vec<_>>();
                    let e = expand_even_sector_vector(&even, n, output_precision);
                    let o = expand_odd_sector_vector(&odd, n, output_precision);
                    assert_eq!(e.len(), 2 * n + 1);
                    assert_eq!(o.len(), 2 * n + 1);
                    assert_eq!(o[n], 0);
                    assert_eq!(e[n], Float::with_val(output_precision, &even[0]));
                    assert!(e.iter().chain(&o).all(|x| x.prec() == output_precision));
                    let work = output_precision.max(input_precision) + 256;
                    let root_two = Float::with_val(work, 2).sqrt();
                    for j in 1..=n {
                        assert_eq!(e[n - j], e[n + j]);
                        assert_eq!(o[n - j], -o[n + j].clone());
                        for (actual, source) in [(&e[n + j], &even[j]), (&o[n + j], &odd[j - 1])] {
                            let reference = Float::with_val(work, source) / &root_two;
                            let error = (Float::with_val(work, actual) - &reference).abs();
                            let tolerance = reference.clone().abs()
                                * (Float::with_val(work, 1) >> (output_precision - 3));
                            assert!(error <= tolerance);
                            checks += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checks, 1404);
}

fn exact_quotient(a: &[Float], v: &[Float]) -> Rational {
    let n = v.len();
    let denominator = v
        .iter()
        .map(|x| x.to_rational().unwrap().square())
        .sum::<Rational>();
    (0..n * n)
        .map(|j| {
            a[j].to_rational().unwrap()
                * v[j / n].to_rational().unwrap()
                * v[j % n].to_rational().unwrap()
        })
        .sum::<Rational>()
        / denominator
}
#[test]
fn public_form_decomposition_matches_exact_rationals_under_extreme_scaling() {
    let kinds = [
        CcmFormComponentKind::Archimedean,
        CcmFormComponentKind::Prime,
        CcmFormComponentKind::Pole,
        CcmFormComponentKind::Other,
    ];
    let signs = [-3, 1, 2, -1];
    let mut reports = 0;
    let mut checks = 0;
    for q in [64, 128, 256] {
        for case in 0usize..15 {
            let n = 1 + case % 5;
            let p = if case % 3 == 0 { 64 } else { q };
            let v = (0..n)
                .map(|j| {
                    Float::with_val(
                        q,
                        ((j * 7 + case) % 17 + 1) as i32 * if j % 2 == 0 { 1 } else { -1 },
                    ) / 13u32
                })
                .collect::<Vec<_>>();
            let matrices = (0..4)
                .map(|k| {
                    (0..n * n)
                        .map(|j| {
                            Float::with_val(
                                q,
                                j as i32 * 3 - (j % n) as i32 * 5 + case as i32 * (k + 1) + k - 9,
                            ) / (7 + k)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let total = (0..n * n)
                .map(|j| {
                    matrices.iter().zip(signs).fold(
                        Float::with_val(q, if j / n == j % n { 1 } else { 0 }) / 32u32,
                        |sum, (m, s)| sum + Float::with_val(q, &m[j] * s),
                    )
                })
                .collect::<Vec<_>>();
            let expected = Float::with_val(p, exact_quotient(&total, &v));
            let component_exact = matrices
                .iter()
                .map(|m| exact_quotient(m, &v))
                .collect::<Vec<_>>();
            let contributions = component_exact
                .iter()
                .zip(signs)
                .map(|(x, s)| Float::with_val(p, x.clone() * s))
                .collect::<Vec<_>>();
            let reconstructed = Float::with_val(
                p,
                contributions
                    .iter()
                    .map(|x| x.to_rational().unwrap())
                    .sum::<Rational>(),
            );
            let residual = Float::with_val(
                p,
                (expected.to_rational().unwrap() - reconstructed.to_rational().unwrap()).abs(),
            );
            for vs in [-700_000_000i32, 0, 700_000_000] {
                for ms in [-300_000_000i32, 0, 300_000_000] {
                    let source = v.iter().map(|x| x.clone() << vs).collect::<Vec<_>>();
                    let components = matrices
                        .iter()
                        .enumerate()
                        .map(|(j, m)| CcmFormComponentMatrixHp {
                            kind: kinds[j],
                            signed_coefficient: signs[j],
                            matrix_row_major: m.iter().map(|x| x.clone() << ms).collect(),
                        })
                        .collect::<Vec<_>>();
                    let total = total.iter().map(|x| x.clone() << ms).collect::<Vec<_>>();
                    let result =
                        evaluate_ccm_form_components_hp(&total, &components, &source, p).unwrap();
                    assert_eq!(result.total_value, expected.clone() << ms);
                    assert_eq!(result.reconstructed_total, reconstructed.clone() << ms);
                    assert_eq!(result.cancellation_residual, residual.clone() << ms);
                    checks += 3;
                    for j in 0..4 {
                        assert_eq!(
                            result.components[j].rayleigh_value,
                            Float::with_val(p, &component_exact[j]) << ms
                        );
                        assert_eq!(
                            result.components[j].signed_contribution,
                            contributions[j].clone() << ms
                        );
                        checks += 2;
                    }
                    reports += 1;
                }
            }
        }
    }
    assert_eq!(reports, 405);
    assert_eq!(checks, 4455);
}

#[test]
fn invalid_states_and_nonfinite_forms_fail_explicitly() {
    for value in [0.0, f64::NAN, f64::INFINITY] {
        let candidate = CcmStateCandidateHp {
            algebraic_index: 0,
            eigenvalue: Float::with_val(128, 1),
            eigenvector: vec![Float::with_val(128, value)],
            parity: CcmParity::Even,
        };
        assert!(select_ccm_state_hp(CcmStateTarget::AlgebraicGround, &[candidate]).is_err());
    }
    let kinds = [
        CcmFormComponentKind::Archimedean,
        CcmFormComponentKind::Prime,
        CcmFormComponentKind::Pole,
        CcmFormComponentKind::Other,
    ];
    let components = kinds
        .into_iter()
        .map(|kind| CcmFormComponentMatrixHp {
            kind,
            signed_coefficient: 1,
            matrix_row_major: vec![Float::with_val(128, 1)],
        })
        .collect::<Vec<_>>();
    assert!(evaluate_ccm_form_components_hp(
        &[Float::with_val(128, f64::NAN)],
        &components,
        &[Float::with_val(128, 1)],
        128
    )
    .is_err());
    assert!(evaluate_ccm_form_components_hp(
        &[Float::with_val(128, 1)],
        &components,
        &[Float::with_val(128, 0)],
        128
    )
    .is_err());
    assert!(std::panic::catch_unwind(|| expand_even_sector_vector(&[], 0, 128)).is_err());
    assert!(std::panic::catch_unwind(|| expand_odd_sector_vector(
        &[Float::with_val(128, 1)],
        0,
        128
    ))
    .is_err());
    assert!(std::panic::catch_unwind(|| expand_even_sector_vector(
        &[Float::with_val(128, f64::NAN)],
        0,
        128
    ))
    .is_err());
}
