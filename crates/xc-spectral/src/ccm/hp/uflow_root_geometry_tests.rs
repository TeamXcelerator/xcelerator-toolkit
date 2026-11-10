//! Equality against the cold pre-reuse u-flow root route, including refusals.
use super::*;

fn outcome(value: Float) -> EigenvalueResult {
    let p = value.prec();
    EigenvalueResult::Converged(RootRefinement {
        value,
        diagnostics: RootRefinementDiagnostics {
            iterations: 1,
            final_correction: Float::with_val(p, 0),
            residual: Float::with_val(p, 0),
            achieved_decimal_digits: Float::with_val(p, 20),
        },
    })
}

fn cold_values(
    unit: &[Float],
    tangent: &[Float],
    poles: &[Float],
    motions: Option<&[Float]>,
    roots: &[EigenvalueResult],
    p: u32,
) -> Result<Vec<Option<String>>> {
    roots
        .iter()
        .map(|outcome| {
            outcome
                .value()
                .map(|root| {
                    root_response_math::evaluate(unit, tangent, poles, motions, root, p, None)
                        .map(|value| lossless_hp_decimal(&value))
                })
                .transpose()
        })
        .collect()
}

type ExactFloat = (Option<(rug::Integer, i32)>, bool);

fn exact(result: Result<Float>) -> Result<ExactFloat, String> {
    result
        .map(|value| (value.to_integer_exp(), value.is_sign_negative()))
        .map_err(|error| error.to_string())
}

#[test]
fn lazy_uflow_geometry_matches_four_fixed_and_two_moving_passes() {
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let resources = xc_core::RunResources::with_worker_limit(32 << 20, workers).unwrap();
            let _scope = resources.enter();
            for p in [64, 129, 3386] {
                let unit = [1, -3, 2, 5].map(|x| Float::with_val(p, x));
                let poles = [-3, -1, 1, 3].map(|x| Float::with_val(p, x));
                let roots = [
                    outcome(Float::with_val(p, 0.25)),
                    EigenvalueResult::Failed {
                        iterations: 2,
                        reason: "retained missing root".into(),
                    },
                    outcome(Float::with_val(p, 4)),
                ];
                let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
                assert_eq!(resources.usage().working_bytes, 0);
                let motions = [-2, 0, 1, 3].map(|x| Float::with_val(p, x));
                let zero = [0, 0, 0, 0].map(|x| Float::with_val(p, x));
                let total = [8, -3, -1, 6].map(|x| Float::with_val(p, x));
                for coefficients in [
                    [1, -2, 3, -4],
                    [0, 0, 0, 0],
                    [7, -1, -4, 10],
                    [8, -3, -1, 6],
                ] {
                    let tangent = coefficients.map(|x| Float::with_val(p, x));
                    let expected = cold_values(&unit, &tangent, &poles, None, &roots, p).unwrap();
                    assert_eq!(prepared.values(&tangent, None).unwrap(), expected);
                }
                for tangent in [&zero[..], &total[..]] {
                    let expected =
                        cold_values(&unit, tangent, &poles, Some(&motions), &roots, p).unwrap();
                    assert_eq!(prepared.values(tangent, Some(&motions)).unwrap(), expected);
                }
                assert!(resources.usage().working_bytes > 0);
                drop(prepared);
                assert_eq!(resources.usage().working_bytes, 0);
            }
        });
    }
}

#[test]
fn lazy_uflow_geometry_preserves_unavailable_roots_and_first_failure() {
    let p = 64;
    let resources = xc_core::RunResources::with_worker_limit(1 << 20, 1).unwrap();
    let _scope = resources.enter();
    let invalid = [Float::with_val(p, rug::float::Special::Nan)];
    let missing = [EigenvalueResult::Failed {
        iterations: 0,
        reason: "not acquired".into(),
    }];
    let prepared = LazyUFlowRootResponses::new(&invalid, &[], &missing, 0);
    assert_eq!(prepared.values(&[], Some(&[])).unwrap(), vec![None]);
    assert_eq!(resources.usage().working_bytes, 0);
    let unit = [1, -1].map(|x| Float::with_val(p, x));
    let poles = [-1, 1].map(|x| Float::with_val(p, x));
    let tangent = [0, 1].map(|x| Float::with_val(p, x));
    // Root zero has an exactly zero derivative; the later root is a pole.
    let roots = [
        outcome(Float::with_val(p, 0)),
        outcome(Float::with_val(p, 1)),
    ];
    let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
    let expected = cold_values(&unit, &tangent, &poles, None, &roots, p)
        .unwrap_err()
        .to_string();
    assert!(expected.contains("zero secular derivative"));
    assert_eq!(
        prepared.values(&tangent, None).unwrap_err().to_string(),
        expected
    );
    // Invalid tangent wins over even the first numerical geometry failure.
    assert_eq!(
        prepared.values(&invalid, None).unwrap_err().to_string(),
        cold_values(&unit, &invalid, &poles, None, &roots, p)
            .unwrap_err()
            .to_string()
    );
}

#[test]
fn lazy_uflow_geometry_preserves_extended_guard_cancellation_and_moving_terms() {
    let p = 64;
    let resources = xc_core::RunResources::with_worker_limit(16 << 20, 1).unwrap();
    let _scope = resources.enter();
    let unit = [
        Float::with_val(p, 1),
        Float::with_val(p, -1),
        Float::with_val(p, 1) >> 5000u32,
    ];
    let poles = [-3, 3, 2].map(|x| Float::with_val(p, x));
    let tangent = [
        Float::with_val(p, 0),
        Float::with_val(p, 0),
        Float::with_val(p, -1) >> 5001u32,
    ];
    let motions = [1, 1, 1].map(|x| Float::with_val(p, x));
    let zeros = [
        Float::with_val(p, rug::float::Special::NegZero),
        Float::with_val(p, 0),
        Float::with_val(p, 0),
    ];
    let roots = [outcome(Float::with_val(p, 0))];
    let root = roots[0].value().unwrap();
    let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
    for limit in [4096, 8192] {
        let policy = root_response_math::GuardPolicy::new(p, limit).unwrap();
        for (tangent, motion) in [
            (&tangent[..], None),
            (&zeros[..], Some(&motions[..])),
            (&tangent[..], Some(&motions[..])),
        ] {
            let expected = exact(root_response_math::evaluate_with_policy(
                &unit, tangent, &poles, motion, root, policy, None,
            ));
            for _ in 0..2 {
                assert_eq!(
                    exact(prepared.value(0, root, tangent, motion, policy)),
                    expected
                );
            }
        }
    }
    assert!(prepared.clear_optional());
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
fn lazy_uflow_geometry_eviction_keeps_previously_admitted_scalar_success() {
    let p = 128;
    let unit = [1, 2, 3].map(|x| Float::with_val(p, x));
    let poles = [-1, 0, 1].map(|x| Float::with_val(p, x));
    let tangent = [-1, 2, -3].map(|x| Float::with_val(p, x));
    let roots = [outcome(Float::with_val(p, 2))];
    let attempt = u64::try_from(3 * 24 * root_response_math::point_bytes(p, 64).unwrap()).unwrap();
    for capacity in [attempt, 1 << 20] {
        let resources = xc_core::RunResources::with_worker_limit(capacity, 1).unwrap();
        let _scope = resources.enter();
        let expected = cold_values(&unit, &tangent, &poles, None, &roots, p).unwrap();
        let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
        assert_eq!(prepared.values(&tangent, None).unwrap(), expected);
        if capacity > attempt {
            let stored = resources.usage().working_bytes;
            assert!(stored > 0);
            let held = resources
                .try_reserve_working(capacity - stored - attempt + 1)
                .unwrap();
            assert_eq!(prepared.values(&tangent, None).unwrap(), expected);
            assert_eq!(resources.usage().working_bytes, held.bytes());
            drop(held);
        }
        drop(prepared);
        assert_eq!(resources.usage().working_bytes, 0);
    }
}

#[test]
fn lazy_uflow_geometry_does_not_retry_numerical_failure() {
    let p = 128;
    let resources = xc_core::RunResources::with_worker_limit(1 << 20, 1).unwrap();
    let _scope = resources.enter();
    let unit = [Float::with_val(p, 1)];
    let poles = [Float::with_val(p, 0)];
    let roots = [outcome(Float::with_val(p, 2))];
    let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
    prepared.values(&unit, None).unwrap();
    let retained = resources.usage().working_bytes;
    let mut calls = 0;
    let result: Result<()> = prepared.with_workspace(|| {
        calls += 1;
        bail!("unresolved numerical gate")
    });
    assert_eq!(result.unwrap_err().to_string(), "unresolved numerical gate");
    assert_eq!(calls, 1);
    assert_eq!(resources.usage().working_bytes, retained);
    let result: Result<()> = prepared.with_workspace(|| {
        calls += 1;
        Err(xc_core::ConfigError::new("run work cancelled by owner").into())
    });
    assert_eq!(
        result.unwrap_err().to_string(),
        "run work cancelled by owner"
    );
    assert_eq!(calls, 2);
    assert_eq!(resources.usage().working_bytes, retained);
}

#[test]
fn lazy_uflow_geometry_releases_storage_for_later_channel_work() {
    let p = 128;
    let resources = xc_core::RunResources::with_worker_limit(1 << 20, 1).unwrap();
    let _scope = resources.enter();
    let unit = [Float::with_val(p, 1)];
    let poles = [Float::with_val(p, 0)];
    let roots = [outcome(Float::with_val(p, 2))];
    let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
    prepared.values(&unit, None).unwrap();
    assert!(resources.usage().working_bytes > 0);
    let mut calls = 0;
    prepared
        .with_workspace(|| {
            calls += 1;
            let _allocation = resources.try_reserve_working(resources.maximum_bytes())?;
            Ok(())
        })
        .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(resources.usage().working_bytes, 0);
}

#[test]
#[ignore = "supervised six-pass root geometry timing; not a whole-u-flow estimate"]
fn measure_lazy_uflow_root_geometry() {
    use std::{hint::black_box, time::Instant};
    let resources = xc_core::RunResources::with_worker_limit(256 << 20, 1).unwrap();
    let _scope = resources.enter();
    for p in [729, 4281] {
        let poles = (-200..=200)
            .map(|x| Float::with_val(p, x))
            .collect::<Vec<_>>();
        let unit = (0..poles.len())
            .map(|i| Float::with_val(p, 1) / (i + 1))
            .collect::<Vec<_>>();
        let roots = (0..8)
            .map(|i| outcome(Float::with_val(p, 201 + i)))
            .collect::<Vec<_>>();
        let tangents = (0..4)
            .map(|channel| {
                unit.iter()
                    .enumerate()
                    .map(|(i, value)| Float::with_val(p, value * ((i % 5) as i32 - channel)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let zeros = vec![Float::with_val(p, 0); unit.len()];
        let motions = (0..unit.len())
            .map(|i| Float::with_val(p, (i % 3) as i32 - 1))
            .collect::<Vec<_>>();
        let run = |reuse| -> Result<Vec<Vec<Option<String>>>> {
            let prepared = LazyUFlowRootResponses::new(&unit, &poles, &roots, p);
            let values = |tangent: &[Float], motion: Option<&[Float]>| {
                if reuse {
                    prepared.values(tangent, motion)
                } else {
                    cold_values(&unit, tangent, &poles, motion, &roots, p)
                }
            };
            let mut results = Vec::new();
            for tangent in &tangents {
                results.push(values(tangent, None)?);
            }
            results.push(values(&zeros, Some(&motions))?);
            results.push(values(&tangents[3], Some(&motions))?);
            Ok(results)
        };
        assert_eq!(run(false).unwrap(), run(true).unwrap());
        let mut timings = [Vec::new(), Vec::new()];
        for repetition in 0..5 {
            for route in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                black_box(run(route == 1).unwrap());
                timings[route].push(start.elapsed().as_secs_f64());
            }
        }
        println!(
            "u-flow root geometry raw: p={p}, poles={}, roots={}, passes=6, cold={:?}, reused={:?}",
            poles.len(),
            roots.len(),
            timings[0],
            timings[1]
        );
        for times in &mut timings {
            times.sort_by(f64::total_cmp);
        }
        println!("u-flow root geometry median: p={p}, cold={:.6}s, reused={:.6}s, speedup={:.4}x; serial root order, source preparation excluded, owner creation/drop and string conversion included; not a whole-u-flow estimate", timings[0][2], timings[1][2], timings[0][2] / timings[1][2]);
        assert_eq!(resources.usage().working_bytes, 0);
    }
}
