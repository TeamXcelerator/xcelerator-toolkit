//! Optional, build-bound progress for exact retained primitive contractions.
use super::*;
use crate::ccm::capture_runtime::Checkpoints;
use crate::ccm::convergence_capture::OperatorAction;

fn valid_scalar(value: &str, p: u32) -> bool {
    Float::parse(value).ok().is_some_and(|v| {
        let number = Float::with_val(p, v);
        number.is_finite()
            && (!number.is_zero()
                || xc_core::DecimalLiteral::new(value)
                    .and_then(|v| v.canonical())
                    .is_ok_and(|v| v.as_str() == "0"))
    })
}
fn checkpoint_identity<T: serde::Serialize>(
    identity: &T,
    working_bytes: u128,
    workers: usize,
) -> impl serde::Serialize + '_ {
    (
        "retained-component-actions-v1",
        identity,
        rug::float::exp_min(),
        rug::float::exp_max(),
        working_bytes,
        workers,
    )
}
pub(super) fn actions(
    identity: &impl serde::Serialize,
    reuse: bool,
    expected: &[(&str, &ContentDigest, &str)],
    dimension: usize,
    precision: u32,
    compute: impl FnOnce() -> Result<Vec<OperatorAction>>,
) -> Result<Vec<OperatorAction>> {
    let _stage =
        xc_core::performance_stage("ccm.capture.prepared_component_actions", Default::default());
    let checkpoints = Checkpoints::new(&checkpoint_identity(
        identity,
        source_working_budget()?,
        rayon::current_num_threads(),
    ))?;
    actions_with_store(&checkpoints, reuse, expected, dimension, precision, compute)
}
fn actions_with_store(
    checkpoints: &Checkpoints,
    reuse: bool,
    expected: &[(&str, &ContentDigest, &str)],
    dimension: usize,
    precision: u32,
    compute: impl FnOnce() -> Result<Vec<OperatorAction>>,
) -> Result<Vec<OperatorAction>> {
    let valid = |values: &[OperatorAction]| {
        values.len() == expected.len()
            && values
                .iter()
                .zip(expected)
                .all(|(v, (label, digest, convention))| {
                    v.label == *label
                        && v.source_digest == **digest
                        && v.convention == *convention
                        && v.action.len() == dimension
                        && v.action.iter().all(|x| valid_scalar(x, precision))
                })
    };
    if reuse {
        if let Some(values) = checkpoints.load::<Vec<OperatorAction>>("completed")? {
            if valid(&values) && xc_numerics::mpfr_interval::ensure_uniform_exponent_range().is_ok()
            {
                return Ok(values);
            }
        }
    }
    let values = compute()?;
    // A temporary arithmetic/resource failure never becomes a progress record.
    if valid(&values) {
        if let Err(error) = checkpoints.save("completed", &values) {
            xc_core::progress_message!("component action progress unavailable: {error}");
        }
    }
    Ok(values)
}

#[cfg(test)]
mod component_action_checkpoint_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture() -> Vec<OperatorAction> {
        ["tau_pole", "tau_archimedean"]
            .into_iter()
            .map(|label| OperatorAction {
                label: label.into(),
                source_digest: ContentDigest::sha256(label.as_bytes()),
                action: vec![
                    "-0".into(),
                    "1.2345678901234567890123456789".into(),
                    "-2.5e-100".into(),
                ],
                convention: "exact retained component checkpoint fixture".into(),
            })
            .collect()
    }
    fn expected(values: &[OperatorAction]) -> Vec<(&str, &ContentDigest, &str)> {
        values
            .iter()
            .map(|v| (v.label.as_str(), &v.source_digest, v.convention.as_str()))
            .collect()
    }
    fn store(directory: &std::path::Path, identity: &str, budget: u128) -> Checkpoints {
        Checkpoints::local(
            &checkpoint_identity(&identity, budget, rayon::current_num_threads()),
            directory.to_path_buf(),
        )
        .unwrap()
    }
    fn serial(run: impl FnOnce() + Send) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
    }

    #[test]
    fn component_action_checkpoint_cold_warm_and_explicit_recompute_preserve_bytes() {
        serial(|| {
            let directory = xc_core::test_support::TestDir::new("component-action-reuse");
            let store = store(&directory, "same-source", 8 << 30);
            let values = fixture();
            let expected = expected(&values);
            let calls = AtomicUsize::new(0);
            let compute = || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(values.clone())
            };
            let cold = actions_with_store(&store, true, &expected, 3, 128, compute).unwrap();
            let warm = actions_with_store(&store, true, &expected, 3, 128, compute).unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let bytes = serde_json::to_vec(&values).unwrap();
            assert_eq!(serde_json::to_vec(&cold).unwrap(), bytes);
            assert_eq!(serde_json::to_vec(&warm).unwrap(), bytes);
            // Explicit recomputation must run even when a complete valid
            // checkpoint exists, then replace it with the actual new result.
            let mut replacement = values.clone();
            replacement[0].action[1] = "2".into();
            let refreshed = actions_with_store(&store, false, &expected, 3, 128, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(replacement.clone())
            })
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(refreshed, replacement);
            let warm = actions_with_store(&store, true, &expected, 3, 128, || {
                panic!("fresh complete action should be reused")
            })
            .unwrap();
            assert_eq!(
                serde_json::to_vec(&warm).unwrap(),
                serde_json::to_vec(&replacement).unwrap()
            );
        });
    }

    #[test]
    fn component_action_checkpoint_source_budget_and_pool_changes_miss() {
        let directory = xc_core::test_support::TestDir::new("component-action-context");
        let values = fixture();
        let expected = expected(&values);
        let calls = AtomicUsize::new(0);
        for (threads, identity, budget, expected_calls) in [
            (1, "source-a", 8u128 << 30, 1),
            (1, "source-a", 8u128 << 30, 1),
            (1, "source-b", 8u128 << 30, 2),
            (1, "source-a", 4u128 << 30, 3),
            (2, "source-a", 8u128 << 30, 4),
            (2, "source-a", 8u128 << 30, 4),
            (1, "source-a", 8u128 << 30, 4),
        ] {
            let actual = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let store = store(&directory, identity, budget);
                    actions_with_store(&store, true, &expected, 3, 128, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(values.clone())
                    })
                    .unwrap()
                });
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            assert_eq!(
                serde_json::to_vec(&actual).unwrap(),
                serde_json::to_vec(&values).unwrap()
            );
        }
    }

    #[test]
    fn component_action_checkpoint_rejects_invalid_sealed_actions_and_corrupt_bytes() {
        serial(|| {
            let directory = xc_core::test_support::TestDir::new("component-action-invalid");
            let store = store(&directory, "same-source", 8 << 30);
            let values = fixture();
            let expected = expected(&values);
            let calls = AtomicUsize::new(0);
            for case in 0..11 {
                let mut invalid = values.clone();
                match case {
                    0 => {
                        invalid.pop();
                    }
                    1 => invalid.swap(0, 1),
                    2 => invalid[0].label = "wrong label".into(),
                    3 => invalid[0].source_digest = ContentDigest::sha256(b"wrong source"),
                    4 => invalid[0].convention = "wrong convention".into(),
                    5 => {
                        invalid[0].action.pop();
                    }
                    6 => invalid[0].action[0] = "not a number".into(),
                    7 => invalid[0].action[0] = "NaN".into(),
                    8 => invalid[0].action[0] = "inf".into(),
                    9 => invalid[0].action[0] = "1e-9999999999".into(),
                    _ => invalid[0].action[0] = "1e9999999999".into(),
                }
                // A correctly sealed checkpoint still needs domain and
                // source/shape validation before its computation is skipped.
                store.save("completed", &invalid).unwrap();
                let actual = actions_with_store(&store, true, &expected, 3, 128, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(values.clone())
                })
                .unwrap();
                assert_eq!(calls.load(Ordering::SeqCst), case + 1);
                assert_eq!(
                    serde_json::to_vec(&actual).unwrap(),
                    serde_json::to_vec(&values).unwrap()
                );
            }
            let data_dir = std::fs::read_dir(&directory)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            let path = data_dir.join(ContentDigest::sha256(b"completed").0);
            let mut corrupt = std::fs::read(&path).unwrap();
            let position = corrupt.iter().position(|&b| b == b'1').unwrap();
            corrupt[position] = b'9';
            std::fs::write(&path, corrupt).unwrap();
            let actual = actions_with_store(&store, true, &expected, 3, 128, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(values.clone())
            })
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 12);
            assert_eq!(
                serde_json::to_vec(&actual).unwrap(),
                serde_json::to_vec(&values).unwrap()
            );
        });
    }

    #[test]
    fn component_action_checkpoint_errors_and_invalid_computations_are_not_cached() {
        serial(|| {
            let directory = xc_core::test_support::TestDir::new("component-action-failure");
            let store = store(&directory, "same-source", 8 << 30);
            let values = fixture();
            let expected = expected(&values);
            let calls = AtomicUsize::new(0);
            let error = actions_with_store(&store, true, &expected, 3, 128, || {
                calls.fetch_add(1, Ordering::SeqCst);
                bail!("transient component action resource refusal")
            })
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "transient component action resource refusal"
            );
            assert!(store
                .load::<Vec<OperatorAction>>("completed")
                .unwrap()
                .is_none());
            let mut invalid = values.clone();
            invalid[0].action[0] = "NaN".into();
            assert_eq!(
                actions_with_store(&store, true, &expected, 3, 128, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(invalid.clone())
                })
                .unwrap(),
                invalid
            );
            assert!(store
                .load::<Vec<OperatorAction>>("completed")
                .unwrap()
                .is_none());
            let actual = actions_with_store(&store, true, &expected, 3, 128, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(values.clone())
            })
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 3);
            assert_eq!(
                serde_json::to_vec(&actual).unwrap(),
                serde_json::to_vec(&values).unwrap()
            );
            // Reuse=false also must not hide a new failure behind that success.
            let error = actions_with_store(&store, false, &expected, 3, 128, || {
                calls.fetch_add(1, Ordering::SeqCst);
                bail!("new refusal during explicit recomputation")
            })
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "new refusal during explicit recomputation"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 4);
            assert_eq!(
                store
                    .load::<Vec<OperatorAction>>("completed")
                    .unwrap()
                    .unwrap(),
                values
            );
        });
    }
}
