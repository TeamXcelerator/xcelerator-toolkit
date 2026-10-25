//! Run-owned authentication of immutable retained sources. No artifact identity changes.
use super::*;
use crate::ccm::{
    retained_evidence::{RetainedMatrix, RetainedRoots},
    state_geometry::RetainedState,
};
use std::sync::Arc;

type Range = (i32, i32);
fn range() -> Range {
    (rug::float::exp_min(), rug::float::exp_max())
}

pub(super) struct Admitted<T> {
    pub(super) value: T,
    range: Range,
    _reservation: Option<xc_core::MemoryReservation>,
}
impl<T> std::ops::Deref for Admitted<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}
#[derive(Default)]
pub(super) struct Sources {
    state: Option<Arc<Admitted<RetainedState>>>,
    roots: Option<Arc<Admitted<RetainedRoots>>>,
    matrix: Option<Arc<Admitted<Matrix>>>,
}
impl Sources {
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
}
pub(super) struct Matrix {
    manifest: ArtifactManifest,
    cutoff: String,
    modes: usize,
    precision: u32,
    entries: xc_cache::SharedValidated<Vec<Float>>,
}
impl Matrix {
    pub(super) fn view(&self) -> RetainedMatrix<'_> {
        RetainedMatrix {
            manifest: self.manifest.clone(),
            cutoff: self.cutoff.clone(),
            modes: self.modes,
            precision: self.precision,
            entries: crate::ccm::retained_evidence::RetainedMatrixEntries::borrowed(&self.entries),
        }
    }
}
fn admit<S, T>(
    sources: &Mutex<S>,
    slot: fn(&mut S) -> &mut Option<Arc<Admitted<T>>>,
    resources: &xc_core::RunResources,
    valid: impl FnOnce(&Admitted<T>) -> Result<bool>,
    decode: impl FnOnce() -> Result<(T, u64)>,
) -> Result<Arc<Admitted<T>>> {
    let candidate = {
        let mut sources = sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot(&mut sources).clone()
    };
    if let Some(candidate) = candidate {
        if candidate.range == range() && valid(&candidate)? {
            return Ok(candidate);
        }
    }
    {
        let mut sources = sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot(&mut sources) = None;
    }
    // Authentication, decoding, size estimation and resource admission may
    // clear the run memo. Its eviction callback acquires `sources`, so none of
    // that work may hold this mutex. The run itself is exclusively borrowed;
    // concurrent eviction only removes optional retained ownership.
    let (value, bytes) = decode()?;
    // Retention is optional. A tight memo budget must not reject an otherwise
    // valid diagnostic: use the fully checked temporary without retaining it.
    let reservation = resources.try_reserve_retained(bytes).ok();
    let retained = reservation.is_some();
    let value = Arc::new(Admitted {
        value,
        range: range(),
        _reservation: reservation,
    });
    if retained {
        let mut sources = sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot(&mut sources) = Some(value.clone());
    }
    Ok(value)
}
impl RetainedCcmRun {
    pub(super) fn admitted_state(&mut self) -> Result<Arc<Admitted<RetainedState>>> {
        admit(
            &self.source_admissions,
            |sources| &mut sources.state,
            &self.resources,
            |state| {
                Ok(self
                    .eigenpair
                    .as_ref()
                    .is_some_and(|r| r.manifest == state.manifest))
            },
            || {
                let _stage =
                    xc_core::performance_stage("ccm.capture.state_admission", Default::default());
                let record = self
                    .eigenpair
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("retained eigenpair unavailable"))?;
                let state = RetainedState::from_payload(
                    &record.manifest,
                    &record.payload,
                    std::slice::from_ref(&record.manifest.content_digest),
                )?;
                let bytes = retained_state_bytes(&state)?;
                Ok((state, bytes))
            },
        )
    }
    pub(super) fn admitted_matrix(
        &mut self,
        state: &RetainedState,
    ) -> Result<Option<Arc<Admitted<Matrix>>>> {
        let Some(manifest) = self.source.tau_manifest.as_ref() else {
            return Ok(None);
        };
        admit(
            &self.source_admissions,
            |sources| &mut sources.matrix,
            &self.resources,
            |matrix| {
                Ok(matrix.manifest == *manifest
                    && matrix.cutoff == state.cutoff
                    && matrix.modes == state.modes
                    && matrix.precision == state.precision
                    && matrix.entries.ptr_eq(&self.source.tau))
            },
            || {
                let _stage =
                    xc_core::performance_stage("ccm.capture.matrix_admission", Default::default());
                RetainedMatrix::from_admitted_runtime(
                    manifest.clone(),
                    state.cutoff.clone(),
                    state.modes,
                    state.precision,
                    &self.source.tau,
                )?;
                let matrix = Matrix {
                    manifest: manifest.clone(),
                    cutoff: state.cutoff.clone(),
                    modes: state.modes,
                    precision: state.precision,
                    entries: self.source.tau.clone(),
                };
                // The shared matrix already owns its allocation reservation. Only the
                // metadata is additionally retained by this adapter.
                let bytes = crate::ccm::capture_runtime::retained_json_bytes(manifest)?
                    .saturating_add(state.cutoff.capacity() as u64)
                    .saturating_add(4096);
                Ok((matrix, bytes))
            },
        )
        .map(Some)
    }
    pub(super) fn admitted_roots(
        &mut self,
        state: &RetainedState,
    ) -> Result<Option<Arc<Admitted<RetainedRoots>>>> {
        let (Some(root), Some(secular)) =
            (&self.source.root_manifest, &self.source.secular_manifest)
        else {
            return Ok(None);
        };
        let root_record = self.retained_records.iter().find(|r| {
            r.manifest.key == root.key && r.manifest.content_digest == root.content_digest
        });
        let secular_record = self.retained_records.iter().find(|r| {
            r.manifest.key == secular.key && r.manifest.content_digest == secular.content_digest
        });
        let (Some(root_record), Some(secular_record)) = (root_record, secular_record) else {
            return Ok(None);
        };
        admit(
            &self.source_admissions,
            |sources| &mut sources.roots,
            &self.resources,
            |roots| {
                Ok(*root == roots.manifest
                    && *secular == roots.secular_manifest
                    && xc_cache::manifest_depends_on(&roots.secular_manifest, &state.manifest)?)
            },
            || {
                let _stage = xc_core::performance_stage(
                    "ccm.capture.root_source_admission",
                    Default::default(),
                );
                let roots = RetainedRoots::from_payload(
                    root,
                    &root_record.payload,
                    secular,
                    &secular_record.payload,
                    state,
                    &[root.content_digest.clone(), secular.content_digest.clone()],
                )?;
                let bytes = crate::ccm::capture_runtime::retained_json_bytes(&(
                    &roots.manifest,
                    &roots.secular_manifest,
                    &roots.dataset,
                    &roots.completeness,
                    &roots.acquisition,
                    &roots.cutoff,
                    roots.modes,
                ))?;
                Ok((roots, bytes))
            },
        )
        .map(Some)
    }
}

// Decimal payload size does not bound MPFR storage: even a zero carries the
// full requested precision. Account for allocated vector slots and actual limbs.
fn coefficient_bytes(values: &Vec<Float>) -> u64 {
    (values.capacity() as u64)
        .saturating_mul(std::mem::size_of::<Float>() as u64)
        .saturating_add(values.iter().fold(0u64, |sum, value| {
            sum.saturating_add(
                u64::from(value.prec())
                    .div_ceil(64)
                    .saturating_mul(8)
                    .saturating_add(64),
            )
        }))
}
fn retained_state_bytes(state: &RetainedState) -> Result<u64> {
    Ok(crate::ccm::capture_runtime::retained_json_bytes(&(
        &state.manifest,
        &state.cutoff,
        state.modes,
        state.precision,
        &state.eigenvalue,
        &state.selection_policy,
    ))?
    .saturating_add(coefficient_bytes(&state.coefficients)))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_admission_validation_and_decode_can_evict_the_run_memo() {
        let resources = xc_core::RunResources::new(100).unwrap();
        let memo = xc_cache::ValidatedValueMemo::new(resources.clone(), 100);
        let sources = Arc::new(Mutex::new(None::<Arc<Admitted<u32>>>));
        let weak_sources = Arc::downgrade(&sources);
        let eviction: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            *weak_sources.upgrade().unwrap().lock().unwrap() = None;
        });
        memo.register_eviction(&eviction);
        let first = admit(
            &sources,
            |slot| slot,
            &resources,
            |_| Ok(true),
            || {
                // Fail promptly if a future change holds the lock here, rather
                // than letting this regression itself hang in the callback.
                assert!(sources.try_lock().is_ok());
                memo.clear();
                Ok((7, 40))
            },
        )
        .unwrap();
        assert_eq!(resources.usage().retained_bytes, 40);
        let hit = admit(
            &sources,
            |slot| slot,
            &resources,
            |value| {
                assert_eq!(value.value, 7);
                assert!(sources.try_lock().is_ok());
                memo.clear();
                Ok(true)
            },
            || panic!("an authenticated hit must not decode again"),
        )
        .unwrap();
        assert!(Arc::ptr_eq(&first, &hit));
        assert!(sources.lock().unwrap().is_none());
        assert_eq!(
            resources.usage().retained_bytes,
            40,
            "eviction must not uncharge a value still used by its caller"
        );
        drop(first);
        drop(hit);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn source_admission_revalidates_changed_bindings_and_does_not_retain_failures() {
        let resources = xc_core::RunResources::new(100).unwrap();
        let sources = Mutex::new(None::<Arc<Admitted<u32>>>);
        let first = admit(
            &sources,
            |slot| slot,
            &resources,
            |_| Ok(true),
            || Ok((1, 40)),
        )
        .unwrap();
        let next = admit(
            &sources,
            |slot| slot,
            &resources,
            |_| Ok(false),
            || Ok((2, 40)),
        )
        .unwrap();
        assert!(!Arc::ptr_eq(&first, &next));
        assert_eq!(next.value, 2);
        assert_eq!(resources.usage().retained_bytes, 80);
        drop(first);
        drop(next);
        assert!(admit(
            &sources,
            |slot| slot,
            &resources,
            |_| Ok(false),
            || { anyhow::bail!("invalid source") }
        )
        .is_err());
        assert!(sources.lock().unwrap().is_none());
        assert_eq!(resources.usage().retained_bytes, 0);

        let occupied = resources.try_reserve_retained(100).unwrap();
        let temporary = admit(
            &sources,
            |slot| slot,
            &resources,
            |_| Ok(true),
            || Ok((3, 40)),
        )
        .unwrap();
        assert_eq!(temporary.value, 3);
        assert!(
            sources.lock().unwrap().is_none(),
            "optional retention must respect the budget"
        );
        assert_eq!(resources.usage().retained_bytes, 100);
        drop(temporary);
        drop(occupied);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn source_admission_charges_high_precision_zero_limbs_and_spare_slots() {
        let mut values = Vec::with_capacity(8);
        values.push(Float::with_val(1_000_000, 0));
        values.push(Float::with_val(1_000_000, 1));
        assert!(coefficient_bytes(&values) >= 250_000 + 8 * std::mem::size_of::<Float>() as u64);
        let small = vec![Float::with_val(64, 0), Float::with_val(64, 1)];
        assert!(coefficient_bytes(&values) > coefficient_bytes(&small) + 249_000);
    }
}
