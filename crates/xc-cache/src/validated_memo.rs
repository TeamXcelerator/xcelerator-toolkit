//! Run-scoped immutable validated values; allocation budgets use runtime bytes.
use super::*;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    sync::{Arc, Mutex},
};
use xc_core::{MemoryReservation, RunResources};

struct SharedRuntime<T> {
    value: T,
    bytes: u64,
    _reservation: Option<MemoryReservation>,
}
/// A validated immutable runtime allocation. Cloning shares its allocation and reservation.
pub struct SharedValidated<T>(Arc<SharedRuntime<T>>);
impl<T> Clone for SharedValidated<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> std::ops::Deref for SharedValidated<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.value
    }
}
impl<T> AsRef<[T]> for SharedValidated<Vec<T>> {
    fn as_ref(&self) -> &[T] {
        &self.0.value
    }
}
impl<T> AsRef<T> for SharedValidated<T> {
    fn as_ref(&self) -> &T {
        &self.0.value
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for SharedValidated<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.value.fmt(f)
    }
}
impl<T: PartialEq> PartialEq for SharedValidated<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.value == other.0.value
    }
}
impl<T: Eq> Eq for SharedValidated<T> {}
impl<T> SharedValidated<T> {
    pub fn into_owned(self) -> T
    where
        T: Clone,
    {
        match Arc::try_unwrap(self.0) {
            Ok(value) => value.value,
            Err(shared) => shared.value.clone(),
        }
    }
    pub fn retained_bytes(&self) -> u64 {
        self.0.bytes
    }
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Admit an already owned value. Prefer reserving before its allocation when possible.
    pub fn new(value: T, bytes: u64) -> Result<Self, CacheError> {
        let reservation = RunResources::current()
            .map(|r| r.try_reserve_retained(bytes))
            .transpose()
            .map_err(|e| CacheError::InvalidManifest(e.to_string()))?;
        Ok(Self(Arc::new(SharedRuntime {
            value,
            bytes,
            _reservation: reservation,
        })))
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key {
    payload_type: TypeId,
    value_type: TypeId,
    artifact: ArtifactKey,
    payload_digest: ContentDigest,
    context: String,
}
struct Entry {
    value: Arc<dyn Any + Send + Sync>,
    bytes: u64,
    sequence: u64,
}
#[derive(Default)]
struct Memo {
    entries: std::collections::HashMap<Key, Entry>,
    recency: BTreeMap<u64, Key>,
    retained: u64,
    next: u64,
}
struct Scope {
    resources: RunResources,
    maximum_bytes: u64,
    memo: Mutex<Memo>,
    evictions: Mutex<Vec<std::sync::Weak<dyn Fn() + Send + Sync>>>,
}
/// Explicit lifetime for validated allocations. Independent runs never share hidden retention.
#[derive(Clone)]
pub struct ValidatedValueMemo(Arc<Scope>);
thread_local! {static CURRENT_MEMO:RefCell<Option<ValidatedValueMemo>>=const {RefCell::new(None)};
static STATISTICS:std::cell::Cell<ValidatedValueMemoStatistics>=const {std::cell::Cell::new(ValidatedValueMemoStatistics{hits:0,insertions:0})};}
pub struct ValidatedValueMemoScope {
    previous: Option<ValidatedValueMemo>,
    _local: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for ValidatedValueMemoScope {
    fn drop(&mut self) {
        CURRENT_MEMO.with(|c| *c.borrow_mut() = self.previous.take());
    }
}
impl ValidatedValueMemo {
    pub fn current() -> Option<Self> {
        CURRENT_MEMO.with(|c| c.borrow().clone())
    }
    pub fn new(resources: RunResources, maximum_bytes: u64) -> Self {
        Self(Arc::new(Scope {
            maximum_bytes: maximum_bytes.min(resources.maximum_bytes()),
            resources,
            memo: Mutex::new(Memo::default()),
            evictions: Mutex::new(Vec::new()),
        }))
    }
    pub fn enter(&self) -> ValidatedValueMemoScope {
        ValidatedValueMemoScope {
            previous: CURRENT_MEMO.with(|c| c.replace(Some(self.clone()))),
            _local: std::marker::PhantomData,
        }
    }
    /// Register optional run-owned retention to release on workspace pressure.
    /// The caller owns the callback lifetime. Callbacks must not recursively clear
    /// this memo; they run outside its locks and release only optional ownership.
    #[doc(hidden)]
    pub fn register_eviction(&self, callback: &Arc<dyn Fn() + Send + Sync>) {
        self.0
            .evictions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::downgrade(callback));
    }
    pub fn clear(&self) {
        let callbacks = {
            let mut entries = self
                .0
                .evictions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let callbacks = entries
                .iter()
                .filter_map(std::sync::Weak::upgrade)
                .collect::<Vec<_>>();
            entries.retain(|entry| entry.strong_count() != 0);
            callbacks
        };
        for callback in callbacks {
            callback();
        }
        let old = {
            let mut m = self
                .0
                .memo
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *m)
        };
        drop(old);
    }
    pub fn retained_bytes(&self) -> u64 {
        self.0
            .memo
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retained
    }
}
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ValidatedValueMemoStatistics {
    pub hits: u64,
    pub insertions: u64,
}
#[doc(hidden)]
pub fn validated_value_memo_thread_statistics() -> ValidatedValueMemoStatistics {
    STATISTICS.with(std::cell::Cell::get)
}
fn event(hit: bool) {
    STATISTICS.with(|c| {
        let mut v = c.get();
        if hit {
            v.hits += 1
        } else {
            v.insertions += 1
        };
        c.set(v);
    });
}
pub(super) struct ValidatedValueMemoBinding<'a> {
    payload_type: TypeId,
    pub(super) context: &'a ValidatedValueMemoContext,
    scope: ValidatedValueMemo,
    size: fn(&dyn Any) -> Option<u64>,
}
impl ValidatedValueMemoBinding<'_> {
    fn key<R: 'static>(
        &self,
        artifact: &ArtifactKey,
        digest: &ContentDigest,
        context: &str,
    ) -> Key {
        Key {
            payload_type: self.payload_type,
            value_type: TypeId::of::<R>(),
            artifact: artifact.clone(),
            payload_digest: digest.clone(),
            context: context.into(),
        }
    }
    pub(super) fn lookup<R: Clone + Send + Sync + 'static>(
        &self,
        artifact: &ArtifactKey,
        digest: &ContentDigest,
    ) -> Option<R> {
        let retained = {
            let mut m = self.scope.0.memo.lock().ok()?;
            let key = std::iter::once(&self.context.reuse)
                .chain(&self.context.stronger)
                .map(|c| self.key::<R>(artifact, digest, c))
                .find(|k| m.entries.contains_key(k))?;
            let sequence = m.next;
            m.next += 1;
            let entry = m.entries.get_mut(&key)?;
            let previous = entry.sequence;
            entry.sequence = sequence;
            let value = entry.value.clone();
            m.recency.remove(&previous);
            m.recency.insert(sequence, key);
            value
        };
        let value = retained.downcast::<R>().ok()?.as_ref().clone();
        event(true);
        Some(value)
    }
    pub(super) fn insert<R: Clone + Send + Sync + 'static>(
        &self,
        artifact: &ArtifactKey,
        digest: &ContentDigest,
        _payload_bytes: u64,
        context: &str,
        value: &R,
    ) {
        let Some(bytes) = (self.size)(value) else {
            return;
        };
        if bytes > self.scope.0.maximum_bytes {
            return;
        }
        // Only shared handles reach this binding. Allocation/clone/destruction
        // stay outside the global critical section.
        let shared: Arc<dyn Any + Send + Sync> = Arc::new(value.clone());
        let key = self.key::<R>(artifact, digest, context);
        let mut evicted = Vec::new();
        {
            let Ok(mut m) = self.scope.0.memo.lock() else {
                return;
            };
            if m.entries.contains_key(&key) {
                return;
            }
            while bytes > self.scope.0.maximum_bytes.saturating_sub(m.retained) {
                let Some((_, old)) = m.recency.pop_first() else {
                    break;
                };
                if let Some(entry) = m.entries.remove(&old) {
                    m.retained -= entry.bytes;
                    evicted.push(entry);
                }
            }
            let sequence = m.next;
            m.next += 1;
            m.recency.insert(sequence, key.clone());
            m.entries.insert(
                key,
                Entry {
                    value: shared,
                    bytes,
                    sequence,
                },
            );
            m.retained += bytes;
        }
        drop(evicted);
        event(false);
    }
}
/// Shared variant of the typed execution boundary. The estimator must cover the
/// complete runtime allocation, including heap capacity and backend limbs.
/// Reservation precedes validation/allocation. Outside an explicit memo scope,
/// values are still shared but no hidden copy is retained for later calls.
pub fn resolve_or_compute_json_artifact_shared<T, R, Compute, Estimate, Validate>(
    request: &ArtifactExecutionCacheRequest<'_>,
    context: &ValidatedValueMemoContext,
    compute: Compute,
    estimate: Estimate,
    validate: Validate,
) -> Result<ArtifactExecutionCacheResult<SharedValidated<R>>, CacheError>
where
    T: Serialize + DeserializeOwned + 'static,
    R: Send + Sync + 'static,
    Compute: FnOnce() -> Result<(T, Vec<crate::DependencyRef>), CacheError>,
    Estimate: Fn(&T) -> u64,
    Validate: Fn(&T) -> Result<R, CacheError>,
{
    resolve_shared_runtime(
        request,
        context,
        compute,
        |value| {
            value
                .downcast_ref::<SharedValidated<R>>()
                .map(SharedValidated::retained_bytes)
        },
        |payload, resources, scope| {
            let bytes = estimate(payload).saturating_add(512);
            let reservation = reserve_runtime(resources, scope, bytes)?;
            Ok(SharedValidated(Arc::new(SharedRuntime {
                value: validate(payload)?,
                bytes,
                _reservation: reservation,
            })))
        },
    )
}

/// Authenticate and validate a typed payload, optionally memoizing only its
/// successful admission. No payload or caller-owned runtime graph is retained.
/// A budget that cannot hold the 512-byte token disables retention, not validation.
pub fn resolve_or_compute_json_artifact_admission<T, Compute, Validate>(
    request: &ArtifactExecutionCacheRequest<'_>,
    context: &ValidatedValueMemoContext,
    compute: Compute,
    validate: Validate,
) -> Result<ArtifactExecutionCacheResult<SharedValidated<()>>, CacheError>
where
    T: Serialize + DeserializeOwned + 'static,
    Compute: FnOnce() -> Result<(T, Vec<crate::DependencyRef>), CacheError>,
    Validate: Fn(&T) -> Result<(), CacheError>,
{
    resolve_shared_runtime(
        request,
        context,
        compute,
        |value| {
            let value = value.downcast_ref::<SharedValidated<()>>()?;
            // An ephemeral token never creates a zero-byte memo entry.
            value.0._reservation.as_ref().map(|_| value.0.bytes)
        },
        |payload, resources, scope| {
            validate(payload)?;
            // This optional token must not consume a validator's workspace.
            let reservation = reserve_runtime(resources, scope, 512).ok().flatten();
            Ok(SharedValidated(Arc::new(SharedRuntime {
                value: (),
                bytes: if reservation.is_some() { 512 } else { 0 },
                _reservation: reservation,
            })))
        },
    )
}

fn reserve_runtime(
    resources: Option<&RunResources>,
    scope: Option<&ValidatedValueMemo>,
    bytes: u64,
) -> Result<Option<MemoryReservation>, CacheError> {
    let Some(resources) = resources else {
        return Ok(None);
    };
    match resources.try_reserve_retained(bytes) {
        Ok(reservation) => Ok(Some(reservation)),
        Err(_) => {
            if let Some(scope) = scope {
                scope.clear();
            }
            resources
                .try_reserve_retained(bytes)
                .map(Some)
                .map_err(|e| CacheError::InvalidManifest(e.to_string()))
        }
    }
}

fn resolve_shared_runtime<T, R, Compute, Build>(
    request: &ArtifactExecutionCacheRequest<'_>,
    context: &ValidatedValueMemoContext,
    compute: Compute,
    size: fn(&dyn Any) -> Option<u64>,
    build: Build,
) -> Result<ArtifactExecutionCacheResult<SharedValidated<R>>, CacheError>
where
    T: Serialize + DeserializeOwned + 'static,
    R: Send + Sync + 'static,
    Compute: FnOnce() -> Result<(T, Vec<crate::DependencyRef>), CacheError>,
    Build: Fn(
        &T,
        Option<&RunResources>,
        Option<&ValidatedValueMemo>,
    ) -> Result<SharedValidated<R>, CacheError>,
{
    let scope = CURRENT_MEMO.with(|c| c.borrow().clone());
    let resources = scope
        .as_ref()
        .map(|s| s.0.resources.clone())
        .or_else(RunResources::current);
    let binding = scope.as_ref().map(|scope| ValidatedValueMemoBinding {
        payload_type: TypeId::of::<T>(),
        context,
        scope: scope.clone(),
        size,
    });
    let result = resolve_or_compute_json_artifact_core(
        request,
        binding.as_ref(),
        || compute().map(|(value, deps)| (value, deps, ArtifactProductionAssessment::default())),
        |payload| build(payload, resources.as_ref(), scope.as_ref()),
    )?;
    Ok(ArtifactExecutionCacheResult {
        value: match result.value {
            ValidatedExecutionValue::Decoded(_, v) | ValidatedExecutionValue::Retained(v) => v,
        },
        access: result.access,
        reused_manifest: result.reused_manifest,
        produced_manifest: result.produced_manifest,
    })
}

#[cfg(test)]
mod eviction_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn optional_unit_admission_preserves_validation_when_the_token_cannot_fit() {
        let scratch = crate::test_support::TestDir::new("optional-unit-admission");
        let store = crate::FilesystemCacheStore::new(
            "workstation",
            scratch.join("cache"),
            true,
            CacheVisibility::Local,
        );
        let resolver = CacheResolver::new(vec![crate::CacheLayer {
            precedence: 0,
            store: Box::new(store),
        }]);
        let semantic = SemanticKeyEnvelope {
            schema_version: 1,
            artifact_kind: "quadrature_rule".into(),
            mathematical_semantics_version: "admission-fixture-v1".into(),
            resolved_mathematical_parameters: serde_json::json!({"order":3}),
            normalization: None,
            target: None,
            subspace: None,
            source_data_identities: BTreeMap::new(),
            algorithm_semantics: None,
        };
        let version = ToolkitVersion::parse("0.16.0").unwrap();
        let policy = CachePolicy {
            current_toolkit_version: version.clone(),
            minimum_quality: CacheQuality::Validated,
            accepted_schema_versions: vec![1],
            allow_deprecated: false,
            allow_quarantined: false,
            allowed_visibilities: vec![CacheVisibility::Local],
        };
        let request = ArtifactExecutionCacheRequest {
            operation: "admission.fixture",
            semantic_key: &semantic,
            logical_key: "admission/fixture",
            resolver: Some(&resolver),
            reference_resolver: None,
            acceptance: Some(&policy),
            ordered_overlays: vec!["workstation".into()],
            mode: ArtifactExecutionCacheMode::PreferReuse,
            write_on_miss: true,
            write_visibility: CacheVisibility::Local,
            produced_quality: CacheQuality::Validated,
            producer_toolkit_version: version.clone(),
            minimum_reader_version: version,
            maximum_reader_version: None,
            tags: BTreeMap::new(),
            provenance_digest: None,
            production_sink: None,
        };
        let resources = RunResources::new(1024).unwrap();
        let _resources = resources.enter();
        let active = resources.try_reserve_retained(800).unwrap();
        let memo = ValidatedValueMemo::new(resources.clone(), 512);
        let _scope = memo.enter();
        let context = ValidatedValueMemoContext::exact("unit-admission-fixture");
        let calls = AtomicUsize::new(0);
        let computes = AtomicUsize::new(0);
        // Cold admission computes once and still publishes the ordinary bytes.
        let cold = resolve_or_compute_json_artifact_admission(
            &request,
            &context,
            || {
                computes.fetch_add(1, Ordering::SeqCst);
                Ok((vec![1_u32, 2, 3], vec![]))
            },
            |value| {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(value, &[1, 2, 3]);
                Ok(())
            },
        )
        .unwrap();
        let manifest = cold.produced_manifest.unwrap();
        assert_eq!(cold.value.retained_bytes(), 0);
        assert_eq!(computes.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(memo.0.memo.lock().unwrap().entries.is_empty());
        assert_eq!(resources.usage().retained_bytes, 800);

        let ordinary = resolve_or_compute_json_artifact_with_dependencies(
            &request,
            || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                panic!("warm ordinary read must not compute")
            },
            |value| {
                assert_eq!(value, &[1, 2, 3]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(ordinary.reused_manifest, Some(manifest.clone()));
        assert_eq!(ordinary.value, vec![1, 2, 3]);
        // The general shared-R API must still reject an unfunded allocation.
        assert!(resolve_or_compute_json_artifact_shared(
            &request,
            &context,
            || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                panic!("warm shared read must not compute")
            },
            |_| 0,
            |_| -> Result<(), CacheError> {
                panic!("required reservation must precede runtime validation")
            },
        )
        .is_err());
        for _ in 0..3 {
            let warm = resolve_or_compute_json_artifact_admission(
                &request,
                &context,
                || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                    panic!("warm admission must not compute")
                },
                |value| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(value, &[1, 2, 3]);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(warm.reused_manifest, Some(manifest.clone()));
            assert_eq!(warm.value.retained_bytes(), 0);
            assert!(memo.0.memo.lock().unwrap().entries.is_empty());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        let error = resolve_or_compute_json_artifact_admission(
            &request,
            &context,
            || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                panic!("a failed validation must not retry compute")
            },
            |_| Err(CacheError::InvalidManifest("domain rejection".into())),
        )
        .err()
        .expect("invalid payload must still fail without token capacity");
        assert!(error.to_string().contains("domain rejection"));
        assert_eq!(resources.usage().retained_bytes, 800);

        // Once room is available, the same API retains a charged unit token.
        drop(active);
        {
            let scratch_context = ValidatedValueMemoContext::exact("full-validator-workspace");
            let result = resolve_or_compute_json_artifact_admission(
                &request,
                &scratch_context,
                || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                    panic!("full-workspace validation must reuse the payload")
                },
                |_| {
                    let _scratch = resources
                        .reserve_working(1024)
                        .map_err(|error| CacheError::InvalidManifest(error.to_string()))?;
                    assert_eq!(resources.usage().working_bytes, 1024);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(result.value.retained_bytes(), 512);
            assert_eq!(resources.usage().working_bytes, 0);
        }
        memo.clear();
        for _ in 0..2 {
            let result = resolve_or_compute_json_artifact_admission(
                &request,
                &context,
                || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                    panic!("retained admission must not compute")
                },
                |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(result.value.retained_bytes(), 512);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        assert_eq!(memo.retained_bytes(), 512);
        assert_eq!(resources.usage().retained_bytes, 512);
        let object_path = crate::FilesystemCacheStore::new(
            "workstation",
            scratch.join("cache"),
            true,
            CacheVisibility::Local,
        )
        .object_path(&manifest.objects[0].content_digest);
        std::fs::write(object_path, b"corrupt").unwrap();
        let required = ArtifactExecutionCacheRequest {
            mode: ArtifactExecutionCacheMode::RequireReuse,
            write_on_miss: false,
            ..request
        };
        assert!(resolve_or_compute_json_artifact_admission(
            &required,
            &context,
            || -> Result<(Vec<u32>, Vec<crate::DependencyRef>), CacheError> {
                panic!("corruption must not compute")
            },
            |_| panic!("corruption must not reach domain validation"),
        )
        .is_err());
        memo.clear();
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn eviction_registration_does_not_retain_callback_owner() {
        let resources = RunResources::new(1024).unwrap();
        let memo = ValidatedValueMemo::new(resources, 512);
        let owner = Arc::new(AtomicUsize::new(0));
        let weak_owner = Arc::downgrade(&owner);
        let callback: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            owner.fetch_add(1, Ordering::SeqCst);
        });
        memo.register_eviction(&callback);
        assert_eq!(Arc::strong_count(&callback), 1);
        assert!(weak_owner.upgrade().is_some());
        drop(callback);
        assert!(weak_owner.upgrade().is_none());
        memo.clear();
        assert!(memo.0.evictions.lock().unwrap().is_empty());
    }

    #[test]
    fn clear_releases_optional_sources_but_preserves_active_reservations() {
        let resources = RunResources::new(1024).unwrap();
        let _resource_scope = resources.enter();
        let memo = ValidatedValueMemo::new(resources.clone(), 512);
        let active = SharedValidated::new(vec![1_u8], 128).unwrap();
        let optional = Arc::new(Mutex::new(vec![
            SharedValidated::new(vec![2_u8], 64).unwrap(),
            active.clone(),
        ]));
        let retained_sources = optional.clone();
        let callback: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            retained_sources.lock().unwrap().clear();
        });
        memo.register_eviction(&callback);
        let context = ValidatedValueMemoContext::exact("eviction-accounting-test");
        let binding = ValidatedValueMemoBinding {
            payload_type: TypeId::of::<Vec<u8>>(),
            context: &context,
            scope: memo.clone(),
            size: |value| {
                value
                    .downcast_ref::<SharedValidated<Vec<u8>>>()
                    .map(SharedValidated::retained_bytes)
            },
        };
        let key = ArtifactKey::new("test", "active", b"active").unwrap();
        let digest = ContentDigest::sha256(b"active");
        binding.insert(&key, &digest, 1, &context.reuse, &active);
        let memo_only = SharedValidated::new(vec![3_u8], 32).unwrap();
        binding.insert(
            &ArtifactKey::new("test", "memo-only", b"memo-only").unwrap(),
            &ContentDigest::sha256(b"memo-only"),
            1,
            &context.reuse,
            &memo_only,
        );
        drop(memo_only);
        assert_eq!(memo.retained_bytes(), 160);
        assert_eq!(resources.usage().retained_bytes, 224);
        memo.clear();
        assert!(optional.lock().unwrap().is_empty());
        assert_eq!(memo.retained_bytes(), 0);
        assert_eq!(resources.usage().retained_bytes, 128);
        assert_eq!(&**active, &[1_u8]);
        assert!(resources.try_reserve_retained(897).is_err());
        memo.clear();
        assert_eq!(resources.usage().retained_bytes, 128);
        drop(active);
        assert_eq!(resources.usage().retained_bytes, 0);
        assert_eq!(resources.usage().working_bytes, 0);
    }

    #[test]
    fn eviction_callbacks_run_outside_both_memo_locks() {
        let memo = ValidatedValueMemo::new(RunResources::new(1024).unwrap(), 512);
        let observed = Arc::new(AtomicUsize::new(0));
        let callback_memo = memo.clone();
        let callback_observed = observed.clone();
        let later: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let callback: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            // Nonblocking probes fail immediately under a lock regression,
            // before calling public APIs that would otherwise deadlock.
            assert!(callback_memo.0.memo.try_lock().is_ok());
            assert!(callback_memo.0.evictions.try_lock().is_ok());
            assert_eq!(callback_memo.retained_bytes(), 0);
            callback_memo.register_eviction(&later);
            callback_observed.fetch_add(1, Ordering::SeqCst);
        });
        memo.register_eviction(&callback);
        memo.clear();
        assert_eq!(observed.load(Ordering::SeqCst), 1);
        assert_eq!(memo.0.evictions.lock().unwrap().len(), 2);
        drop(callback);
        memo.clear();
        assert_eq!(observed.load(Ordering::SeqCst), 1);
        assert!(memo.0.evictions.lock().unwrap().is_empty());
    }
}
