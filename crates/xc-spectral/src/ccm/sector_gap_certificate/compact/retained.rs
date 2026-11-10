//! Run-local reuse of successful exact reconstruction; never a portable trust flag.
use super::*;
use std::sync::{Arc, Mutex};

struct Entry {
    digest: ContentDigest,
    exponent_min: i32,
    exponent_max: i32,
    verified: Arc<VerifiedCompactSectorGap>,
}

/// One optional retained matrix per run, charged to its owner's available budget.
#[derive(Clone)]
pub(crate) struct Memo {
    entry: Arc<Mutex<Option<Entry>>>,
    resources: xc_core::RunResources,
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Memo>> = const { std::cell::RefCell::new(None) };
    #[cfg(test)]
    pub(super) static FRESH_REPLAYS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) struct Scope {
    previous: Option<Memo>,
    _local: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}

pub(crate) fn freshly<T>(replay: impl FnOnce() -> T) -> T {
    let _scope = Scope {
        previous: CURRENT.with(|c| c.replace(None)),
        _local: std::marker::PhantomData,
    };
    replay()
}

impl Memo {
    pub(crate) fn new(resources: xc_core::RunResources) -> Self {
        Self {
            entry: Arc::new(Mutex::new(None)),
            resources,
        }
    }
    pub(crate) fn enter(&self) -> Scope {
        Scope {
            previous: CURRENT.with(|c| c.replace(Some(self.clone()))),
            _local: std::marker::PhantomData,
        }
    }
    pub(crate) fn eviction_callback(&self) -> Arc<dyn Fn() + Send + Sync> {
        let weak = Arc::downgrade(&self.entry);
        Arc::new(move || {
            if let Some(entry) = weak.upgrade() {
                *entry
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            }
        })
    }
}

/// Internal consumers alone use this route. Public reconstruction and verification
/// always replay. The exact payload includes matrix hashes, recipe and every proof
/// semantics identifier; equality additionally checks the complete immutable value.
pub(crate) fn reconstruct(
    c: &CompactCcmSectorGapCertificate,
) -> Result<Arc<VerifiedCompactSectorGap>> {
    let memo = CURRENT.with(|m| m.borrow().clone());
    let Some(memo) = memo else {
        return reconstruct_verified_compact_sector_gap(c).map(Arc::new);
    };
    memo.resources.check_cancelled()?;
    let digest = ContentDigest::sha256_finite_json(c)?;
    let exponent_min = rug::float::exp_min();
    let exponent_max = rug::float::exp_max();
    {
        let entry = memo
            .entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = entry.as_ref().filter(|e| {
            e.digest == digest
                && e.exponent_min == exponent_min
                && e.exponent_max == exponent_max
                && e.verified.certificate() == c
        }) {
            return Ok(entry.verified.clone());
        }
    }
    // Release the previous optional allocation before admitting a different proof.
    *memo
        .entry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    let _owner = memo.resources.enter();
    let verified = Arc::new(reconstruct_verified_compact_sector_gap(c)?);
    *memo
        .entry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Entry {
        digest,
        exponent_min,
        exponent_max,
        verified: verified.clone(),
    });
    Ok(verified)
}
