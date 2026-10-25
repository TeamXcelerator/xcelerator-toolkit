//! Cooperative run-owned reservations for retained allocations and workspaces.
//! Byte estimates belong to allocating callers; this is not an OS RSS limit.
use crate::ConfigError;
use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex};

#[derive(Clone, Debug)]
pub struct RunResources(Arc<Inner>, bool);
#[derive(Debug)]
struct Inner {
    maximum_bytes: u64,
    worker_limit: usize,
    state: Mutex<Usage>,
    released: Condvar,
    cancelled: std::sync::atomic::AtomicBool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunResourceUsage {
    pub retained_bytes: u64,
    pub working_bytes: u64,
    pub peak_bytes: u64,
}
#[derive(Debug, Default)]
struct Usage {
    retained: u64,
    working: u64,
    independent: u64,
    peak: u64,
    owners: std::collections::HashMap<std::thread::ThreadId, usize>,
}
thread_local! { static CURRENT: RefCell<Option<RunResources>> = const { RefCell::new(None) }; }

/// Restores the previous run, including during unwinding. Kept on its owner thread.
pub struct RunResourceScope {
    previous: Option<RunResources>,
    _local: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for RunResourceScope {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}
/// One allocation reservation. Share its owner rather than reserving each Arc clone.
#[derive(Debug)]
pub struct MemoryReservation {
    resources: RunResources,
    bytes: u64,
    working: bool,
    owner: std::thread::ThreadId,
}

/// A complete, independently progressing task budget. Its owner must use a
/// separate resource ledger and must not wait on work in this ledger. Dropping
/// it releases the whole task budget, including its temporary retained data.
pub struct IndependentReservation {
    resources: RunResources,
    bytes: u64,
}
impl Drop for IndependentReservation {
    fn drop(&mut self) {
        let mut state = self
            .resources
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.working -= self.bytes;
        state.independent -= self.bytes;
        self.resources.0.released.notify_all();
    }
}
impl MemoryReservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}
impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let mut state = self
            .resources
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.working {
            state.working -= self.bytes;
            let count = state
                .owners
                .get_mut(&self.owner)
                .expect("reservation owner");
            *count -= 1;
            if *count == 0 {
                state.owners.remove(&self.owner);
            }
        } else {
            state.retained -= self.bytes;
        }
        self.resources.0.released.notify_all();
    }
}
impl RunResources {
    pub fn new(maximum_bytes: u64) -> Result<Self, ConfigError> {
        Self::with_worker_limit(maximum_bytes, usize::MAX)
    }
    /// Limit on the numerical pool, not on all process threads.
    /// Owners and prefetch coordinators may also perform serial numerical work.
    pub fn with_worker_limit(maximum_bytes: u64, worker_limit: usize) -> Result<Self, ConfigError> {
        if worker_limit == 0 {
            return Err(ConfigError::new("run worker limit must be positive"));
        }
        if maximum_bytes == 0 {
            return Err(ConfigError::new("run memory budget must be positive"));
        }
        Ok(Self(
            Arc::new(Inner {
                maximum_bytes,
                worker_limit,
                state: Mutex::new(Usage::default()),
                released: Condvar::new(),
                cancelled: std::sync::atomic::AtomicBool::new(false),
            }),
            false,
        ))
    }
    pub fn current() -> Option<Self> {
        CURRENT.with(|c| c.borrow().clone())
    }
    pub fn enter(&self) -> RunResourceScope {
        let previous = CURRENT.with(|c| c.replace(Some(self.clone())));
        RunResourceScope {
            previous,
            _local: std::marker::PhantomData,
        }
    }
    pub fn check_workers(&self, actual: usize) -> Result<(), ConfigError> {
        if actual > self.0.worker_limit {
            return Err(ConfigError::new(format!(
                "run worker budget exceeded: actual={actual}, maximum={}",
                self.0.worker_limit
            )));
        }
        Ok(())
    }
    pub fn worker_limit(&self) -> usize {
        self.0.worker_limit
    }
    pub fn maximum_bytes(&self) -> u64 {
        self.0.maximum_bytes
    }
    /// Request cooperative cancellation of this ledger's work, for example an
    /// independent preparation whose foreground prerequisite failed. Other run
    /// ledgers are unaffected. This does not discard any completed artifact.
    pub fn cancel(&self) {
        let _state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.0
            .cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        self.0.released.notify_all();
    }
    pub fn check_cancelled(&self) -> Result<(), ConfigError> {
        if self.0.cancelled.load(std::sync::atomic::Ordering::Acquire) {
            Err(ConfigError::new("run work cancelled by owner"))
        } else {
            Ok(())
        }
    }
    /// Admit an independent task without waiting. Use its own capped ledger
    /// and dedicated worker so it can finish even while foreground workers wait.
    pub fn try_reserve_independent(
        &self,
        bytes: u64,
    ) -> Result<IndependentReservation, ConfigError> {
        let mut s = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.check_cancelled()?;
        let used = s.retained + s.working;
        if bytes == 0 || bytes > self.maximum_bytes() - used {
            return Err(ConfigError::new(
                "independent task exceeds available run budget",
            ));
        }
        s.working += bytes;
        s.independent += bytes;
        s.peak = s.peak.max(used + bytes);
        Ok(IndependentReservation {
            resources: self.clone(),
            bytes,
        })
    }
    /// Foreground view that waits for independent tasks when releasing their
    /// reservations alone would make a request fit. This applies even to nested
    /// and retained requests, so speculative overlap cannot introduce a refusal.
    /// Other pressure retains the usual nonblocking/deadlock-avoidance behavior.
    pub fn waiting_for_independent(&self) -> Self {
        Self(self.0.clone(), true)
    }
    pub fn usage(&self) -> RunResourceUsage {
        let s = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        RunResourceUsage {
            retained_bytes: s.retained,
            working_bytes: s.working,
            peak_bytes: s.peak,
        }
    }
    pub fn try_reserve_retained(&self, bytes: u64) -> Result<MemoryReservation, ConfigError> {
        self.reserve(bytes, false, false)
    }
    /// Wait only for active work to release memory. Retained results never cause
    /// an indefinite wait: if they prevent admission, the caller gets an error.
    pub fn reserve_working(&self, bytes: u64) -> Result<MemoryReservation, ConfigError> {
        self.reserve(bytes, true, true)
    }
    pub fn try_reserve_working(&self, bytes: u64) -> Result<MemoryReservation, ConfigError> {
        self.reserve(bytes, true, false)
    }
    fn reserve(
        &self,
        bytes: u64,
        working: bool,
        wait: bool,
    ) -> Result<MemoryReservation, ConfigError> {
        let mut s = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let owner = std::thread::current().id();
        loop {
            self.check_cancelled()?;
            let used = s
                .retained
                .checked_add(s.working)
                .expect("admitted run accounting");
            if bytes <= self.0.maximum_bytes - used {
                if working {
                    s.working += bytes;
                    *s.owners.entry(owner).or_default() += 1;
                } else {
                    s.retained += bytes;
                }
                s.peak = s.peak.max(used + bytes);
                return Ok(MemoryReservation {
                    resources: self.clone(),
                    bytes,
                    working,
                    owner,
                });
            }
            let independent_can_admit = self.1
                && s.independent > 0
                && bytes <= self.0.maximum_bytes - (used - s.independent);
            if !independent_can_admit
                && (!wait
                    || s.owners.contains_key(&owner)
                    || s.working == 0
                    || bytes > self.0.maximum_bytes.saturating_sub(s.retained))
            {
                return Err(ConfigError::new(format!("run memory budget exceeded: requested={bytes}, retained={}, working={}, maximum={}",s.retained,s.working,self.0.maximum_bytes)));
            }
            s = self
                .0
                .released
                .wait(s)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_task_cannot_turn_foreground_admission_into_a_failure() {
        let base = RunResources::new(100).unwrap();
        let r = base.waiting_for_independent();
        let _own = r.reserve_working(20).unwrap();
        let background = r.try_reserve_independent(50).unwrap();
        assert!(base.try_reserve_retained(40).is_err());
        // Still refuse requests that cannot fit even after the independent task.
        assert!(r.try_reserve_retained(81).is_err());
        std::thread::scope(|s| {
            s.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                drop(background);
            });
            // Nested, nonblocking retained reservation waits only for independent
            // work, which has its own worker and never needs this ledger.
            let allocation = r.try_reserve_retained(80).unwrap();
            assert_eq!(allocation.bytes(), 80);
        });
        assert_eq!(r.usage().working_bytes, 20);
        assert_eq!(r.usage().retained_bytes, 0);
    }
    #[test]
    fn cancellation_wakes_memory_waiters_without_cancelling_other_ledgers() {
        let r = RunResources::new(100).unwrap();
        let _held = r.reserve_working(90).unwrap();
        let child = r.clone();
        let t = std::thread::spawn(move || child.reserve_working(50));
        r.cancel();
        assert!(t
            .join()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
        assert!(RunResources::new(100).unwrap().reserve_working(50).is_ok());
    }
    #[test]
    fn reservations_compose_and_release_across_threads() {
        let r = RunResources::new(100).unwrap();
        let retained = r.try_reserve_retained(40).unwrap();
        let work = r.reserve_working(50).unwrap();
        assert!(r.try_reserve_working(11).is_err());
        assert!(r.try_reserve_retained(11).is_err());
        let child = r.clone();
        let t = std::thread::spawn(move || child.reserve_working(60).unwrap());
        drop(work);
        let second = t.join().unwrap();
        assert_eq!(r.usage().peak_bytes, 100);
        drop(second);
        drop(retained);
        assert_eq!(r.usage().retained_bytes + r.usage().working_bytes, 0);
    }
    #[test]
    fn nested_work_never_waits_on_itself() {
        let r = RunResources::new(100).unwrap();
        let _first = r.reserve_working(60).unwrap();
        assert!(r.reserve_working(50).is_err());
        assert!(r.reserve_working(40).is_ok());
    }
    #[test]
    fn retained_pressure_fails_instead_of_waiting_forever() {
        let bounded = RunResources::with_worker_limit(100, 2).unwrap();
        assert!(bounded.check_workers(2).is_ok());
        assert!(bounded.check_workers(3).is_err());
        let r = RunResources::new(100).unwrap();
        let _a = r.try_reserve_retained(90).unwrap();
        assert!(r.reserve_working(11).is_err());
        let _scope = r.enter();
        assert_eq!(RunResources::current().unwrap().maximum_bytes(), 100);
    }
}
