// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Resource policy, dry-run estimates, and cooperative cancellation.

use crate::config_resolution::{canonical_json, sha256_hex};
use crate::{ConfigDigest, ConfigError};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const GIB: u64 = 1024 * 1024 * 1024;
const TIB: u64 = 1024 * GIB;

/// Scheduling profile. Profiles supply planning defaults only; they never
/// change the mathematical target, precision, or requested assurance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceProfile {
    NormalWorkstation,
    HighMemoryWorkstation,
    ExternalCompute,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Memory,
    TemporaryDisk,
    PermanentDisk,
    NetworkTransfer,
    CpuTime,
    WallTime,
    Threads,
}

impl Display for ResourceKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Memory => "memory",
            Self::TemporaryDisk => "temporary disk",
            Self::PermanentDisk => "permanent disk",
            Self::NetworkTransfer => "network transfer",
            Self::CpuTime => "CPU time",
            Self::WallTime => "wall time",
            Self::Threads => "threads",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourcePolicy {
    pub profile: ResourceProfile,
    pub maximum_memory_bytes: Option<u64>,
    pub maximum_temporary_disk_bytes: Option<u64>,
    pub maximum_permanent_disk_bytes: Option<u64>,
    pub maximum_transfer_bytes: Option<u64>,
    pub maximum_cpu_seconds: Option<u64>,
    pub maximum_wall_seconds: Option<u64>,
    pub maximum_threads: Option<usize>,
    pub allow_out_of_core: bool,
    pub allow_distributed: bool,
    pub checkpoint_interval_seconds: Option<u64>,
}

impl ResourcePolicy {
    /// The resolved values, rather than the profile name alone, belong in the
    /// run fingerprint.
    pub fn for_profile(profile: ResourceProfile) -> Self {
        match profile {
            ResourceProfile::NormalWorkstation => Self {
                profile,
                maximum_memory_bytes: Some(16 * GIB),
                maximum_temporary_disk_bytes: Some(100 * GIB),
                maximum_permanent_disk_bytes: Some(20 * GIB),
                maximum_transfer_bytes: Some(20 * GIB),
                maximum_cpu_seconds: Some(4 * 24 * 60 * 60),
                maximum_wall_seconds: Some(24 * 60 * 60),
                maximum_threads: Some(8),
                allow_out_of_core: false,
                allow_distributed: false,
                checkpoint_interval_seconds: Some(15 * 60),
            },
            ResourceProfile::HighMemoryWorkstation => Self {
                profile,
                maximum_memory_bytes: Some(128 * GIB),
                maximum_temporary_disk_bytes: Some(TIB),
                maximum_permanent_disk_bytes: Some(250 * GIB),
                maximum_transfer_bytes: Some(250 * GIB),
                maximum_cpu_seconds: Some(28 * 24 * 60 * 60),
                maximum_wall_seconds: Some(7 * 24 * 60 * 60),
                maximum_threads: Some(64),
                allow_out_of_core: true,
                allow_distributed: false,
                checkpoint_interval_seconds: Some(10 * 60),
            },
            ResourceProfile::ExternalCompute => Self {
                profile,
                maximum_memory_bytes: Some(TIB),
                maximum_temporary_disk_bytes: Some(10 * TIB),
                maximum_permanent_disk_bytes: Some(2 * TIB),
                maximum_transfer_bytes: Some(2 * TIB),
                maximum_cpu_seconds: Some(448 * 24 * 60 * 60),
                maximum_wall_seconds: Some(14 * 24 * 60 * 60),
                maximum_threads: Some(256),
                allow_out_of_core: true,
                allow_distributed: true,
                checkpoint_interval_seconds: Some(5 * 60),
            },
        }
    }

    pub fn digest(&self) -> Result<ConfigDigest, ConfigError> {
        let value = serde_json::to_value(self).map_err(|error| {
            ConfigError::new(format!("resource policy serialization failed: {error}"))
        })?;
        let canonical = canonical_json(&value).map_err(|error| {
            ConfigError::new(format!("resource policy canonicalization failed: {error}"))
        })?;
        Ok(ConfigDigest(sha256_hex(canonical.as_bytes())))
    }
}

impl Default for ResourcePolicy {
    fn default() -> Self {
        Self::for_profile(ResourceProfile::NormalWorkstation)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEstimate {
    pub operator_dimension: usize,
    pub resident_memory_bytes: Option<u64>,
    pub temporary_memory_bytes: Option<u64>,
    pub temporary_disk_bytes: Option<u64>,
    pub persistent_artifact_bytes: Option<u64>,
    pub transfer_bytes: Option<u64>,
    pub estimated_cpu_seconds: Option<u64>,
    pub estimated_wall_seconds: Option<u64>,
    pub requested_threads: Option<usize>,
    pub estimated_operator_applications: Option<u64>,
    pub estimated_factorizations: Option<u64>,
    pub time_class: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceViolation {
    pub resource: ResourceKind,
    pub estimated: u64,
    pub maximum: u64,
    pub unit: String,
}

impl Display for ResourceViolation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "estimated {} {} {} exceeds policy {} {}",
            self.resource, self.estimated, self.unit, self.maximum, self.unit
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FeasibilityReport {
    pub profile: ResourceProfile,
    pub feasible: bool,
    pub estimate: ResourceEstimate,
    pub violations: Vec<ResourceViolation>,
    /// Bounded resources for which the planner did not provide an estimate.
    pub unestimated: Vec<ResourceKind>,
}

impl ResourcePolicy {
    pub fn assess(&self, estimate: ResourceEstimate) -> FeasibilityReport {
        let mut violations = Vec::new();
        let mut unestimated = Vec::new();

        let memory = match (
            estimate.resident_memory_bytes,
            estimate.temporary_memory_bytes,
        ) {
            (Some(resident), Some(temporary)) => Some(resident.saturating_add(temporary)),
            (Some(resident), None) => Some(resident),
            (None, Some(temporary)) => Some(temporary),
            (None, None) => None,
        };
        assess_u64(
            ResourceKind::Memory,
            memory,
            self.maximum_memory_bytes,
            "bytes",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::TemporaryDisk,
            estimate.temporary_disk_bytes,
            self.maximum_temporary_disk_bytes,
            "bytes",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::PermanentDisk,
            estimate.persistent_artifact_bytes,
            self.maximum_permanent_disk_bytes,
            "bytes",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::NetworkTransfer,
            estimate.transfer_bytes,
            self.maximum_transfer_bytes,
            "bytes",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::CpuTime,
            estimate.estimated_cpu_seconds,
            self.maximum_cpu_seconds,
            "seconds",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::WallTime,
            estimate.estimated_wall_seconds,
            self.maximum_wall_seconds,
            "seconds",
            &mut violations,
            &mut unestimated,
        );
        assess_u64(
            ResourceKind::Threads,
            estimate.requested_threads.map(|value| value as u64),
            self.maximum_threads.map(|value| value as u64),
            "threads",
            &mut violations,
            &mut unestimated,
        );

        FeasibilityReport {
            profile: self.profile,
            feasible: violations.is_empty(),
            estimate,
            violations,
            unestimated,
        }
    }
}

fn assess_u64(
    resource: ResourceKind,
    estimated: Option<u64>,
    maximum: Option<u64>,
    unit: &str,
    violations: &mut Vec<ResourceViolation>,
    unestimated: &mut Vec<ResourceKind>,
) {
    match (estimated, maximum) {
        (Some(estimated), Some(maximum)) if estimated > maximum => {
            violations.push(ResourceViolation {
                resource,
                estimated,
                maximum,
                unit: unit.to_owned(),
            });
        }
        (None, Some(_)) => unestimated.push(resource),
        _ => {}
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "resource")]
pub enum CancellationReason {
    UserRequested,
    ResourceBudgetReached(ResourceKind),
    Shutdown,
    Superseded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancellationState {
    pub requested: bool,
    pub reason: Option<CancellationReason>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancellationError {
    pub reason: CancellationReason,
}

impl Display for CancellationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation cancelled: {:?}", self.reason)
    }
}

impl Error for CancellationError {}

/// Cheaply cloned token polled at safe operation boundaries.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    requested: Arc<AtomicBool>,
    reason: Arc<Mutex<Option<CancellationReason>>>,
    wall_deadline: Arc<Option<Instant>>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            reason: Arc::new(Mutex::new(None)),
            wall_deadline: Arc::new(None),
        }
    }

    /// Create a token whose ordinary cooperative checks also enforce the
    /// policy's elapsed wall-time budget.
    pub fn for_policy(policy: &ResourcePolicy) -> Self {
        match policy.maximum_wall_seconds {
            Some(seconds) => Self::with_wall_time_limit(Duration::from_secs(seconds)),
            None => Self::new(),
        }
    }

    /// Create a token with a wall-time deadline relative to now.
    pub fn with_wall_time_limit(limit: Duration) -> Self {
        let now = Instant::now();
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            reason: Arc::new(Mutex::new(None)),
            wall_deadline: Arc::new(now.checked_add(limit)),
        }
    }

    /// Returns true only for the first request accepted by this token.
    pub fn cancel(&self, reason: CancellationReason) -> bool {
        let mut current = self
            .reason
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.requested.load(Ordering::Acquire) {
            return false;
        }
        *current = Some(reason);
        self.requested.store(true, Ordering::Release);
        true
    }

    pub fn is_cancelled(&self) -> bool {
        self.enforce_wall_deadline();
        self.requested.load(Ordering::Acquire)
    }

    fn enforce_wall_deadline(&self) {
        if self.requested.load(Ordering::Acquire) {
            return;
        }
        if self
            .wall_deadline
            .as_ref()
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.cancel(CancellationReason::ResourceBudgetReached(
                ResourceKind::WallTime,
            ));
        }
    }

    pub fn state(&self) -> CancellationState {
        CancellationState {
            requested: self.is_cancelled(),
            reason: self
                .reason
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        }
    }

    pub fn check(&self) -> Result<(), CancellationError> {
        if !self.is_cancelled() {
            return Ok(());
        }
        let reason = self
            .state()
            .reason
            .unwrap_or(CancellationReason::UserRequested);
        Err(CancellationError { reason })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_resolved_budget_is_enforced() {
        let policy = ResourcePolicy {
            maximum_memory_bytes: Some(100),
            maximum_temporary_disk_bytes: Some(100),
            maximum_permanent_disk_bytes: Some(100),
            maximum_transfer_bytes: Some(100),
            maximum_cpu_seconds: Some(100),
            maximum_wall_seconds: Some(100),
            maximum_threads: Some(2),
            ..ResourcePolicy::default()
        };
        let estimate = ResourceEstimate {
            resident_memory_bytes: Some(80),
            temporary_memory_bytes: Some(30),
            temporary_disk_bytes: Some(101),
            persistent_artifact_bytes: Some(102),
            transfer_bytes: Some(103),
            estimated_cpu_seconds: Some(104),
            estimated_wall_seconds: Some(105),
            requested_threads: Some(3),
            ..ResourceEstimate::default()
        };
        let report = policy.assess(estimate);
        assert!(!report.feasible);
        assert_eq!(report.violations.len(), 7);
    }

    #[test]
    fn named_profiles_change_limits_not_mathematics() {
        let normal = ResourcePolicy::for_profile(ResourceProfile::NormalWorkstation);
        let high_memory = ResourcePolicy::for_profile(ResourceProfile::HighMemoryWorkstation);
        let external = ResourcePolicy::for_profile(ResourceProfile::ExternalCompute);
        assert!(high_memory.maximum_memory_bytes > normal.maximum_memory_bytes);
        assert!(external.maximum_memory_bytes > normal.maximum_memory_bytes);
        assert!(external.allow_distributed);
        assert_ne!(normal.digest().unwrap(), external.digest().unwrap());

        let unchanged_workload = ResourceEstimate {
            operator_dimension: 250_000,
            resident_memory_bytes: Some(64 * GIB),
            temporary_memory_bytes: Some(4 * GIB),
            temporary_disk_bytes: Some(200 * GIB),
            persistent_artifact_bytes: Some(10 * GIB),
            transfer_bytes: Some(GIB),
            estimated_cpu_seconds: Some(24 * 60 * 60),
            estimated_wall_seconds: Some(12 * 60 * 60),
            requested_threads: Some(8),
            time_class: "large-dense".to_owned(),
            ..ResourceEstimate::default()
        };
        let normal_report = normal.assess(unchanged_workload.clone());
        let high_memory_report = high_memory.assess(unchanged_workload.clone());
        assert!(!normal_report.feasible);
        assert!(high_memory_report.feasible);
        assert_eq!(normal_report.estimate, unchanged_workload);
        assert_eq!(high_memory_report.estimate, unchanged_workload);
    }

    #[test]
    fn cancellation_is_shared_and_idempotent() {
        let token = CancellationToken::new();
        let worker = token.clone();
        assert!(token.cancel(CancellationReason::ResourceBudgetReached(
            ResourceKind::WallTime
        )));
        assert!(!token.cancel(CancellationReason::UserRequested));
        assert!(worker.is_cancelled());
        assert_eq!(
            worker.check().unwrap_err().reason,
            CancellationReason::ResourceBudgetReached(ResourceKind::WallTime)
        );
    }

    #[test]
    fn policy_wall_deadline_becomes_an_actionable_shared_cancellation() {
        let policy = ResourcePolicy {
            maximum_wall_seconds: Some(0),
            ..ResourcePolicy::default()
        };
        let token = CancellationToken::for_policy(&policy);
        let worker = token.clone();
        assert!(worker.is_cancelled());
        assert_eq!(
            token.check().unwrap_err().reason,
            CancellationReason::ResourceBudgetReached(ResourceKind::WallTime)
        );
        assert_eq!(worker.state(), token.state());
    }
}
