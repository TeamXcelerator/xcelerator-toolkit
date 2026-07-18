// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Typed progress events for long-running research workflows.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "event")]
pub enum ProgressEvent {
    PlanCreated {
        plan_id: String,
    },
    CacheLookupStarted {
        artifact_kind: String,
        logical_key: String,
    },
    CacheHit {
        layer: String,
        content_digest: String,
    },
    CacheRejected {
        layer: String,
        reason: String,
    },
    CacheMiss {
        artifact_kind: String,
        logical_key: String,
    },
    ArtifactBuildStarted {
        artifact_kind: String,
        logical_key: String,
    },
    ArtifactCheckpoint {
        artifact_kind: String,
        completed_units: u64,
        total_units: Option<u64>,
    },
    ArtifactCompleted {
        artifact_kind: String,
        content_digest: String,
    },
    SolverIteration {
        algorithm: String,
        iteration: usize,
        diagnostics: BTreeMap<String, String>,
    },
    PrecisionEscalated {
        from_bits: u32,
        to_bits: u32,
        reason: String,
    },
    CrossCheckStarted {
        primary_algorithm: String,
        independent_algorithm: String,
    },
    CrossCheckCompared {
        accepted: bool,
        summary: String,
    },
    CertificationStarted {
        certificate_kind: String,
    },
    CertificateCompleted {
        certificate_id: String,
    },
    PublicationStaged {
        repository: String,
        artifact_count: usize,
    },
    Message {
        level: String,
        text: String,
    },
}

pub trait ProgressSink: Send + Sync {
    fn emit(&self, event: ProgressEvent);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoopProgress;

impl ProgressSink for NoopProgress {
    fn emit(&self, _event: ProgressEvent) {}
}

#[derive(Clone, Default)]
pub struct CollectingProgress {
    events: Arc<Mutex<Vec<ProgressEvent>>>,
}

impl CollectingProgress {
    pub fn events(&self) -> Vec<ProgressEvent> {
        self.events
            .lock()
            .expect("progress event lock poisoned")
            .clone()
    }

    pub fn clear(&self) {
        self.events
            .lock()
            .expect("progress event lock poisoned")
            .clear();
    }
}

impl ProgressSink for CollectingProgress {
    fn emit(&self, event: ProgressEvent) {
        self.events
            .lock()
            .expect("progress event lock poisoned")
            .push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collecting_sink_preserves_event_order() {
        let sink = CollectingProgress::default();
        sink.emit(ProgressEvent::Message {
            level: "info".to_owned(),
            text: "first".to_owned(),
        });
        sink.emit(ProgressEvent::Message {
            level: "info".to_owned(),
            text: "second".to_owned(),
        });
        let events = sink.events();
        assert_eq!(events.len(), 2);
        match &events[1] {
            ProgressEvent::Message { text, .. } => assert_eq!(text, "second"),
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
