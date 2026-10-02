// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Typed progress events for long-running research workflows.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};

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

/// Process-wide destination for the toolkit's human-readable progress and
/// diagnostic messages. With no sink installed (the default), each message is
/// written to stderr as one line, exactly as the libraries always have.
static MESSAGE_SINK: RwLock<Option<Arc<dyn ProgressSink>>> = RwLock::new(None);

/// Install (or with `None`, remove) the process-wide message sink and return
/// the previous one. An application can silence library messages with
/// [`NoopProgress`], capture them with [`CollectingProgress`], or forward them
/// to its own logger. Each message arrives as [`ProgressEvent::Message`] with
/// level `"info"`.
pub fn set_message_sink(sink: Option<Arc<dyn ProgressSink>>) -> Option<Arc<dyn ProgressSink>> {
    let mut slot = MESSAGE_SINK
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    std::mem::replace(&mut *slot, sink)
}

/// Deliver one library message; see [`set_message_sink`].
pub fn emit_message(text: String) {
    let sink = MESSAGE_SINK
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone();
    match sink {
        Some(sink) => sink.emit(ProgressEvent::Message {
            level: "info".to_owned(),
            text,
        }),
        None => eprintln!("{text}"),
    }
}

/// Format and deliver a library message through [`emit_message`].
#[macro_export]
macro_rules! progress_message {
    ($($arg:tt)*) => {
        $crate::emit_message(::std::format!($($arg)*))
    };
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
    fn installed_message_sink_receives_library_messages() {
        let sink = CollectingProgress::default();
        let previous = set_message_sink(Some(Arc::new(sink.clone())));
        crate::progress_message!("[HP] stage {} took {:.1}s", "tau", 1.25);
        set_message_sink(previous);
        assert!(sink.events().iter().any(|event| matches!(
            event,
            ProgressEvent::Message { level, text }
                if level == "info" && text == "[HP] stage tau took 1.2s"
        )));
    }

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
