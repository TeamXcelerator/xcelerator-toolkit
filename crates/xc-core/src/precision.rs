use crate::ConfigError;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// How working precision may increase when verification checks fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "value")]
pub enum PrecisionEscalation {
    Fixed,
    AddBits(u32),
    Multiply { numerator: u32, denominator: u32 },
}

impl Default for PrecisionEscalation {
    fn default() -> Self {
        Self::AddBits(256)
    }
}

/// Explicit arbitrary-precision policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionPolicy {
    pub initial_bits: u32,
    pub maximum_bits: u32,
    pub guard_bits: u32,
    pub escalation: PrecisionEscalation,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            initial_bits: 256,
            maximum_bits: 8192,
            guard_bits: 64,
            escalation: PrecisionEscalation::AddBits(256),
        }
    }
}

impl PrecisionPolicy {
    pub fn fixed(bits: u32) -> Self {
        Self {
            initial_bits: bits,
            maximum_bits: bits,
            guard_bits: 0,
            escalation: PrecisionEscalation::Fixed,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.initial_bits < 32 {
            return Err(ConfigError::new("initial_bits must be at least 32"));
        }
        if self.maximum_bits < self.initial_bits {
            return Err(ConfigError::new(
                "maximum_bits must be greater than or equal to initial_bits",
            ));
        }
        match self.escalation {
            PrecisionEscalation::Fixed => {}
            PrecisionEscalation::AddBits(0) => {
                return Err(ConfigError::new("AddBits escalation must be positive"));
            }
            PrecisionEscalation::Multiply {
                numerator,
                denominator,
            } if denominator == 0 || numerator <= denominator => {
                return Err(ConfigError::new(
                    "Multiply escalation must have numerator > denominator > 0",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn next_bits(&self, current_bits: u32) -> Option<u32> {
        if current_bits >= self.maximum_bits {
            return None;
        }
        let candidate = match self.escalation {
            PrecisionEscalation::Fixed => return None,
            PrecisionEscalation::AddBits(bits) => current_bits.saturating_add(bits),
            PrecisionEscalation::Multiply {
                numerator,
                denominator,
            } => {
                current_bits
                    .saturating_mul(numerator)
                    .saturating_add(denominator - 1)
                    / denominator
            }
        };
        Some(candidate.min(self.maximum_bits))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum AdaptivePrecisionDecision {
    Accepted { diagnostic: String },
    Retry { reason: String },
    TerminalInconclusive { reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptivePrecisionEvaluation<T> {
    pub value: T,
    pub decision: AdaptivePrecisionDecision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptivePrecisionAttempt {
    pub attempt_index: usize,
    pub precision_bits: u32,
    pub decision: AdaptivePrecisionDecision,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum AdaptivePrecisionOutcome<T> {
    Accepted {
        value: T,
        attempts: Vec<AdaptivePrecisionAttempt>,
    },
    Inconclusive {
        last_value: T,
        attempts: Vec<AdaptivePrecisionAttempt>,
        reason: String,
    },
}

impl<T> AdaptivePrecisionOutcome<T> {
    pub fn attempts(&self) -> &[AdaptivePrecisionAttempt] {
        match self {
            Self::Accepted { attempts, .. } | Self::Inconclusive { attempts, .. } => attempts,
        }
    }

    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

#[derive(Debug)]
pub enum AdaptivePrecisionRunError<E> {
    InvalidPolicy(ConfigError),
    Attempt(E),
}

impl<E: Display> Display for AdaptivePrecisionRunError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy(error) => write!(formatter, "invalid precision policy: {error}"),
            Self::Attempt(error) => write!(formatter, "adaptive precision attempt failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for AdaptivePrecisionRunError<E> {}

/// Run a domain computation under one reusable precision-escalation contract.
/// The first working precision includes guard bits; later attempts escalate
/// the actual preceding working precision. Every completed verification
/// decision is retained, including the last retry at the precision ceiling.
pub fn run_adaptive_precision<T, E, F>(
    policy: &PrecisionPolicy,
    mut attempt: F,
) -> Result<AdaptivePrecisionOutcome<T>, AdaptivePrecisionRunError<E>>
where
    F: FnMut(u32) -> Result<AdaptivePrecisionEvaluation<T>, E>,
{
    policy
        .validate()
        .map_err(AdaptivePrecisionRunError::InvalidPolicy)?;
    let mut precision_bits = policy
        .initial_bits
        .saturating_add(policy.guard_bits)
        .min(policy.maximum_bits);
    let mut attempts = Vec::new();
    loop {
        let evaluation = attempt(precision_bits).map_err(AdaptivePrecisionRunError::Attempt)?;
        let attempt_index = attempts.len() + 1;
        attempts.push(AdaptivePrecisionAttempt {
            attempt_index,
            precision_bits,
            decision: evaluation.decision.clone(),
        });
        match evaluation.decision {
            AdaptivePrecisionDecision::Accepted { .. } => {
                return Ok(AdaptivePrecisionOutcome::Accepted {
                    value: evaluation.value,
                    attempts,
                });
            }
            AdaptivePrecisionDecision::TerminalInconclusive { reason } => {
                return Ok(AdaptivePrecisionOutcome::Inconclusive {
                    last_value: evaluation.value,
                    attempts,
                    reason,
                });
            }
            AdaptivePrecisionDecision::Retry { reason } => {
                let Some(next_bits) = policy.next_bits(precision_bits) else {
                    return Ok(AdaptivePrecisionOutcome::Inconclusive {
                        last_value: evaluation.value,
                        attempts,
                        reason: format!(
                            "precision ceiling {} reached after retry request: {reason}",
                            policy.maximum_bits
                        ),
                    });
                };
                precision_bits = next_bits;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_policy_escalates_and_caps() {
        let p = PrecisionPolicy {
            initial_bits: 128,
            maximum_bits: 512,
            guard_bits: 32,
            escalation: PrecisionEscalation::Multiply {
                numerator: 3,
                denominator: 2,
            },
        };
        assert_eq!(p.next_bits(128), Some(192));
        assert_eq!(p.next_bits(400), Some(512));
        assert_eq!(p.next_bits(512), None);
    }

    #[test]
    fn generic_adaptive_runner_applies_guard_escalates_and_records_acceptance() {
        let policy = PrecisionPolicy {
            initial_bits: 64,
            maximum_bits: 256,
            guard_bits: 32,
            escalation: PrecisionEscalation::Multiply {
                numerator: 2,
                denominator: 1,
            },
        };
        let outcome = run_adaptive_precision(&policy, |precision_bits| {
            Ok::<_, std::convert::Infallible>(AdaptivePrecisionEvaluation {
                value: precision_bits,
                decision: if precision_bits == 256 {
                    AdaptivePrecisionDecision::Accepted {
                        diagnostic: "verification passed".to_owned(),
                    }
                } else {
                    AdaptivePrecisionDecision::Retry {
                        reason: "verification margin is insufficient".to_owned(),
                    }
                },
            })
        })
        .unwrap();
        assert!(outcome.is_accepted());
        assert_eq!(
            outcome
                .attempts()
                .iter()
                .map(|attempt| attempt.precision_bits)
                .collect::<Vec<_>>(),
            vec![96, 192, 256]
        );
        let decoded: AdaptivePrecisionOutcome<u32> =
            serde_json::from_slice(&serde_json::to_vec(&outcome).unwrap()).unwrap();
        assert_eq!(decoded, outcome);
    }

    #[test]
    fn generic_adaptive_runner_preserves_ceiling_and_terminal_evidence() {
        let policy = PrecisionPolicy {
            initial_bits: 64,
            maximum_bits: 192,
            guard_bits: 32,
            escalation: PrecisionEscalation::Multiply {
                numerator: 2,
                denominator: 1,
            },
        };
        let outcome = run_adaptive_precision(&policy, |precision_bits| {
            Ok::<_, std::convert::Infallible>(AdaptivePrecisionEvaluation {
                value: format!("evidence-at-{precision_bits}"),
                decision: AdaptivePrecisionDecision::Retry {
                    reason: "interval remains unresolved".to_owned(),
                },
            })
        })
        .unwrap();
        let AdaptivePrecisionOutcome::Inconclusive {
            last_value,
            attempts,
            reason,
        } = outcome
        else {
            panic!("precision-ceiling fixture was unexpectedly accepted");
        };
        assert_eq!(last_value, "evidence-at-192");
        assert_eq!(attempts.len(), 2);
        assert!(reason.contains("precision ceiling 192"));

        let terminal = run_adaptive_precision(&PrecisionPolicy::fixed(128), |_| {
            Ok::<_, std::convert::Infallible>(AdaptivePrecisionEvaluation {
                value: "singular metric",
                decision: AdaptivePrecisionDecision::TerminalInconclusive {
                    reason: "positive-definite precondition failed".to_owned(),
                },
            })
        })
        .unwrap();
        assert_eq!(terminal.attempts().len(), 1);
        assert!(!terminal.is_accepted());
    }
}
