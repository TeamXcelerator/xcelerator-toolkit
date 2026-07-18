use serde::{Deserialize, Serialize};

/// Operational completion is independent from mathematical assurance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    Successful,
    Failed,
    Cancelled,
    Inconclusive,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultStatus {
    Converged,
    Approximate,
    Failed,
    Inconclusive,
    UnresolvedCluster,
    InsufficientPrecision,
    InvalidConfiguration,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    ResidualTolerance,
    BackwardErrorTolerance,
    CertifiedEnclosure,
    UnresolvedCluster,
    MaximumIterations,
    MaximumPrecision,
    Breakdown,
    InvalidTarget,
    IndependentRoutesDisagree,
    UserCancelled,
}
