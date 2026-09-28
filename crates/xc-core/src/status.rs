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
    UnresolvedEigenspace,
    InsufficientPrecision,
    InvalidConfiguration,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    /// A source-bound count proved that the closed requested interval is empty.
    EmptySelection,
    ResidualTolerance,
    BackwardErrorTolerance,
    /// Every requested pair passed at least one of the residual/backward-error
    /// tests, but neither test passed for the complete requested block.
    ResidualOrBackwardErrorTolerance,
    CertifiedEnclosure,
    UnresolvedCluster,
    UnresolvedEigenspace,
    MaximumIterations,
    MaximumPrecision,
    Breakdown,
    InvalidTarget,
    IndependentRoutesDisagree,
    UserCancelled,
}
