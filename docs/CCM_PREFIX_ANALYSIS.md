# Retained-source prefix diagnostics (0.15.0)

## Boundary

This is **additional analysis of retained point matrices**, not a new CCM
root/eigenstate solver, a fitted convergence predictor, or an interval proof.
The original Tau matrices, eigenpairs, roots, and numerical payload identities
are not changed by enabling it. Source acquisition and diagnostic computation
are separate: no missing source is silently reconstructed.

`xc_numerics::prefix::analyze_prefixes` computes one fixed-order unpivoted LDLT
factorization and its inverse-transpose columns. The storage is packed by
lower row and innovation column. Total arithmetic is O(D^3), working storage
O(D^2). It does not slice a pivoted LU factorization or calculate a fresh
inverse/eigensystem at every prefix. It stops at a nonpositive, nonfinite, or
numerically unresolved computed pivot. The pivot screen is a declared policy,
not an interval error bound. No reordering, absolute-value repair, shift,
regularization, or source precision inflation is hidden in the operation.

The factor and back-substitution dependencies retain their original order.
Independent innovation Gram cross terms use indexed parallel blocks with
reusable MPFR scratch. Products and divisions retain separate rounding, and
the final adjacent-pair sum retains the same tree, including odd tails.
Independent lower-factor entries in each accepted column run in parallel
for sufficiently large matrices with multiple workers. Pivots and stop screens
remain ordered. Each entry keeps its original inner sum; long inner sums use
fixed power-of-two blocks when column work is too small to distribute. After factorization,
independent innovation columns are assigned to workers; each back-substitution
retains its serial inner tree. Worker count does not change rounding or
numerical payloads. See [convergence research](PREFIX_CONVERGENCE.md) for derived
observables, cancellation measurements, local shard reads and nesting checks.

## Definitions

Dimension k is not mode index N. For an even sector, k=N+1, with coefficient
order zero mode followed by increasing positive modes in the orthonormal
reflection-even basis. The full Tau storage must first be projected correctly;
its raw upper-left block is not this ordered even-sector prefix.

For A_k = [[A_(k-1), b], [b^T, a]], define

    w = A_(k-1)^(-1) b
    v_k = (-w, 1)
    sigma_k = a - b^T w
    M_k = v_k^T v_k
    I_k = M_k / sigma_k

The inverse update is diag(A_(k-1)^(-1),0) + v_k v_k^T / sigma_k.
Writing T1=tr(A_k^(-1)) and T2=tr(A_k^(-2)), the increments are

    T1_k = T1_(k-1) + I_k
    T2_k = T2_(k-1) + I_k^2
           + 2 sum_(i<k) (v_i^T v_k)^2/(sigma_i sigma_k).

Shorter innovations are extended by zeros. Column k of L^(-T) supplies v_k.
The engine also records T1^2/T2, I_k/T1, and the computed expressions

    1/sqrt(T2) <= smallest_eigenvalue <= T1/T2.

These inequalities hold for the exact positive-definite matrix and exact
moments. Rounded evaluations are explicitly **estimates**, not certified
bounds. Eigenvalue depth is -log10(eigenvalue), not root matching depth or
profile distance. Negative depths are meaningful and are not clamped.
A generalized Gram metric is deliberately not accepted by this API. For
nonorthonormal coordinates the relevant moments involve A^(-1)G, not A^(-1).
The separate generalized one-border research diagnostic remains available.

## Public API and retention

1. Resolve an approved retained source using the existing Toolkit resolver.
   Supply its runtime `ArtifactManifest`, exact logical `payload.json` bytes,
   and the campaign's explicit approved payload-digest list to
   `RetainedEvenMatrix::from_payload`. Canonical warehouse manifests must first
   be resolved through the existing adapter; do not confuse canonical-package
   digests with the logical payload digest in the runtime manifest.
2. Supply optional `RetainedEvenEigenpair` checkpoint sources in the same way.
   They must be exactly even in the stored full-coordinate representation.
   The export creates a normalized copy, never edits the retained eigenstate.
3. Call `analyze_retained_prefixes`, or
   `analyze_retained_prefixes_via_cache` with a separate diagnostic cache
   context. Sources can remain reuse-only while new diagnostic children are
   computed. The combined bundle caches scalar ladders and selected exports
   together, keyed by every source and policy; it is not a new cache backend.

The managed kinds `ccm_prefix_analysis` and `ccm_retained_reduction_check`
belong to `ccm-evidence` in both public and private catalogs. Public diagnostics
require authenticated public parents and explicit source-publication eligibility;
historical private children fail closed. Registration does not publish payloads.
`check_retained_reduction_via_cache` adds source-bound, budgeted reduction checks
without rebuilding the source. The new kind has producer/reader floor 0.15.0;
ordinary numerical artifact kind floors remain unchanged.

Prefixes retain their largest-parent identity. They are not labeled as
independently assembled canonical smaller matrices. Checkpoint eigenstates
name their own exact sources; their residuals are evaluated against the actual
parent prefix and can fail. A different-prefix source is not silently replaced
by a restricted largest eigenvector. Missing eigenstates and unresolved exports
are recorded as such, without invalidating the source or inventing a result.

## Precision and export

Authoritative scalars and raw innovations use lossless round-trip decimal
strings. Source, working, and accepted export precision are recorded separately.
Working precision cannot down-round the source. Additional working bits cannot
recover information missing from the original matrix.

`checked_decimal_export` checks the original identity first, then encodes and
decodes each declared candidate width at the same working precision before
checking it. The checkpoint packet checks normalization, vector preservation,
A v = sigma e_k, and the supplied eigenpair residual. Overlaps are computed
from the decoded vectors. Generic cancellation-sensitive generator identities
can use the same callback API. Passing a component/serialization screen is not
an independent matrix-assembly cross-check or interval certificate.

The current export gate uses scaled backward errors. It does not guarantee
relative forward accuracy of a deeply cancelled eigenvalue or identify a branch
inside an unresolved cluster. The explicit retained source identity, vector
preservation screen, and residual-matrix label must accompany every use.

## Capture levels

`ccm::capture::CcmCapturePlan` is the shared versioned recipe. Its `ultra`
constructor includes prefix diagnostics and an endpoint vector packet, while
claim/research/gap/maximum retain their existing defaults. Any level can request
explicit checkpoints using `with_prefix_checkpoints`.

This is intentionally a **two-phase interface**: `primary_options()` returns
backward-compatible existing capture options; `prefix_options(source_bits)`
returns the retained-source follow-up request. Applications must execute both
phases. Calling `primary_options()` alone does not capture prefixes. Applications
must integrate the retained phase and persist its outcomes explicitly; see
[capture levels and integration](CAPTURE_LEVELS.md). Persist the resolved plan
to replay a historical output set.

Ultra does not imply a certificate, aggregate prime arithmetic, changed
quadrature, missing-source recomputation, or public disclosure. Explicit root
and sector certification remain separately requested through their existing
APIs. Capture levels do not select different numerical algorithms. Changes to
solver semantics between releases are described in [numerical compatibility](NUMERICAL_COMPATIBILITY.md).

## Standalone retained-file example

    cargo run -p xc-spectral --release --features hp --example ccm_prefix_retained -- \
      /private/request.json /private/new-prefix-report.json

The request has `matrix: {manifest, payload}`, `eigenpairs: [{manifest,payload}]`,
`approved_payload_digests: ["hex SHA-256", ...]`, and `options` matching
`PrefixAnalysisOptions`. Relative source paths are resolved from the request's
directory. The output must not already exist. No network, publication, or
source discovery is performed by the example.

Canonical shard sources use the alternative `shard_manifest` form with explicit
local read limits. Optional `nesting_matrices` add source-bound exact block
comparisons to the report. See the [local shard guide](PREFIX_CONVERGENCE.md#read-a-canonical-local-shard-without-rebuilding-its-matrix).

## Qualification and provenance

The recurrences are derived from the block inverse identity. Implementation
and tests use AI assistance and the existing exact-rational and MPFR toolkit
facilities. No external source implementation or additional dependency is
incorporated. Fixtures use synthetic matrices and public configurations.

Tests compare against independently implemented exact rational Gauss-Jordan
inversion, include Hilbert and near-degenerate fixtures, malformed input,
unresolved pivots, serialization cancellation, retained-source tampering,
missing sources, private routing, and derived-only cache reuse. See the release
[validation guide](VALIDATION.md) for commands, outcomes and coverage limits.
Synthetic tests do not establish accuracy for an application's complete dataset.

The extended policy also computes `tr(A^-3)` through a retained normalized Gram
triangle, exposes a two-mode fit and trace closure, and can omit innovation
cancellation explicitly. New Ultra plans select full capture. See
[third-moment formulas, residual expansion and compatibility](PREFIX_CONVERGENCE.md).
The factorization, independent innovation solves and moment accumulation now
run in separate phases while preserving the legacy arithmetic and report bytes.
