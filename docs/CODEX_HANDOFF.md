# Desktop / VS Code Codex handoff — 0.14.4

## Start here

Owner Ronnie Andrews, Jr. explicitly requested integration of the Toolkit work
into `main` in the September 6, 2026 CCM research conversation. Commits follow
`v0.14.4: ...`. This work used AI-generated implementation assistance. No
independent human review or private release audit is asserted. The repository
owner controls release approval; no release tag was created by this integration.

Read `CCM_HARDENING.md` and `CCM_PREFIX_ANALYSIS.md` before changing behavior.
The original baseline is `e4e8ec5912c12ae12836187813b6962e5cd71aa6` (0.14.3).
The earlier 0.14.4 source commit is
`9a28cf3b4db2ae4a72915bfe39e61591481a6907`. The work-branch checkpoint before
this integration was `afe5d30e6e73e535d685899fedf584dd5209b9b1`.
Use the current verified `main` commit for the handoff; do not use a stale
source-preparation workflow or an earlier failed Actions run as its status.

## Implemented in the Toolkit

- Corrected finite endpoint in cutoff-free zero mode; certificate schema and
  mathematical semantics distinguish it from defective historical outputs.
  Analytic tail budgets scale with requested precision and dimension.
- Default prime/LU scratch-allocation reuse and streaming even-sector
  dependency validation. The original production arithmetic and full-Tau
  eigenstate replay are retained.
- Separate exact-rational research cutoff, aggregate-prime and bucketed-order
  routes, one-border Gram/Schur, source-to-root transfer, conditional transform
  strip estimates, and process high-water-memory diagnostics.
- `xc_numerics::prefix`: fixed-order unpivoted LDLT ladder, innovations,
  tr(A^-1), tr(A^-2), concentration and eigenvalue-depth estimates; checked
  decimal export callback. O(D^3) arithmetic / O(D^2) packed working storage.
- `ccm::prefix`: approved retained-byte authentication, even-basis handling,
  normalized export copies, post-decode identity checks, exact source-bound
  derived cache. Missing checkpoint eigenstates stay missing.
- `ccm::capture::CcmCapturePlan`: shared versioned capture recipe, with new
  prefix diagnostics in `ultra` and explicit lower-level requests.
- `ccm_prefix_retained` example for a complete local retained-file operation;
  private-only kind registration in Toolkit routing; read-only impact inventory.

## Desktop follow-through (not completed by this Toolkit merge)

1. Wire the paper applications to BOTH phases of `CcmCapturePlan`.
   `primary_options()` feeds their existing solve/capture operation; then
   `capture_retained_diagnostics()` accepts the exact retained matrix/eigenpair
   sources and a separate diagnostic cache context. Calling only
   `primary_options()` DOES NOT capture prefixes. Paper 3 currently lacks an
   Ultra enum in its original application. Preserve Paper 1's applicable
   natural-parity behavior and make exclusions explicit rather than projecting
   a natural eigenstate silently into an even one.
2. Finish and qualify the Paper 1 wrapper flags / structured Claim 8 comparison
   and the Paper 3 scientific-verdict, root-eligibility, and resume-provenance
   work. Do not assume old work-branch preparation scripts constitute integrated
   application code. Inspect their real source diffs. Historical dependency tags
   and published paper measurements must remain reproducible.
3. Register `ccm_prefix_analysis` in the PRIVATE evidence family only when the
   owner authorizes remote publication. No private registry or shard was edited
   here. Resolve canonical warehouse manifests through the existing adapters;
   the public prefix API takes the runtime manifest and logical payload bytes.
4. Acquire the research continuation packet: two existing ladders and nine
   checkpoints. Their reference implementation/data were not supplied in this
   chat. Compare every declared scalar and selected vector to frozen tolerances,
   including after serialization. Do not substitute the synthetic fixture tests
   for that acceptance gate.
5. Run the owner's private release audit and representative large-N/high-bit
   workloads on the intended Windows/WSL machine. Recheck determinism and strict
   reuse, benchmark cold and warm paths separately, and review mathematical
   semantics, privacy, and licensing before tagging a release. Native Linux
   tests here do not qualify unsupported WSL parallel GMP root construction.

## Artifact preservation rules

Do not raise ordinary producer floors because a diagnostic was added. The only
existing-kind floor changed here is the known-defective
`ccm_sector_gap_certificate` (minimum producer/reader 0.14.4). Existing valid
quadrature, components, Tau/sector matrices, factorization, eigenstate, root,
profile, and distance artifacts retain their prior floors. Historical defective
certificates can remain stored, but cannot establish corrected-formula claims.

New diagnostics are CHILDREN of exact retained sources, not replacements.
Alternative arithmetic has a distinct identity; `ultra` never enables it.
The combined prefix bundle is keyed by parent, checkpoint eigenstates, basis,
algorithm, and precision/export policy. It is not yet a separately persisted
factorization artifact. Reusing the same bundle avoids repeated factorization;
requesting a different checkpoint/export policy can require new analysis.

A prefix of the largest matrix is not automatically an independently assembled
canonical smaller matrix. Dimension k=N+1 is not N. All current ladder moment
inequalities use an orthonormal Euclidean basis. Generalized A-zG one-border
analysis is separate; no generalized trace-moment implementation is claimed.

Do not mark an unresolved diagnostic as successful, take absolute values of
bad pivots, regularize, or reconstruct missing sources. Extra working bits do
not recover information absent from the retained matrix. Export checks are
backward-error screens and do not prove forward eigenvalue/root error in an
ill-conditioned problem. Interval certification of new prefix/overlap/kernel
claims remains a later, explicitly requested proof-specific task.

## Local qualification commands

Use an isolated cache root, no remote source, and no publication. Install the
existing optional FLINT shared-library prerequisites for `arb`; no new package
requirements were introduced. Run:

```sh
cargo fmt --all -- --check
cargo test --workspace --locked --no-fail-fast
cargo test --workspace --features hp,arb --locked --no-fail-fast
cargo clippy --workspace --all-targets --features hp,arb --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features hp,arb --locked
cargo test --manifest-path tests/external-consumer/Cargo.toml --features hp --locked
python3 -m unittest discover -s tools -p 'test_*.py'
cargo run -p xc-spectral --features hp --example ccm_prefix_retained -- \
  /private/request.json /private/new-prefix-report.json
```

The example never overwrites an existing output. The source manifest/payload
allowlist must come from the intended campaign; a checksum authenticates bytes
against that approved record, not the mathematics or an external signature.
