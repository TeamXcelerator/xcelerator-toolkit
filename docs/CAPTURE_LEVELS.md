# CCM capture levels and application integration

`ccm::capture::CcmCapturePlan` defines a versioned set of requested
measurements. Applications execute the plan and retain its outcomes. Selecting
a level does not execute a campaign, change a numerical algorithm, request a
certificate, or publish data.

For a new calculation, `ccm::hp::capture_run::RetainedCcmRun` executes a seeded
or independently acquired primary claim once through an explicit managed cache
context. Save `primary()` before supplemental work, then call
`capture_diagnostic` for each requested primary ID. Each call returns measurement
bytes with authenticated source manifests; an error leaves the primary state
available for other diagnostics. Pass `retained_even_sources` to the shared
runner for prefix, checkpoint, and budgeted reduction capture. Finalize the
managed session after retaining the receipt.

The adapter preserves primary parity and root acquisition. Natural/adaptive
states cannot supply even-state response diagnostics or checkpoint exports.
`distance::hp::capture_ccm_profile_via_cache` retains an eigenfunction profile
without a runtime target specification, using the same identity and exact bytes
as the established distance route. Target-dependent children still need the
supplied target. Publication follows the caller's session, including verified
encoded transport reuse. Newly computed diagnostic payloads are delivered to
capture observers even when staging supports encoded-only production. This
avoids an empty receipt observation after a successful ZIP cache write; it does
not require decoding unrelated dependency payloads.

## Default coverage

The table describes the shared v0.15.0 plan, not a similarly named option in an
application that implements its own capture policy.

| Level | Additional measurements requested by the shared plan |
|---|---|
| Claim / Research | No supplemental groups beyond the application's primary computation |
| Gap | Evenness and selected sector analysis with two eigenpairs |
| Maximum | Evenness, full sector eigenvalue spectra with a bounded number of selected vectors, root conditioning, profile, target distance, resolution and target residual analysis |
| Ultra | Maximum plus deviation decomposition, prime-power response, u-flow response and retained prefix analysis |

Ultra's default prefix vector checkpoint is the full source even-sector
dimension. The scalar prefix ladder covers accepted pivots up to that
dimension. Use `with_prefix_checkpoints` to request other checkpoints; supply
their exact retained eigenstates when overlaps are needed. Ultra does not mean
all eigenvectors, all root windows or an unbounded parameter grid.

New Ultra plans request the third inverse moment, cube-moment endpoints, a
T2/T3 two-mode fit with T1 closure, and the established prefix diagnostics.
`with_prefix_diagnostics(PrefixDiagnosticPolicy::full())` requests these at
other levels. An explicit policy can omit innovation cancellation or retain
the historical two-moment capture. The policy enters the plan and child identity.
Old serialized plans keep their old policy; they are not silently expanded.

Extended policies use prefix semantics v3. Compatible retained matrices and
eigenstates can be reused while computing the new child; `RequireReuse` fails
until that requested child exists. The legacy two-moment policy retains v2
identity and byte reuse. Exact nesting comparisons require explicit smaller
sources through the retained-prefix example. See [prefix convergence](PREFIX_CONVERGENCE.md).

Use `with_prefix_working_precision(512)` to analyze retained 256-bit source
entries with additional arithmetic precision. `with_prefix_export_policy`
accepts a `PrefixExportPolicy` specifying pivot margin, decimal-width schedule
and export tolerance. These controls are persisted in the resolved plan;
working precision below the source precision is rejected. Additional working
bits reduce diagnostic rounding but do not recover missing construction digits.
Numerically equivalent tolerance spellings share a canonical decimal identity.

Each prefix builder enables retained prefix capture, including at Claim and
Research levels. Checkpoints, precision and export policy can be supplied in
any order. A precision or policy override alone requests the scalar ladder;
it does not add vector checkpoints at those lower levels. `validate()` rejects
a pivot margin at or above an explicitly selected working precision. When
working precision is inherited, call `validate_for_source_precision(bits)`
to bind that check before execution; `prefix_options(bits)` also performs it.

Target-dependent groups require the runtime target definition and suitable
source data. Unavailable inputs remain missing outcomes. The supported finite
set, precision and source identities must accompany the resulting data.

## Execute and retain the outcomes

`CcmCapturePlan::execute_with_receipt` runs the requested primary diagnostic
adapter, executes retained prefix analysis once, and persists a managed
`research_capture_receipt`. The adapter receives each primary diagnostic ID
and returns a `CapturedDiagnostic` or a typed `CaptureFailure`. Existing
numerical APIs remain responsible for their own source acquisition and reuse.

1. Resolve the plan and supply the exact retained even matrix and eigenstates.
   Keep the dependency commit, lockfile, binary identity, features and exact
   inputs with the application run.
2. Provide the primary diagnostic adapter. `CapturedDiagnostic::from_cached`
   binds a numerical cache result to its produced or reused manifest. Use
   `CapturedDiagnostic::new` for other serializable measurements and their
   authenticated source manifests.
3. Call `execute_with_receipt`, or `execute_with_receipt_and_reduction` with a
   `RetainedReductionRequest` to include dimension, precision and tolerance
   budgets for the cubic reduction check.
4. Inspect the returned receipt and measurements. Every requested group has a
   terminal outcome. Missing sources, blocked budgets and returned errors are
   recorded without preventing independent diagnostics. A checkpoint missing
   its eigenstate is recorded as missing; its available innovation data remains
   in the prefix ladder. Panics and process termination are not converted into
   successful receipts.

The receipt embeds the resolved plan, measurement values, their digests and
exact dependency references. Changed outcomes produce a new record identity,
so an incomplete attempt cannot shadow a later backfill. Individual numerical
children use their normal cache policies. Full receipts are private-only;
public-only record requests fail before the diagnostic adapter runs.

Validated, cross-checked, certified and published source grades are accepted;
staged, quarantined and deprecated sources are rejected. The receipt retains a
validated dependency floor and does not inherit a source's certification.
Private receipts preserve actionable returned errors after secret screening.
Only empty or secret-bearing error text is replaced by a generic explanation.

Receipt logical keys group attempts as
`research/research_capture_receipt/{plan_digest}/attempt/{record_digest}`.
`capture_receipt_plan_prefix` returns the grouping prefix; manifests also carry
the `research_plan_digest` tag. Pass the prefix to `CacheResolver::matching_keys`
with an explicit result limit to enumerate locally indexed attempts. Compare requested/completed outcomes when
selecting an attempt. Numerical validity still requires inspecting each report.

For a non-CCM workflow, `xc_cache::collect_capture` provides the same automatic
outcome accounting, and `capture_and_persist` adds managed persistence.
`persist_capture` stores an already collected record. Warm record reads check
coverage, embedded evidence and exact dependencies. The low-level `receipt`,
`primary_options` and `capture_retained_diagnostics` interfaces remain available
for applications that need to control each phase themselves.

`CaptureReceipt::is_complete()` reports completion of requested acquisition
outcomes. A completed diagnostic report can contain failed exports or an
unresolved numerical result. Inspect those fields separately. In particular,
`resolution_tolerance_met` is `Some(false)` when the uniform-grid ladder is
exhausted, `Some(true)` when applicable refinements pass, and `None` when no
applicable refinement was requested. These are empirical checks, not rigorous
error bounds.

Upgrading a Cargo dependency does not implement these application steps. A
preexisting `--ultra` flag must be checked against this contract. Rebuild from
the updated lockfile and verify an example run's receipt and artifact identities
before scheduling a larger campaign.

## Additional explicit work

| Facility | How it is requested in v0.15.0 |
|---|---|
| Retained reduction similarity/orthogonality report | Supply `RetainedReductionRequest` to `execute_with_receipt_and_reduction` to execute and record it; the standalone `check_retained_reduction_via_cache` API also remains available |
| Full Gauss--Legendre rule verification | Call `check_gauss_legendre_rule_hp` with an order budget; ordinary reads apply cheaper screens |
| Root and sector certificates | Request the corresponding certification APIs explicitly |
| Additional prefix checkpoints and overlap eigenstates | Set checkpoints and supply their retained sources explicitly |
| Frozen hypothesis scoring and replication packets | `evaluate_hypothesis_packet` retains selected bytes; `persist_hypothesis_evaluation` replay-checks and caches the packet with exact source dependencies |
| Performance records | Enable `XC_PERF_REPORT` and retain the report with run metadata |

Managed publication kinds exist for prefix analysis, retained reduction, capture
receipts and hypothesis evaluations. Full receipts and evaluation packets are
private-only because they may contain runtime research definitions and outcomes.
Automatic model fitting, identifiability analysis, coupled mechanism reports,
experiment ranking and campaign resume/scheduling are not supplied by the
capture plan.

For controlled research comparisons, retain signed differences across finite
dimension, construction quadrature and arithmetic precision separately.
Preserve original-operator residuals, branch/index/gap eligibility, root
conditioning and export acceptance alongside the observable. Explicitly
identify missing cases so a successful subset cannot masquerade as a complete
cohort. See [frozen research evidence](RESEARCH_EVIDENCE.md).

## Existing sources and publication

Backfill requests may reuse valid parents and compute missing children under
their exact source-bound identities. A required-reuse diagnostic context fails
if the child is absent. Source acquisition and diagnostic computation can use
separate contexts and budgets; the retained phase never substitutes a newly
computed source for a missing retained one. See [numerical compatibility](NUMERICAL_COMPATIBILITY.md)
before comparing results produced by different solver or export semantics.

`ccm_prefix_analysis` and `ccm_retained_reduction_check` use the existing
`ccm-evidence` family in both visibility lanes. The reduction payload shape is
described by [the shared schema](schemas/ccm-retained-reduction-check-v1.schema.json).
The shared receipt and evaluation shape schemas are also supplied in both
registries; their kinds are admitted only by the private evidence catalog.
No new shard or registry protocol version is needed. Source-only diagnostics
require authenticated public parents for public publication. Target distance,
resolution, residual and decomposition kinds remain private-only.

Check registration against the selected registry and active shard:

```sh
cargo run -p xc-cache --example audit_kind_registration -- \
  /local/registry/families/ccm-evidence.json \
  /local/evidence-shard/cache-repository.json
```

Registration permits routing a type; it does not prove that any requested
payload exists. Local capture and remote publication are separate operations.


## Compiled outcome-accounting example

```sh
XC_CACHE_REMOTE=none XC_PUBLISH_TARGET=none \
  cargo run -p xc-cache --example research_capture --locked
```

This synthetic example stores and prints a receipt containing one measured
value and one missing input. It demonstrates managed persistence and outcome
accounting; its values are not research observations. Select a scratch
`XC_CACHE_ROOT` to keep qualification data separate from other local work.
