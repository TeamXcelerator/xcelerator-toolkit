# CCM capture levels and application integration

`ccm::capture::CcmCapturePlan` defines a versioned set of requested
measurements. Applications execute the plan and retain its outcomes. Selecting
a level does not execute a campaign, change a numerical algorithm, request a
new sector-gap certificate, or publish data. Finite transform enclosure diagnostics
use Arb when available and can consume an already supplied sector certificate.

For a new calculation, `ccm::hp::capture_run::RetainedCcmRun` executes a seeded
or independently acquired primary claim once through an explicit managed cache
context. Save `primary()` before supplemental work, then call
`capture_diagnostic_outcome` for each requested primary ID. Each call returns measurement
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

The table describes the shared v0.16.0 plan (capture-plan v7), not a similarly
named option in an application that implements its own capture policy.

| Level | Additional measurements requested by the shared plan |
|---|---|
| Claim / Research | No supplemental groups beyond the application's primary computation |
| Gap | Evenness and selected sector analysis with two eigenpairs |
| Maximum | Evenness, full sector eigenvalue spectra with a bounded number of selected vectors, root conditioning, profile, target distance, resolution and target residual analysis |
| Ultra | Maximum plus deviation decomposition, prime-power response, u-flow response, retained prefix analysis, state geometry, indexed transforms, total operator energy, root-window summaries, and the twenty extended diagnostic groups, reference projection, numerical coverage summaries, assembly error with exact-form eigenvalue enclosures, and checkpoint low-spectrum enclosures |

See [extended diagnostics](EXTENDED_RESEARCH.md) for the additional groups and
external input file. Old v1/v2/v3/v4/v5 plans retain their request sets.

New Ultra plans use capture-plan v6 and request [retained research diagnostics](RETAINED_RESEARCH.md)
through the actual primary eigenpair. Old serialized plans keep their original
request set and receipt identity. Ultra requests reference projection automatically; other levels may select it with
`with_reference_projection()`. An external reference is still required. A missing
source is never a completed projection. Local cohort discovery adds independent
configuration comparisons; the frozen stabilization-rule operation remains separate.
See [Ultra completeness](ULTRA_COMPLETENESS.md) for preparation, budgets,
checkpoint recovery, finite enclosures and numerical coverage.

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

Extended policies use prefix semantics v10; the two-moment policy uses v8.
Both apply directed, scaled export checks and replay all numerical fields on
cache reuse. Compatible retained matrices and eigenstates can be reused while
computing the new child; `RequireReuse` fails until that requested child exists.
Historical v1/v2/v3 observations keep their original labels and assurance. Exact nesting comparisons require explicit smaller
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
`CcmTargetDistanceHp::resolution_tolerance_met` describes the distances
actually returned at Q: `Some(true)` requires every applicable Q/2Q comparison
to pass, and `Some(false)` means at least one fails. The separate
`resolution_ladder_tolerance_met` describes the last attempted adjacent pair,
which may be 2Q/4Q. A passing later pair does not qualify the returned Q value.
Both are `None` when no applicable refinement was requested. The capture
diagnostic labels these scopes separately and retains the original evidence
under `retained_evidence`. These are empirical agreement checks, not rigorous
error bounds. Earlier reports with a single passing flag require inspection
of their retained Q/2Q comparison before interpreting the Q-resolution value.

Upgrading a Cargo dependency does not implement these application steps. A
preexisting `--ultra` flag must be checked against this contract. Rebuild from
the updated lockfile and verify an example run's receipt and artifact identities
before scheduling a larger campaign.

### Exact-form error bars and checkpoint spectra (v7)

`assembly_error` encloses the exact finite CCM Weil form at the declared
cutoff, integer or fractional, with the cutoff-free FLINT/Arb closed form and
compares it entrywise with the stored Tau matrix. It reports rigorous norms of
the difference for the full matrix, the even and odd sectors (including the
parity transform's own rounding) and the pole, archimedean and prime
components, a per-mode error profile and the surviving digits. By Weyl's
inequality every retained stored eigenvalue enclosure widens by the sector
bound; the measurement lists these exact-form enclosures for the selected
eigenvalue, each retained sector eigenvalue, the even gap and the lowest
odd-minus-even separation, with the budget (stored or assembly) that limits each.

When the retained state and a second even eigenvalue enclosure are available,
`assembly_error` also encloses the exact finite-form roots. The state residual
and the assembly bound give a Davis-Kahan angle bound to the exact ground state;
the resulting weight intervals and exact poles define an interval secular
function, and interval Newton from each computed root encloses the root of every
secular function in that box, including the exact form's. Each root reports its
exact-form enclosure and resolved digits, or the reason it is unresolved.

`checkpoint_spectra` encloses the three lowest eigenvalues, the gap and the gap
ratio of leading even-sector blocks along a halving ladder of dimensions from
the source down to 8. The even-sector matrix nests exactly in N, so each row is
the corresponding smaller configuration. These are directed enclosures for the
stored matrix; widen eigenvalues by `assembly_error.even_sector.spectral_upper`
(gaps by twice it) for exact-form enclosures.

### Root certification inside Ultra

Requesting root certification adds one `root_certificate` measurement and never
removes data. Every requested ordinal receives a row:

| Row outcome | Meaning |
|---|---|
| `certified_finite_enclosure` | Interval enclosure of the root of the exact stored secular source, with the computed value's agreement (`inside`, `outside`, `no_computed_value`) |
| `computed_not_certified` | Computed value and solver status retained; the row states why it was not certified |
| `not_computed_not_certified` | Neither a computed value nor a certificate |

The whole range is certified first; on failure it is bisected so every
certifiable ordinal is still certified. Ordinals beyond the exact positive root
count of the pole range are reported as such. A stagnated computed root that
falls outside its certified enclosure is flagged, and the ordinal whose
enclosure it lies in is named. Integer and fractional cutoffs are supported.
Numerical coverage counts certified rows as resolved and computed-but-not-certified
rows as qualified. The embedded certificates replay independently.

Counts come from the exact secular numerator and FLINT/Arb isolation when its
rational workspace fits the budget. Larger sources use a directed pole-gap
census instead: between adjacent poles, G(z) = (z - p_g)(p_{g+1} - z) R(z) is
smooth on the closed gap, has the same interior roots and takes the residue
signs at the poles. Enclosures of G or G' that exclude zero prove no root or a
monotone piece with exactly one simple root, and other pieces are halved. The
enclosures are order-8 Taylor models around each piece's midpoint: point
coefficients keep the cancellation among the residue terms that natural
interval sums lose, and only the remainder, scaled by the radius to the eighth
power, uses natural bounds. A gap that cannot be
halved further at the working precision is retried with 64, 256, 1024 and 4096
guard bits, and is otherwise reported unresolved with its reason. A piece is
split where G has a proven sign (the midpoint or the first of a fixed sequence
of nearby fractions), so a root exactly at a midpoint does not stall it. The
census counts gaps only as far as the requested ordinals (a parallel batch may
evaluate a few more, never more than twice the roots still needed), so the
total positive count is recorded only when every gap was counted. Interval
Newton certifies each root in either route, and each certificate records and
replays its own route.

Sector-gap certification is additive in the same way. When it cannot complete,
for example because the guide spectra are precision-limited at the requested
configuration, the capture keeps the computed sector gap and every other
measurement and records the reason (`sector_gap_certification_limitation`).

## Additional explicit work

| Facility | How it is requested in v0.15.1 |
|---|---|
| Retained reduction similarity/orthogonality report | Supply `RetainedReductionRequest` to `execute_with_receipt_and_reduction` to execute and record it; the standalone `check_retained_reduction_via_cache` API also remains available |
| Full Gauss--Legendre rule verification | Fresh rules pass the full O(n^2) root/weight check. Ordinary cache reads perform O(n) admission (shape, ordering, symmetry, low moments and three root/weight probes); `check_gauss_legendre_rule_hp` and `verify_gl_cache_dir` perform the full check with an explicit order budget |
| Root and sector certificates | Request the corresponding certification options explicitly; root certification records per-root outcomes as described above |
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
