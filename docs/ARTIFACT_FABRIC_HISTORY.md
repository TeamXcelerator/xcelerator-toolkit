# Artifact fabric history

Historical release details; see [current release notes](RELEASE_NOTES.md) for the active version.

## v0.15.0 foundation: reproducible research evidence

Artifact publication now skips unnecessary Git delta searches on compressed
archives and exposes batch timings and transfer progress. See the
[measured scope and recovery guidance](PUBLICATION_PERFORMANCE.md).

This release turns retained numerical results into source-bound data for finite
research tests: typed observables, frozen hypotheses, resolution budgets,
prefix and Schur diagnostics, checked vector exports, capture receipts, and
stable reduction checks. Managed receipts record every requested diagnostic
outcome, and evaluation packets retain selected observation bytes for score
replay.

Capture accepts certified and cross-checked sources, preserves actionable
failure reasons, and groups receipt attempts by their resolved plan. Prefix
working precision and export policies are explicit overrides. Scheduled
stabilization retains changing quadrature identities across an N ladder.

Prefix ladders retain model-qualified eigenvalue estimates, pivot and innovation
cancellation diagnostics, and optional exact nesting checks. New Ultra plans
also capture the third inverse moment, stronger moment endpoints and a two-mode
fit with a trace-closure residual. Cancellation capture can be omitted through
an explicit policy. A local shard manifest can be
loaded directly with explicit resource limits and verified source provenance.

The numerical hardening adds corrected sector reduction and selected-vector
solves under new identities, exact prolate cache inputs, stronger quadrature
and factor validation, and guarded HP root refinement. Distance profiles and
measurements now round-trip at their declared precision; child diagnostics use
the same retained parents on cold, backfill, refresh, and reuse paths. An
unresolved Q/2Q/4Q check is visible in the returned capture result.

Performance work includes fewer matrix/component allocations, reusable linear
storage tridiagonal solves, shared function evaluations, lazy 4Q refinement,
reusable MPFR inner-loop storage, parallel prefix factor columns and independent
innovation solves in separate phases,
parallel cross terms, long inner sums and reduction checks with fixed summation order, and cached retained-reduction
children.
Response capture prepares fixed root geometry once and processes independent
events and roots in parallel with bounded temporary storage. Fresh responses
reuse their production checks through an exact process-local payload seal;
cached responses retain numerical replay. Both paths report event progress.
See [response performance and validation](CCM_RESPONSE_PERFORMANCE.md).
See the release validation for measured evaluation counts, test coverage and
the limits of these performance claims.

Both public and private evidence catalogs register `ccm_prefix_analysis` and
`ccm_retained_reduction_check`. Public retained diagnostics require authenticated
public parents; target-derived evidence keeps its private publication policy.

- [Release notes and migration](RELEASE_NOTES.md)
- [Numerical compatibility and existing artifacts](NUMERICAL_COMPATIBILITY.md)
- [Complete positive movable-root discovery](CCM_COMPLETE_ROOT_DISCOVERY.md)
- [Release validation](VALIDATION.md)
- [Prefix formulas, precision, and capture](CCM_PREFIX_ANALYSIS.md)
- [Prefix convergence models and local shard inputs](PREFIX_CONVERGENCE.md)
- [Frozen research evidence workflow](RESEARCH_EVIDENCE.md)
- [Capture levels and application integration](CAPTURE_LEVELS.md)

New Ultra plans retain [state geometry](STATE_GEOMETRY.md): raw and unit-norm
normalizers, spatial moments, energy tails, sampled sign measurements and grid
refinement evidence. An explicit retained-file entry point creates the same
managed child without repeating the primary calculation. Historical plans and
artifacts keep their original identities.

Ultra requests a finite set of measurements. The capture runner executes a
primary diagnostic adapter and retained-source follow-up, then saves a managed
receipt including missing, blocked and failed work. Budgeted reduction checks
can join that receipt; full quadrature verification remains explicit. Capture
completion and numerical acceptance are reported separately. Applications adopt
the runner through the documented API; existing application flags do not change
merely because the library was updated.

`ccm::hp::capture_run::RetainedCcmRun` supplies the primary adapter for new
applications: compute the claim once, save its primary result, and attempt
supplemental diagnostics independently with authenticated source manifests.
Eigenfunction profiles remain capturable when the runtime target is absent;
target-dependent children report their unavailable input separately.

## Existing artifact fabric (0.14.3)

Version 0.14.3 adds backward-compatible multi-shard cache rollover, building on
the CCM capture and managed-publication functionality delivered in 0.14.1.
Publishing children of shard-reused artifacts to a new
destination now stages the full dependency closure without recomputation,
including identity-first dependencies later referenced by their real cache
keys. Author publication also accounts for validated workstation and remote
reuse hits, including strict `require_reuse` runs; destination verification
suppresses redundant commits, while an execution that observed no artifacts
fails instead of reporting a vacuous success. Historical exact dependencies
may be added to immutable repository closure without displacing an
equal-or-newer live semantic-index entry; ordinary producer downgrades remain
forbidden. Publication preflight recognizes those superseded exact identities
only when the retained manifest has a canonical batch proof, matching the
reader's historical-resolution rule. Long destination scans refresh GitHub write evidence before both
candidate authorization and remote mutation, so the five-minute authority
window cannot expire merely while a large existing family is inspected.
Cold multipart reuse batches missing Git blobs before concurrent verified
streaming, verifies reused parts during the reconstruction copy, quarantines
and re-downloads a corrupt reused part once with best-effort quarantine
cleanup, removes a corrupt complete package so it can be rebuilt, and reports fetch,
reconstruction, and decode time separately. Workstation ZIP hits read a
compressed object up to the split part size once, within a bounded
process-wide in-memory allowance, and avoid a redundant second hash. Exact published
identities are looked up through a persistent per-semantic-digest inventory
rather than a directory walk. Exact dependency sets are prepared
progressively in bounded repository batches, with independent shard sessions
allowed to proceed concurrently; complete local packages and retained parts
are excluded from remote preparation, and each filtered blob reserves the
caller's exact retained-part bound rather than a generic 100 MB allowance.
Publication staging reuses verified
encoded packages, stages dependencies from those packages without inflating
their payloads, and directly links their verified split parts when available,
instead of recompressing, recopying, or repeatedly decoding artifacts that are
already present. Direct encoded adoption requires the exact persisted encoder
profile; unprofiled legacy objects remain readable but are re-encoded before
publication. The unchanged single-entry route retains its published V1
transport identity; V2 is reserved for packages that actually contain
multiple items on the corrected ZIP64 route.
Dependency closure resolves each member by exact content
digest, so a newer artifact under the same semantic key no longer blocks
publication of children that name the older one.

Cache transport tuning is explicit and bounded. `XC_CACHE_PREFETCH_CONCURRENCY`
controls independent repository preparation and `XC_CACHE_DOWNLOAD_CONCURRENCY`
controls verified part-download workers (both default to 4 and clamp to 1--8).
`XC_CACHE_SINGLE_PASS_ZIP_BYTES` sets the per-object in-memory ZIP threshold
(default 90 MiB), while `XC_CACHE_IN_MEMORY_ZIP_BYTES` sets the process-wide
allowance shared by all such reads (default 256 MiB). Memory overrides must be
between 1 byte and 16 GiB; invalid or zero values retain the defaults.
Warm distance/profile hits avoid fresh eigensolves, quadrature, and sampling
work; distance capture reuses the exact managed `ccm_weil_eigenpair`, binds
its content digest and dependency closure into every affected artifact
identity, and reuses exact managed Gauss--Legendre artifacts across
configurations and exact eigenfunction values
across nested refinement grids; root refinements share their secular-pole
vector; and invalid capture resolutions or semantically mismatched retained
payloads fail early. Maximum capture adds separate numerical-analysis
artifacts. Target-dependent work reads its definition from the private runtime
path named by `XC_TARGET_SPEC_FILE`; only the specification digest enters cache
identity, and those derived artifacts are private-only. Explicit finite sector certification can additionally retain exact
cutoff-free parity, ordering, simplicity, and positivity evidence. The
corrected distance/profile semantic identities supersede legacy artifacts that
were not bound to their canonical eigenpair; unrelated artifact payloads,
schemas, and numerical definitions are unchanged.

---
