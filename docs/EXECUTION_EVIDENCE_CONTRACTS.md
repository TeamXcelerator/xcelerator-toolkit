# Execution and evidence contracts

These contracts describe numerical evidence, artifact compatibility, and resource
ownership. Executable validators enforce admission. Finite-matrix evidence does
not by itself establish a continuum error bound.

## Stored-state resolution and compatibility

The default `CcmGroundIndexPolicy::Require` policy retains the released
`stored_state_residual_and_one_norm_scale_gap_v1` resolution record and Weil
eigenpair semantic identity. Its ground-index proof must succeed. Existing
compatible eigenpairs and their dependent artifacts remain eligible for reuse.

The opt-in `RecordOutcome` policy uses
`stored_state_residual_and_admitted_gap_v2` in its identity and resolution
record, whether its ground-index proof succeeds or fails. Residual inclusion
does not require index admission. Separation and angle conclusions do require
it: an unproved index leaves those fields unavailable with a reason. A record
from the earlier policy semantics cannot supply these conclusions by reuse.
Matrix, quadrature, and sector-gap-certificate identities are unaffected.

Native operator cross-check records use v3 semantics to include action-error
agreement. Managed artifact producer and reader floors remain 0.16.0. Semantic
identity changes are specific to the affected calculations, not a general
cache epoch change.

## Public API migration from 0.16

Version 0.17 changes the following public Rust interfaces:

- `operator_error` is added to `EigenpairReportF64`, `BlockEigenReportF64`,
  `MatrixFreeGeneralizedEigenpairReportF64`,
  `MatrixFreeGeneralizedEigenpairReportHp`, `BlockGeneralizedEigenReportHp`,
  `BlockShiftInvertReportHp`, `ShiftInvertKrylovReportHp`, and
  `ThickRestartLanczosReportHp`. Callers constructing these reports must provide
  truthful action evidence; unknown error is not exact error. Older serialized
  native reports without this evidence deserialize as unknown.
  `EigenpairReportHp` retains its existing field shape and records evidence
  through its diagnostics and provenance.
- `ApproximationKind::OperatorApplication` is a new variant. Exhaustive matches
  must handle it, preserving the distinction between declared operator error
  and ordinary scalar roundoff.
- `CcmStoredStateResolution::residual_angle_upper` is `Option<String>`.
  Consumers must handle an unavailable angle and inspect the accompanying
  resolution reason. A missing bound must not be interpreted as zero.
- The legacy owned `resolve_or_compute_json_artifact_memoized` API validates
  results but no longer retains an implicit process-wide decoded copy. Consumers
  needing reuse should use the shared resolver within an explicit
  `ValidatedValueMemo` scope and supply a runtime-allocation size estimator.

`SharedValidated<T>` clones share one allocation and reservation. Converting to
an owned value can copy when another handle still owns the allocation. Memo keys
bind payload digest, artifact key, payload and runtime types, and validation
context. A stronger validation context still requires admission under that
context; sharing does not bypass validation.

## Run resources and retention

Supplemental retained-source recovery can evaluate indexed transforms and
prime-power or u-flow responses without reacquiring the primary matrix,
eigenpair, or secular roots. The original portable primary result supplies
the response root selection and must match authenticated retained payloads,
including failed-root details. The original quadrature base and parity policy
are required for response replay; missing or inconsistent source declarations
are refused. Secondary spectral preparation and tangent solves may still run.
Recovery uses the live producers and preserves their artifact identities and
numerical qualifications. Completing acquisition does not certify every bound.
U-flow source admission precedes quadrature setup. Paired MPFR sine/cosine
evaluation preserves directed interval endpoints, accumulation order, and
successful artifact identities. Separate phase timings identify table setup,
integrals, pole/archimedean actions, and prime-event work.

Signed and indexed finite transforms first use the legacy 4096-bit guard
policy and its existing artifact identity. If that arithmetic exhausts its
precision ladder, an extended attempt uses a distinct identity containing its
guard limit. The default extended limit is the next power of two at or above
the requested precision plus 64 bits, with a minimum of 4096. Set
`XC_TRANSFORM_MAX_GUARD_BITS` to a power of two from 4096 through 1048576 to
override that limit. Invalid values are refused. The directed enclosure and
relative-width requirements are unchanged. A larger limit permits another
attempt; it does not guarantee resolution or improve source-state accuracy.
Automatic retained arithmetic input preparation uses the same transform
policy, with resulting input content bound by its digest.

Exact response point vectors are admitted separately from exact secular
polynomial construction. Their estimate includes dyadic exponent spans,
distance denominators, and concurrent rational row temporaries. A run-owned
reservation stays alive while those vectors and row workspaces are in use.
This admission does not raise the separate exact-polynomial limits.

`RunResources` accounts for retained matrices, validated memo values, and covered
diagnostic workspaces. Estimates describe runtime allocation, not serialized
bytes. They do not include every temporary,
allocator overhead, native-library cache, or process allocation, and are not an
RSS guarantee. Use an external process supervisor for a hard memory ceiling.

A retained run limits its memo to one quarter of its capture memory budget.
Caller-held shared values remain charged after memo eviction. Dropping the final
handle releases the reservation. Run scopes must be entered on the thread
performing acquisition; a scope does not implicitly follow work onto another
thread. Independent runs do not share hidden retention.

The planned and legacy GL mapping adapters, integral batch callbacks, and the
SafeCapped owner callback explicitly propagate that context and restore the
worker's previous context afterward. Arbitrary Rayon or native callbacks still
require explicit propagation when making their own reservations. Up-front
parent reservations also cover numerical kernels that allocate no independently
reserved workspace. Unscoped standalone calls do not acquire a hidden global
budget.

An explicitly supplied run budget is authoritative; read-only cache wrappers
and retained-source replay do not silently reduce it to 8 GiB. Exact payload
readers verify size and digest through temporary files. Retained Tau decoding
charges parsed Float storage and bounded parser scratch, with no fixed 256 MiB
payload or 8 GiB matrix ceiling. Reservations follow shared owned allocations.
Storage chunk size is independent of memory admission and decoded scratch-disk
requirements. Explicit caller source-size limits are still enforced.
For read-only seekable reads, the cache policy admits the 2 MiB explicit buffer
floor separately from `maximum_temporary_disk_bytes`. That disk ceiling applies
to each decoded spool; concurrent readers consume the sum of their spool sizes.
ZIP metadata and inflater overhead are additional to the buffer floor. This
transport path does not enable an out-of-core numerical algorithm.

Automatic finite matrix diagnostics admit dimensions above 512 through checked
shape, operation, and memory estimates. Only those formerly blocked dimensions
use a new admission identity; smaller requests retain their options and hashes.
The supported precision and exact-rational domains remain enforced. Convergence
request hashing and size checks stream their serialization. Insufficient working
memory remains an unretained outcome, so a larger-budget run can retry.

Automatic directional, tail-block, and cluster checks retain a packed triangle
of interval endpoints instead of a dense matrix of endpoint strings. Canonical
row-major text is generated for size checks, parsing, and request hashing;
each pass must agree on the exact endpoint spellings. Shifted matrices borrow
unchanged entries and own only their diagonals. Interval inertia consumes its
converted matrix as the elimination workspace. Request bytes, operation counts,
rounding order, and mathematical thresholds are unchanged.

Their workspace reservations follow the diagnostic and the guard actually
attempted. Completed attempts release their workspace before a larger guard
is considered. A later memory refusal remains operational and unretained; it
does not make an earlier unresolved attempt reusable. Retained parent objects
remain charged throughout.

`XC_RESEARCH_PREP_WORKERS` controls simultaneous tasks in retained arithmetic
preparation. It defaults to 16, accepts a positive integer, and is capped by the
Rayon pool and run worker allowance. It changes scheduling only. It does not
create another pool, cap unrelated GPU products, or bound total process threads.
Internal dense eigensolvers still require admission of their actual numerical
pool. Basis rows, moment terms, exact form rows, and interval block updates use
bounded batches; every reduction retains its original order. Exact coefficient
and dense block reservations follow their separate ownership lifetimes.

Exact form construction uses a streamed integer route when source exponents,
precision, and every ordered intermediate fit its proved dyadic bounds. It
retains the basis, a bounded batch of matrix-vector rows, and triangular form
accumulators. Integer products have bounded scratch; final conversion rounds
once to nearest with ties to even. Worker admission shrinks the batch to fit
the remaining run budget. Inputs outside the proved domain use the original
exact-rational route and its resource checks. Both routes preserve the same
rounded output and mathematical scope. These bounds describe owned allocation
estimates, not measured process memory or a promise that every configuration fits.

Exact tail splitting scans stored differences in two ordered passes, retaining
one rational difference at a time. The first determines the required output
precision after all source conversions succeed. Scalar scratch is admitted
before conversion, and promoted output is admitted before its allocation; its
reservation stays with the caller through serialization. Precision promotion
preserves the stored exact difference and adds no source accuracy.

Retained captures execute requested diagnostics inline under their shared run
budget. They do not launch speculative lookahead: optional copied inputs or
completed results must not prevent requested work from obtaining its workspace.
This removes speculative overlap for budgeted capture. The standalone lookahead
lane also declines prefetch whenever a run resource scope is active, from a
Rayon worker, or under SafeCapped. It uses the existing global numerical pool
and does not construct a second pool. The run
records its numerical-pool worker limit; owner and coordinator threads can also
perform serial work. This limit is not a total process-thread ceiling. An
application moving a run to a larger numerical pool must create a run configured
for that pool instead of exceeding its recorded limit.

Outside a run resource scope, at most one completed prefetch result waits.
A rejected prefetch is discarded and computed inline. Budget-limited finite
captures are unretained
so a later run with sufficient resources can retry. Conservative reservations can
block work that an unaccounted execution might fit; retain the blocked reason
when comparing coverage across resource settings.

Arithmetic-energy admission counts owned parsed component and trial arrays,
live directed-interval vectors, parsing scratch, and report representations.
Its Tau matrix is borrowed and remains charged by the retained run. The estimate
allows the full 4096-bit guard ladder, whose retries are sequential. A fresh
managed computation keeps its reservation through report validation,
serialization, and export. Successful numerical records and identities are
unchanged; insufficient-budget results remain unretained.

Telemetry reports retained, active-working, and peak reserved bytes. Concurrent
stage progress reports run totals, not exclusive per-stage allocations. Do not
hold a reservation while waiting for dependent work that requires another;
same-thread nested waits fail instead of deadlocking.

## Arithmetic and checkpoints

Directed inertia and ground-index calculations retain arithmetic order while
reusing scratch values and avoiding redundant conversions. Decimal text retains
MPFR rounding, signs, significant-digit counts, and notation. Allocation scales
with significant digits and exponent text rather than exponent magnitude; the
supported exponent domain is Rug's signed 32-bit decimal exponent.

Public parallel MPFR kernels check exponent-range agreement across their
numerical pool. The infallible quadratic form uses the same serial arithmetic
order if parallel admission fails. Embedders must keep worker exponent state
stable throughout each call.

Root validation preserves result order and the first error in input order.
Execution checkpoints remain bound to the source build and are revalidated.
They are separate from portable published-artifact identities.

Coverage counts must fit the retained-row cardinality. Inconsistent records are
unassessed with a reason. Operator residual totals include declared action
allowances but do not establish a floating-point roundoff certificate.
