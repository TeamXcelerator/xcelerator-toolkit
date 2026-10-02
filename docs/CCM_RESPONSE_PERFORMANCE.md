# Response capture and validation performance

Prime-power response capture retains each active prime power's effect on the
state, eigenvalue and requested roots. Its cost grows with the number of prime
powers, the matrix dimension, the root count and the arithmetic precision.
A warm primary matrix and root window do not imply that this supplemental
response artifact is already available.

## Computation and reuse

The response implementation prepares fixed root derivatives once per source and
root selection. It optionally retains rounded pole denominators, with a 512 MiB
estimated table budget. Above the budget it regenerates denominators while still
reusing the derivatives. This keeps the optional table from growing without a
bound as dimension, precision or root count increases.

Independent prime-power events run in at most sixteen concurrent chunks. Matrix
rows, phase evaluations and root batches use the existing Rayon pool. Each root
batch reuses MPFR reduction storage. The original scalar rounding operations,
root/event order and adjacent-pair summation tree are preserved. Division remains
division; reciprocal multiplication and reassociated sums are not substituted.
Event errors are reported in canonical event order.

Generation and numerical replay print progress at event boundaries, approximately
every thirty seconds for larger jobs, and at completion. The line distinguishes
`compute` from `validate`, and includes completed/total events and elapsed time.
An individual event can exceed the reporting interval; the line is not an ETA.
Operational progress does not enter artifact payloads or cache identities.

## Fresh-data validation

A newly computed prime-power or u-flow response has already passed its producer's
spectral-isolation and bordered-residual checks. Its exact typed JSON is bound to
a process-local SHA-256 seal. The cache publication boundary checks this seal
instead of immediately repeating the full numerical response calculation.
Hashing streams through a bounded buffer; it does not allocate another complete
JSON copy merely to bind or check the seal. Any payload change invalidates it.
The seal is neither persisted nor accepted for a later process or cache read.

Reused response artifacts still undergo source/identity and numerical replay
checks. Their prime-power replay uses the prepared root quantities and bounded
parallel event path. Explicit cache verification/comparison modes also retain
full numerical replay for freshly recomputed results. No assurance level is
upgraded by a fresh-data seal, and it is not a mathematical certificate.

Existing response artifact types, semantic keys, dependency bindings and payload
formats are unchanged. Existing data does not need repair or deletion for these
performance changes. Consumers pinned to an older implementation continue to use
that implementation until their Git dependency and lockfile are updated.

## Verification and measurement

Regressions compare complete response JSON with the frozen prior scalar
implementation, check root responses at 128 through 6708 bits and across worker
counts, exercise the denominator-budget fallback, and reject changed payloads,
wrong sources, poles and zero derivatives. Existing cold/reuse/refresh and repair
regressions remain in the test suite.

The explicitly invoked `response_root_kernel_benchmark` checks the root-response
kernel at dimension 801, 400 roots and 6708 bits, including preparation time.
`response_validation_benchmark` compares complete synthetic response generation,
numerical replay and fresh-data seal costs. Both require exact output equality.
These are software performance controls, not new CCM research observations or an
end-to-end claim runtime forecast. Use release builds with the same source,
precision, worker count and cache policy when comparing runs; see
[performance reporting](PERFORMANCE_REPORTING.md).

The fixtures are deterministic software controls; their root points are not
claimed to be physical CCM roots. Payloads match the prior implementation byte
for byte.

Complete positive-root discovery currently performs its numerator isolation
before looking up the refined root window. This change does not eliminate that
separate warm acquisition cost; it optimizes response production and validation.
