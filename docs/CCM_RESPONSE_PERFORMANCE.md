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

## U-flow event preparation

U-flow reuses exact coefficient rows and their directed interval conversions
across bounded prime-power event batches. `XC_RESPONSE_EVENT_BATCH` is a positive
integer (default `8`); `1` selects single-event storage. The declared working
budget and shared memory ledger admit batch storage before allocation, including
the outer quadrature workspace. If optional CPU batching does not fit, events
use the original streaming path. A larger batch setting is a scheduling choice,
not a larger mathematical domain or a change to the precision ladder.

Each CPU event retains its original multiplication, directed summation,
rounding, and signed-zero behavior. Events and errors are selected in canonical
order even when different events require different precision guards. Returned
batch storage remains reserved until its consumer finishes. The stored GL-node
quadrant test rejects impossible dyadic denominators without allocating a
rational; possible exact quadrants still use the original exact calculation.

Independent sine/cosine phase evaluations use bounded chunks on the existing
Rayon pool. Their interval operations and output order are unchanged. Workers
inherit the run resource scope, and a guard checks matching native exponent
ranges once before dispatching its first resolved event.

These changes preserve stored quadrature tables and successful response values.
They do not remove cold quadrature generation, transcendental evaluation,
bordered solves, or private input preparation. Measure those stages separately
before extrapolating a kernel improvement to total run time.

### Exact GPU prime contractions

`XC_PRIME_RESPONSE_BACKEND=cuda` selects exact GPU contractions for the prime
actions inside u-flow event batches. The common rational coefficient matrix is
represented as an integer numerator matrix and one positive denominator. Two
exact products contract interval centers and outward radii; final interval
arithmetic remains directed. Only a uniquely rounded nonzero action is accepted.
Zero or unresolved actions use the original row calculation at the same guard,
and mathematical row errors replay the original ordering before selection.
CUDA execution failures propagate. They do not silently switch to a CPU backend.

`cpu` is the default. `auto` currently uses the CPU route until complete stage
measurements establish a winning GPU range. This setting does not change dense
LU backend selection or accelerate standalone prime-power response calls.

GPU storage is tiled and configurable:

| Setting | Default | Meaning |
| --- | --- | --- |
| `XC_PRIME_RESPONSE_GPU_MAXIMUM_BYTES` | 4 GiB | Maximum contraction workspace, further bounded by the current run's remaining admitted budget. |
| `XC_PRIME_RESPONSE_GPU_TILE_ROWS` | 64 | Rows per integer tile; positive values through 8193 are accepted and capped to the actual row count. |
| `XC_PRIME_RESPONSE_GPU_HOST_WORKERS` | At most 4 | Host preparation/reconstruction workers, bounded by the pool, caller, and shared worker policy. Explicit oversubscription is refused. |

These controls are operational and do not enter numerical artifact identities.
The strict CUDA route requires admitted common-denominator preparation, batch
storage, and an active run resource owner. Reservations cover owned operands,
integer conversions, reconstruction, results, and admitted device buffers.
Vendor context allocations remain outside the estimated ledger. All compatible
visible devices use the existing exact product scheduler; this adapter adds no
device-specific arithmetic or reduced-precision approximation.

## Fresh-data validation

A newly computed prime-power or u-flow response has already passed its producer's
spectral-isolation and bordered-residual checks. Its exact typed JSON is bound to
a process-local SHA-256 seal. The cache publication boundary checks this seal
instead of immediately repeating the full numerical response calculation.
Hashing streams through a bounded buffer; it does not allocate another complete
JSON copy merely to bind or check the seal. Any payload change invalidates it.
The seal is neither persisted nor accepted for a later process or cache read.

Ordinary computed reuse of a response artifact checks its exact source joins,
schema, parity, roots, cardinalities, finite values, vector norms, recorded
residual acceptance and the source-bound spectral isolation, without replaying
each event's bordered solve; u-flow derivative actions are computed only on a
miss. Explicit cache verification/comparison modes and requests above computed
assurance retain full numerical replay, which for prime-power responses uses the
prepared root quantities and bounded parallel event path. No assurance level is
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
