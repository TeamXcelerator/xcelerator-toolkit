# Release validation

The v0.15.0 numerical source passed the checks below. The
[machine-readable summary](validation/v0.15.0.json) identifies the tested
source digest, features, counts and scope.

| Check | Result |
|---|---|
| Windows default workspace, all targets, debug | 528 passed; 6 ignored |
| WSL GNU/Linux HP + Arb workspace, all targets, release | 1010 passed; 34 ignored |
| Column scheduling, frozen-reference compatibility and stop order | 2 passed; overlaps the workspace suite |
| Final capture source-binding regressions | 12 passed; overlaps the workspace suite |
| Read-only artifact-impact inventory unit tests | 11 passed |
| Publication alias staging, closure and destination reuse | 5 passed; overlaps the workspace suite |
| Canonical managed receipt reuse | 2 passed; overlaps the workspace suite |
| Bounded local-shard reader regressions | 3 passed; overlaps the workspace suite |
| Strict Clippy: both workspace tiers and external consumer | Passed |
| Rustdoc with warnings denied; workspace doctests | Passed; no runnable doctests |
| Independently locked external consumer, native and HP | 2 passed on each tier |
| Public capture and evaluation examples | Capture, freeze, score, replay, store and required reuse passed; both serialized output schemas validated |
| Registry family/active-shard metadata | All 18 pairs passed; 7 shared schemas byte-identical |

The HP/Arb suite ran with Rust 1.98.1 in optimized release builds. Default
Windows checks ran with Rust 1.98.0 in debug builds. Ignored tests are excluded
from workspace totals; executed optional benchmarks are recorded separately.
Counts from the two workspace configurations overlap. Capture completeness and numerical acceptance
remain separate assessments. The tests do not constitute independent
scientific replication of an application dataset.

The root-response correction repeated both complete workspace suites and
strict workspace Clippy tiers, HP rustdoc, the HP external consumer, and the
artifact-impact tests. A new analytic regression checks a genuine secular
root with a tiny boundary sum: production and replay meet the prescribed
accuracy while the former normalization-gauge arithmetic fails it. Other
targeted qualification and performance measurements retain their earlier
scope and source records; the amendment claims no new campaign speedup.

The retained-response repair amendment repeated the complete workspace suites,
strict workspace Clippy tiers, HP rustdoc and the HP external consumer. Its
regressions compare repair against fresh production and warm validation, reject
tampered or unbound inputs, and preserve unsuccessful capture outcomes. Offline
repair and explicit publication are documented in the [repair guide](CCM_RESPONSE_REPAIR.md).

The published-record reader correction repeated both complete workspace suites,
strict workspace Clippy, HP rustdoc and both external consumer tiers. New tests
construct the actual GitHub adapter, then require receipt reuse using canonical
source identities and metadata-only quality checks. They cover local retention,
logical aliases, missing or mismatched sources, and unchanged failed outcomes.
No artifact identity or numerical payload changes for this reader correction.

The publication-alias correction repeated both complete workspace suites, strict
workspace Clippy, HP rustdoc and both external consumer tiers. Five regressions
cover actual staging/reopening, recursive aliases, both destinations, repeat
publication verification, assurance/evidence retention, unbound transports and
distinct historical closures. These are offline software checks; this amendment
does not claim a successful live application publication or new numerical data.

A cold ZIP-cache regression uses the actual canonical publication staging sink
and checks newly computed primary eigenstates, retained matrices, all six
previously missed diagnostic groups, and their warm payload identities. This
exercises encoded production with local staging; it makes no remote mutations.

The retained-run adapter regression verifies failure isolation, authenticated
warm sources, target-independent profile byte identity against the established
distance route, and preservation of natural primary states. A bounded real
application smoke run at lambda-squared=13, N=16, HP-40 captured all 13 Ultra
requests. Missing-target and natural-parity runs retained explicit incomplete
outcomes without losing the primary result. These are integration checks, not
reruns of published paper claims. Earlier performance measurements retain their
original source digest in the machine-readable record; this adapter amendment
does not claim new campaign speedups.

## Reproduce the checks

Use Rust 1.98.1 for the HP/Arb qualification and 1.98.0 for the Windows
baseline, retaining the locked dependencies. The development toolchain file tracks `stable`; set
`RUSTUP_TOOLCHAIN=1.98.1` explicitly for an exact compiler selection. Install the
native prerequisites described in [Getting started](../README.md#getting-started)
and [High precision](../README.md#high-precision).
Run default tests on the host and HP/Arb checks on a supported GNU/Linux setup.
Use a fresh scratch cache with remote acquisition and publication disabled:

```sh
export RUSTUP_TOOLCHAIN=1.98.1
export XC_CACHE_ROOT="$(mktemp -d)"
export XC_CACHE_REMOTE=none
export XC_PUBLISH_TARGET=none
export XC_PUBLISH_EXECUTE=false
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked --no-fail-fast
cargo test --workspace --all-targets --release --features hp,arb --locked --no-fail-fast
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --features hp,arb --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --features hp,arb --locked
cargo test --workspace --doc --features hp,arb --locked
cargo test --manifest-path tests/external-consumer/Cargo.toml --locked
cargo test --manifest-path tests/external-consumer/Cargo.toml --release --features hp --locked
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

The [manual qualification workflow](../.github/workflows/ccm-qualification.yml)
runs the main checks in an isolated cache. It is explicitly dispatched and does
not run on every commit. Its optional release benchmarks are separate from the
default suite. Enable `retained_qualification` for the optimized workspace
checks and bounded retained-evidence measurements. See [external consumer validation](EXTERNAL_CONSUMER.md) for the
standalone application boundary.

## Retained-evidence release measurements

Run `bash tools/qualify_retained_release.sh` with the HP/Arb prerequisites.
The primary timing fixture is the dense symmetric matrix A=HDH, where
H=I-2uu^T/(u^Tu), u_i=i+1 and D_ii=1+i/n for i=0,...,n-1. Entries are
constructed as exact rationals, then rounded once to source precision. The
analytic eigenvalues belong to the ideal rational matrix; measured discrepancies
include source rounding. Every entry is nonzero and all n-1 reduction
subdiagonals are nonzero, both checked by the example.

All 13 cases supply an analytic eigenstate, check decoded vector exports,
persist prefix/reduction children and a receipt, then compare serialized
cold/warm results under required reuse. Every eigenvalue discrepancy is retained.
The three extended-policy cases additionally compare the third inverse moment
with the analytic inverse-cube sum evaluated at 64 extra bits. One/four-worker
receipts match within each policy. Timings exclude concurrent qualification builds.

Here **legacy** means two moments with cancellation, **full** adds the third
moment (the new Ultra default), and **no cancellation** retains the third moment
while omitting innovation cancellation. Each row is one measurement.

| Policy | Dimension | Bits | Workers | Cold (s) | Warm (s) | Peak RSS (MiB) |
|---|---|---|---|---|---|---|
| legacy | 512 | 256 | 1 | 36.655 | 0.190 | 107.6 |
| legacy | 512 | 256 | 4 | 10.576 | 0.188 | 119.6 |
| legacy | 1024 | 256 | 4 | 78.207 | 0.364 | 453.2 |
| legacy | 256 | 256 | 4 | 1.667 | 0.093 | 35.3 |
| full | 256 | 256 | 1 | 4.974 | 0.122 | 34.1 |
| full | 256 | 256 | 4 | 1.813 | 0.123 | 36.8 |
| no cancellation | 256 | 256 | 4 | 1.729 | 0.103 | 36.8 |

Separate uncached stage replay (all dense-reflector):

| Policy | Dimension | Workers | Prefix (s) | Householder (s) | Assessment (s) | QR (s) |
|---|---|---|---|---|---|---|
| legacy | 512 | 1 | 6.477 | 15.282 | 13.977 | 0.270 |
| legacy | 512 | 4 | 1.866 | 4.358 | 3.727 | 0.268 |
| legacy | 1024 | 4 | 13.753 | 31.560 | 30.702 | 1.017 |
| legacy | 256 | 4 | 0.341 | 0.648 | 0.475 | 0.072 |
| full | 256 | 1 | 0.849 | 1.926 | 1.728 | 0.073 |
| full | 256 | 4 | 0.427 | 0.626 | 0.474 | 0.073 |
| no cancellation | 256 | 4 | 0.418 | 0.624 | 0.463 | 0.073 |

Stage replay timings do not sum to cold-cache time. Multi-worker factorization
distributes independent lower-factor entries within each column, retaining
ordered pivots and stop screens. Independent innovation columns run in parallel
with serial inner trees,
then moments accumulate in prefix order with parallel Gram terms. Long factor
sums and reduction assessment rows also run in parallel. Default numerical
payloads preserve the established arithmetic and reduction tree. The third
moment adds a retained Gram triangle and a quadratic border form per prefix;
it has O(D^3) total work and O(D^2) storage, with a measurable additional cost.

All thirteen preceding fixtures, including full and cancellation-omitted
policies, retain their receipt identities,
checkpoint scalars, inverse trace errors, spectra and reduction residuals.
The six structural cases use A=I+11^T/n (dimensions 256, 512 and 1024,
including a 512-bit case), or tridiagonal Toeplitz input with diagonal 2 and
off-diagonal -1/2. Their spectra are {1,...,1,2} and 2-cos(k*pi/(n+1)).
The rank-one innovation/top-eigenstate overlap is checked against 1/(4n-3).

Rank-one reduction can collapse after the first reflection; Toeplitz input
is already tridiagonal. Their timings do not represent dense reduction costs.
The 1024-dimensional rank-one case can report zero spectrum discrepancy;
computed equality is not an error certificate. Warm costs include parsing,
identity and receipt validation. Peak RSS includes stage replay. These
synthetic measurements do not predict CCM campaign cost.

## Retained CCM precision comparison

The full suite constructs one c=13, N=16 CCM point matrix at 256 source bits
with declared Gauss--Legendre orders 128, 131, ..., 176. Identical even-sector
entries are analyzed at 256 and 512 working bits with the full diagnostic
policy; both ladders reach dimension 17.

For every common numeric scalar, the record retains signed (low minus high),
absolute and relative differences, plus empirical decimal agreement:
max(0, floor(-log10(abs(low-high)/abs(high)))). Binary points are decoded at
their own precisions before comparison at 576 bits. A zero difference or zero
reference has a null agreement count, with an explicit equality flag. Model
statuses and unavailable scalar paths are retained separately.

| Final-prefix scalar | Decimal agreement |
|---|---|
| `sigma` | 44 |
| `inverse_trace` | 44 |
| `inverse_square_trace` | 44 |
| `smallest_eigenvalue_lower_estimate` | 44 |
| `smallest_eigenvalue_upper_estimate` | 44 |
| `effective_inverse_rank` | 49 |
| `gap_ratio_estimate` | 44 |
| `smallest_eigenvalue_second_order_estimate` | 44 |
| `innovation_cancellation.decimal_digits_lost` | 47 |
| `third_inverse_moment.inverse_cube_trace` | 44 |
| `third_inverse_moment.smallest_eigenvalue_lower_estimate` | 44 |
| `third_inverse_moment.smallest_eigenvalue_upper_estimate` | 44 |
| `third_inverse_moment.two_mode_fit.smallest_eigenvalue_estimate` | 44 |
| `third_inverse_moment.two_mode_fit.second_eigenvalue_estimate` | 50 |
| `third_inverse_moment.two_mode_fit.relative_trace_closure_residual` | 44 |

These counts measure arithmetic agreement on one retained point matrix.
They are not certified accuracy or trustworthy construction digits: both
computations share source errors. Neither comparison certifies positivity,
assembly accuracy or an Ultra campaign. Two-mode fits remain model estimates;
three moments do not identify a general spectrum.

## Prefix column scheduling measurements

The paired runs below use identical dense signed, strictly diagonally dominant
input on four workers. Both runs use the phased ladder; only the factor schedule
changes from rows to columns. Complete reports match byte for byte. The third
moment is disabled in this comparison. Each pair is one measurement, with no
statistical performance guarantee.

| Dimension | Bits | Row factor schedule (s) | Column factor schedule (s) |
|---|---|---|---|
| 640 | 256 | 5.126 | 3.228 |
| 640 | 2722 | 64.696 | 32.568 |
| 1024 | 256 | 20.230 | 13.541 |

Small and single-worker inputs retain the row schedule. The parallel path
preserves every entry's products, divisions and adjacent-pair inner tree.
Entries in future rows may be computed early; their errors are inspected only
when the original row order reaches them. A speculative overflow therefore
cannot replace an earlier pivot stop. Regressions cover that case, late stops
and the first failing factor-entry coordinate.

Earlier phase-split and prime-component measurements remain in the
machine-readable record, explicitly marked as historical and bound to their
original source digest. They were not retimed for this factor-only change.
Canonical prime scratch reuse still has exact-point regression coverage.

Reproduce the column comparison with:

```sh
cargo test -p xc-numerics --release --features hp --lib qualify_prefix_column_factor -- --ignored --nocapture
```

## Other performance coverage

The nested-grid regression retains 17 points at 2Q for Q=8 and 33 after extension
to 4Q. When 2Q is sufficient, 16 evaluations are avoided. Shared values remain
identical in u and log-u coordinates. This is an evaluation-count result, not a
whole-application timing measurement.

The corrected tridiagonal solve retains linear factor storage and work.
Eigenvalue-only stable reduction omits Q. Prefix ladders currently cache scalar
and export policies together, so changing an export request can require a new
child computation.
