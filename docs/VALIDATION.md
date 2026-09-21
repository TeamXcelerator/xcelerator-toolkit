# Release validation

## v0.15.1

The Gaussian-target amendment replaces first-small-term stopping with a
remaining-tail check, rejects HP term-budget exhaustion and binds the corrected
semantics in cache identities. Five regressions and 163 independent Arb checks
pass, including unequal scales and the historical reference descriptor through
6,708 bits. Complete local v9 native and HP/Arb qualification passes. Historical
target-dependent artifacts and the full mathematical/Research audit remain open.

The capability-identity amendment binds Arb availability in transform-enclosure
and band requests. It prevents an Arb-enabled run from reusing a feature-required
absence result and preserves same-capability warm reuse. Optional Boolean schema
fields preserve legacy readability while rejecting invalid types. Complete local
v8-r2 native and HP/Arb release qualification passes. The corresponding cache
schema mirrors await publication; this run did not request mirror checks.
The full mathematical and historical research audit remains open.

The stable u-flow amendment separates fresh derivative actions from historical
v3 arithmetic using a v4 identity. The generic binary64 root amendment prevents
infinite points labeled refined and discovery windows lost through overflow.
Counterexamples and targeted regressions pass, along with complete local v7
release qualification. This does not clear all historical artifacts or research.

The nonfinite eigenpair amendment rejects NaN/infinite inputs and nonfinite
residual arithmetic. Previously a NaN row could disappear from a maximum fold
and leave a zero residual. Explicit invalid-input and finite-overflow examples
now fail closed; the complete local v5 release tiers pass. This validation repair
does not establish lowest-state selection or clear the entire historical corpus.

The root and revocation amendment requires a directed upper bound on the Newton
correction at the stored finite-source point before reporting convergence. Root
identities bind the exact secular source; fixed roots use payload schema 6.
State selection checks ambiguity at the final optimum. Bootstrap readers enforce
active revocations during ordinary, historical and materialization lookup in
both cache lanes. Complete local v4 qualification passes. Root correction checks
do not certify existence, uniqueness, eigenstate accuracy or zeta correspondence.

The full-mathematics audit amendment fixes directed integer interval conversion,
finite bisection handling, high-mode f64 quadrature, even-sector Ritz selection,
and stable f64/HP CCM kernels and cutoff validation. It also rejects missing
source edges and revalidates adopted canonical manifests during offline reuse.
These repairs pass both complete local release tiers. Independent numerical
comparisons are recorded in the 2026-09-23 full-mathematics revalidation study.
The entire mathematical audit remains open; passing regressions do not clear
all historical results or research conclusions. The earlier amendments below
describe their own historical scope.

The cache-provenance amendment preserves every exact parent named by a staged
child, including cross-family dependencies and coalesced closure aliases. It
keeps newer live-index selections intact and blocks historical fallback for
explicitly quarantined or revoked manifests. Complete native and HP/Arb
qualification passes. Historical cache repair is recorded separately in the
2026-09-23 cache-provenance repair study; numerical formulas are unchanged.

The f64 origin-limit amendment adds direct-integrand and guard-reachability
regressions. Both workspace tiers pass, with all six native checks retained
from the same-source review and complete HP qualification repeated locally.
The guard is inactive for integer C>=2; fractional cutoffs near 1 can reach it.
HP formulas are unchanged. These checks do not clear the historical corpus.

Install the validation prerequisites, then use the documented native/HP build
prerequisites. The record captures the installed schema-engine version.
Run complete qualification from a committed source tree:

```sh
python3 -m pip install -r tools/requirements-validation.txt
python3 tools/check_release.py --tier hp --complete --metadata-root /path/to/repos --output /path/to/new-qualification
```

On Windows use `--tier native`. Omit `--metadata-root` when artifact repositories
are unavailable; the output explicitly marks mirror checks as not requested.
The HP complete command builds the ordinary backfill executable, exercises all
thirty kinds, validates positive and negative schema cases, checks interruption
and budget recovery, regenerates schemas for comparison, checks documentation
reachability and writes the reproducible source digest. All 34 schema files are
covered: 30 are regenerated, and four hand-maintained contracts have explicit
LF-normalized SHA-256 pins in `tools/handmaintained_schemas.json`. Intentional
hand-maintained contract edits require reviewed pin updates. Qualification rejects
optimized Python (`-O` or `PYTHONOPTIMIZE`) so child
acceptance scripts retain their assertions. Main gates also use explicit errors.

The complete command checks prerequisites, committed source identity and exact
repository bytes before compiling, and repeats the committed-source check during
asset qualification. It rejects CRLF text staged in the index, CR/BOM script
bytes in HEAD/index/worktree, and changes to the pinned raw reference table.
Source digests include the table without line-ending normalization. The byte
pins are in `tools/byte_exact_inputs.json`.

For local development, run the native/HP tier without `--complete`; final release
qualification requires committed code and tooling. The final validation record is
a documentation-only amendment of the qualified source snapshot: its recorded
source digest must still equal the final release's committed source digest.
No campaign, historical backfill, artifact publication or GitHub Actions runs.

The [machine-readable record](validation/v0.15.1.json) binds the qualified
source files, toolchains, commands, exit statuses and log hashes.

The published-source capture amendment also checks canonical factor/sector
ancestry, altered parent rejection, root/secular binding, cached prefix/reduction
reuse, and observation of larger retained root windows without changing the
requested numerical result. Checkpoint I/O is quiet by default. New controls cover
source-bound atom/tail preparation, complete origin mass, interval block bounds,
finite polynomial bands, centered contour enclosures and HP-only compatibility.
The near-carrier amendment adds a convergent sinc coefficient series with an
explicit remainder, stable tiny-argument derivatives, and energy-allowance
scale interpretation. Shifted/complex carriers, branch boundaries and zero
or negative trial energies have regressions; the retained C13/N120 contour
replay still certifies its finite-function count with no unresolved rows.

| Check | Result |
|---|---|
| Windows default workspace, all targets, debug | 571 passed; 0 failed; 6 ignored |
| WSL Ubuntu 24.04 HP + Arb workspace, all targets, release | 1139 passed; 0 failed; 38 ignored |
| Strict workspace and external-consumer Clippy, both tiers | Passed |
| Rustdoc with warnings denied and doctests, both tiers | Passed; no runnable doctests |
| Independently locked external consumer, both tiers | 3 passed on each tier |
| Additive backfill CLI | Thirty kinds; cold/warm identity, corrupt input, failed rows, resume and output-preservation checks passed |
| Payload JSON Schema validation | All thirty generated packets passed; negative schema cases and thirty previous packets checked |
| Shared registry/shard metadata | Previous mirror qualification preserved; current capability schema mirror update pending |
| Artifact-impact inventory unit tests | 11 passed |
| Chunk preparation, research query and publication-summary tests | 5 passed |
| Release guards (source provenance, schema inventory, optimization, raw bytes) | 5 passed |
| Fresh Git clones, autocrlf=true and false | Identical pinned reference bytes, LF scripts and committed source digest |
| Historical Windows cache tests from a 132-character checkout path, d608eab | 285 passed; 0 failed; 6 ignored |
| Band interruption, worker identity and raised-budget recovery | Passed synthetic end-to-end checks |

Analytic regressions cover Fourier carrier limits and derivatives, odd/complex
state handling, raw nonorthogonal projection coefficients, singular projection
inputs, total matrix energy, exact ancestry, adaptive root precision, missing
rows, resource budgets and frozen finite cohort rules. Extended regressions
cover analytic origin moments, weighted reference conventions, signed transform
channels, component closure, projected directional response, weighted tail
coverage, cluster matching and conditional error/energy allowances. New checks
cover complex analytic continuation, closed contour retention, signed support
motion, all parent-derived prefixes, Schur feedback, workspace limits, tail-model
energy independent of the retained eigenvalue, and foreign optional input isolation.
Completion regressions cover signed-measure positivity failure, independent
component defects, explicit cohort comparisons, source-independent preparation,
finite moment assembly, checkpoint corruption, adaptive contour exhaustion and
a replayed finite source-certificate angle allowance. Additional regressions
cover changed Git blob storage, mixed metadata/archive import, durable attempt
metrics, band moments, explicit reference joins, tail-on/off model energy and
tiny-energy vector resolution. A deterministic lease test covers backward
wall-clock corrections without changing the fencing generation. Production
adapter tests exercise cold/warm capture and encoded publication staging without
remote writes.

Atom regressions cover inputs beyond the former count limit, authenticated chunks,
self-atom exclusions, coincident evaluations, signed kernel sums, fixed cutoff
energies and band roots, exact decimal exports and deterministic block reductions.

Runtime target regressions cover normalization, content identity, schema policy,
and rejection of nonfinite distances. External providers have separate cutoff,
precision, executable-digest, and protocol checks. These are software checks, not
certificates for an externally defined mathematical function.

The two suite counts overlap. Ignored tests, including live GitHub checks, were
not executed. No GitHub Actions ran. This qualification does not rerun a paper
claim, execute historical backfill, certify an infinite-dimensional result, or
claim a new campaign speedup.

## v0.15.0 historical qualification

The v0.15.0 numerical source passed the checks below. The
[machine-readable summary](validation/v0.15.0.json) identifies the tested
source digest, features, counts and scope.

| Check | Result |
|---|---|
| Windows default workspace, all targets, debug | 533 passed; 6 ignored |
| WSL GNU/Linux HP + Arb workspace, all targets, release | 1023 passed; 37 ignored |
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

The complete positive-root correction repeated both workspace suites, both
strict workspace and external-consumer Clippy tiers, both consumer test tiers,
HP rustdoc and doctests. Four added regressions cover the beyond-band roots,
unsupported inputs, explicit incomplete counts and distinct cold/warm identities.
An explicitly invoked retained-source test also recovered all 50 positive movable
roots at C=2500, N=50, 3386-bit source precision: all refinements converged, with
byte-identical root payloads under cold acquisition, required reuse and refresh.
This is finite point-source qualification, not a full spectral ordinal certificate.
See [complete root discovery](CCM_COMPLETE_ROOT_DISCOVERY.md) for prerequisites,
assurance limits and cache compatibility. Earlier performance records were not
rerun for this correction.

The response performance amendment repeated both workspace and external-consumer
test and strict Clippy tiers, HP rustdoc, doctests and formatting. Four additional
regressions preserve full response bytes and root arithmetic across precisions
and worker counts, and reject corrupted or mismatched inputs. Two optional
synthetic benchmarks ran three times each; these executions are separate from
the ignored-test totals. See [response performance](CCM_RESPONSE_PERFORMANCE.md)
for the measured scopes and fresh-versus-cached validation policy. Existing
scientific and unrelated performance records retain their original scope.

The publication transport amendment repeated both complete workspace suites,
both strict workspace and external-consumer Clippy tiers, both consumer test
tiers, HP rustdoc and doctests, formatting and the artifact-impact checks.
The added progress regression permits only structured numerical Git progress.
Local Git integration checks retain atomic ref updates, historical blob handling,
prepared-object reuse and interrupted publication behavior. Four retained-sample
packs were byte-identical while local packing became faster; see
[publication performance](PUBLICATION_PERFORMANCE.md) for scope and timings.
No live GitHub publication speedup or scientific rerun is claimed.

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

The publication-recovery amendment repeats both full workspace tiers, both strict workspace Clippy tiers, HP rustdoc and both external consumers. Four regressions cover retained production after a staging limit, exact dependency recovery and reopening, corrupt or missing objects, and bounded resource-policy input. This is offline qualification; no live large-artifact recovery is claimed.
