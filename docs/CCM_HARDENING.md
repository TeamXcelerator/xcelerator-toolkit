# CCM 0.14.4 qualification and preservation record

## Authorization and scope

Ronnie Andrews, Jr. (TeamXceleratorDev) explicitly requested the Toolkit audit
fixes, performance/analysis improvements, and integration into `main` in the
September 5–6, 2026 CCM conversation. The original baseline is
`e4e8ec5912c12ae12836187813b6962e5cd71aa6` (0.14.3); the source work continues
from `afe5d30e6e73e535d685899fedf584dd5209b9b1` on the audit work branch.
Implementation assistance is AI-generated. No independent human review,
private release audit, or universal mathematical accuracy guarantee is claimed.
No private registry or warehouse artifact was changed by this implementation.

The integrated scope and desktop follow-through are described in
[CODEX_HANDOFF.md](CODEX_HANDOFF.md). New prefix diagnostics and precision/basis
contracts are described in [CCM_PREFIX_ANALYSIS.md](CCM_PREFIX_ANALYSIS.md).

## Completed local qualification — September 6, 2026

Fixed Linux host, Rust/Cargo 1.98.1, locked declared dependencies, existing
optional dynamically linked FLINT 3.0.1 prerequisites. Numerical tests used
isolated caches with remote access and publication disabled. No paper-sized
private source payload was required. These local logs supersede earlier failed
scratch-source/workflow attempts; they do not erase those historical attempts.

| Check | Observed result |
| --- | --- |
| Formatting | Passed |
| Default workspace tests | 499 passed, 0 failed, 6 ignored |
| HP + Arb workspace tests | 921 passed, 0 failed, 27 ignored |
| Final finite-value guard recheck | Same 921 HP + Arb tests passed; Clippy and docs passed |
| Warnings-as-errors Clippy (all targets, HP + Arb) | Passed |
| Warnings-as-errors documentation | Passed |
| External HP consumer | 1 passed |
| Read-only inventory regression | 6 passed |
| Retained-file prefix example | Compiled and executed successfully |
| Release prime component benchmark | Passed numerical checks and recorded timings |
| Fresh-process established/research-route repetition | 84 runs; exact numerical bytes repeat |
| Fresh-process new retained-prefix exports | 30 runs; exact report bytes repeat |
| Cached numerical payloads and encoded packages | 636 artifact records checked; exact matches |
| Original 0.14.3 vs candidate ordinary routes | Five scenario snapshots exactly match |
| New reader with strict 0.14.3 cache reuse | Five scenarios pass; no added/changed artifact/object files |

Ignored tests are explicitly ignored expensive/manual cases, not a claim that
every ignored benchmark or every feature/platform combination was executed.
The detailed machine-readable receipt is
[validation/ccm-0.14.4.json](validation/ccm-0.14.4.json).

### Repetition and cross-version sample

The 84-run sample consists of ten fresh processes with two Rayon workers, plus
one each with one and four workers, for each of seven routes:

- Ordinary automatic even-sector solver and independent three-root discovery:
  c=13,N=12,p=256; c=100,N=24,p=512; c=500,N=32,p=1024 working bits.
- Explicit legacy inverse iteration at c=13,N=12,p=256.
- Indexed reference-seeded three-root acquisition at c=13,N=12,p=256.
- Exact noninteger research cutoff just below 13, aggregate-prime arithmetic,
  and separately requested quadrature bucketing, N=12,p=256.
- Corrected cutoff-free assembly / finite-matrix inertia certificate c=5,N=2,p=192.

Every run used a fresh initially absent local cache. The output comparison
excluded elapsed runtime, not numerical fields. Generated cached payload and
package bytes, and exact source dependency bindings, were compared without
numerical exclusions; local manifest creation timestamps were not expected to
repeat. The new prefix example additionally ran ten times on each of the three
retained ordinary example source matrices/eigenpairs, preserving those source
files and passing every endpoint export screen. The final finite-value guards
were recompiled and those 30 prefix runs repeated successfully.

The original 0.14.3 source was reconstructed from the retained source archive
and reversed audited source patch. Its five ordinary scenarios exactly match
the retained 0.14.4 reference snapshots, which the candidate independently
reproduced. Then the candidate consumed those freshly generated 0.14.3 caches
under `require_reuse`, returned byte-identical numerical snapshots, and left
159 combined artifact/object files unchanged across those five cases. This is
sampled cross-release preservation, not an all-ten-private-repository audit.

### Mathematical and export tests

The prefix tests use independently implemented exact-rational Gauss–Jordan
inversion as an oracle for 21 prefixes across five synthetic matrices (including
Hilbert, widely separated scales, and near-degenerate dyadic examples). They
check pivots, innovations, moments, rejection/stopping behavior, and decoded
cancellation-sensitive exports. An 80-digit output width is not assumed adequate
for every identity; the original identity is checked before deterministic
width escalation, and decoded values are checked again before acceptance.

### Performance measurement limitation

The release benchmark compares the frozen 0.14.3 prime component, the
operation-order-preserving scratch-reuse route, and the distinct aggregate
research route at c=500,p=256,N=32,64,128. Exact equality is required between the
first two; a declared tolerance applies to the aggregate route. Timings here
were collected on a shared/contended host and are not a stable whole-solver
speedup claim. Use isolated cold/warm deployment measurements for decisions.
The streaming even-sector validator removes a temporary full expected matrix
while retaining the old projection arithmetic and full-Tau eigenstate checks.

## Data preservation and assurance

Existing ordinary artifact keys, compatibility floors, arithmetic sequences,
source-selection rules, and capture structs remain unchanged. New diagnostics
are separately keyed derived children. The new `ccm_prefix_analysis` kind is
private-only; the live private registry was not edited. The sole targeted
existing-kind producer/reader-floor change is the defective cutoff-free
`ccm_sector_gap_certificate` (0.14.4 minimum). Corrected certificates intentionally
have different schema/semantics; no unrelated data are invalidated by that fix.

A repeatable result is not automatically an accurate approximation. Exact
cross-version agreement, same-version repetition, numerical backward-error
screens, independent algorithm checks, and interval certification remain
separate claims. Generalized prefix moments and proof-specific interval
certification of the new diagnostics are not implemented by this extension.
A supplied finite matrix is not promoted to a certified underlying CCM form.

## Not included in this Toolkit-only integration

Paper 1/Paper 3 CLI adoption and claim-driver completion; private registry
registration; the two retained research ladders/nine checkpoint comparison;
owner private release audit and independent review; intended Windows/WSL large-N
qualification; proof-specific certification of new diagnostic claims. No release
tag or private warehouse publication is implied by merging source into main.
