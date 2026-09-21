# Retained research artifacts

Toolkit v0.15.1 adds thirty managed artifact kinds. The original nine are
listed below; ten additional kinds are described in [extended diagnostics](EXTENDED_RESEARCH.md). These retain reproducible,
source-bound observations that can be read independently of a claim script.
They do not assert convergence, identify a ground state, or assume RH.

## Coverage and conventions

| Artifact kind | Retained data | Acquisition |
|---|---|---|
| `ccm_state_geometry_analysis` | Raw normalization and center, spatial moments and shells, evenness defect, sampled real sign/negative mass, two-grid sensitivity | Shared Ultra; retained eigenpair |
| `ccm_indexed_transform_analysis` | Signed finite Fourier transform and derivative at each supplied ordinate/root, absolute term sums, cancellation, guarded Newton correction, missing/unresolved rows | Shared Ultra at retained roots; explicit dataset through backfill |
| `ccm_operator_energy_analysis` | Full Tau Rayleigh energy, retained eigenvalue, eigenvalue defect, relative residual, absolute energy-term sum and cancellation | Shared Ultra; exact Tau-to-state ancestry |
| `ccm_root_band_analysis` | Entire retained root window, source ordinals/statuses/acquisition policy, counts, minimum positive spacing, first three positive inverse moments | Shared Ultra; exact root-to-secular-to-state chain |
| `ccm_reference_source` | Explicit finite Fourier coefficients, definition, precision, cutoff, approximation scope and definition digest | Caller-supplied reference; never executes code |
| `research_reference_dataset` | Explicit evaluation points or known zeta ordinates, coordinate, precision, attribution and missing rows | Caller-supplied data; not a zero certificate |
| `ccm_reference_projection_analysis` | Normalizations, signed overlap, difference norm, raw nonorthogonal Gram matrix/RHS, coefficients, direct fit residual and optional b/B2 convention | Explicit reference and up to eight basis functions |
| `ccm_stabilization_analysis` | Ordered retained eigenvalues at fixed cutoff/precision/selection policy, relative changes, frozen consecutive-step rule and result | Explicit multi-state cohort |
| `research_observation_packet` | Exact supplied text, digest, attribution, hypotheses, borrowed inputs and limitations | Explicit import; numerical assertions are not replayed |

Shape schemas are in [schemas](schemas/). Cache quality `Validated` means the
artifact passed its declared producer/reader checks. It does not turn a point
measurement or an imported assertion into a numerical certificate.

### Finite transforms

Use the [state-geometry Fourier convention](STATE_GEOMETRY.md), with
L = log(lambda_squared), x in [-L/2, L/2], and unit L2(dx) normalization.
The transform is the real quantity integral f(x) exp(i t x) dx for the retained
real-coefficient Hermitian state. Its derivative is evaluated analytically,
including removable Fourier-carrier singularities. No infinite physical tail
is inferred. The Newton correction is -F/F' only where the numerical guards
resolve both channels; no root-error remainder is claimed. The stored
sqrt(L^5/80) expression is the Cauchy-Schwarz second-derivative bound for the
unit-norm finite function, evaluated as a point expression, not an interval.

Every supplied ordinal remains present. Failed source rows remain missing;
stagnated or approximate roots retain their source status. Retained-window
ordinals are not automatically zeta ordinals. Reference-seeded acquisition and
root-domain provenance remain explicit. Working precision defaults to the
maximum declared state/root precision plus 64 bits, including adaptive root
verification precision. Extra arithmetic does not recover source accuracy.

### Projection

Projection uses the full finite support and **L2(dx)**. It is not a half-support
weighted theta-Hermite projection or an automatic implementation of a prolate
trial. Choose `unit_l2_dx` or `center_one` explicitly. A zero center under the
latter convention returns `normalization_unresolved`; an unresolved Gram pivot
returns `rank_or_precision_unresolved`, without invented coefficients.

Basis functions are not silently orthogonalized or renormalized. With exactly
two supplied components and an explicit fixed parameter b, the artifact keeps
B2 = a1 - b*a0 and b_effective = a1/a0 when defined. Its meaning is tied to those
exact component definitions and this metric. It must not be equated to a
similarly named coefficient obtained with a different measure or normalization.

### Energy, windows and stabilization

Energy is the **total retained Tau** quadratic form. It does not attribute
energy to archimedean, pole or prime terms, estimate a directional response, or
resolve an infinite ordinate tail. Window inverse moments omit everything
outside the supplied window and do not label an equilibrium band.

The stabilization producer checks a frozen finite relative-change rule. It
requires increasing N and fixed cutoff, construction precision and recorded
selection policy. Branch tracking and assembly-policy comparability are not
established by this check; `finite_rule_met` is not an N-to-infinity theorem.
A single Ultra run cannot supply a cohort and does not request stabilization.

## Live capture

New shared Ultra plans use `ccm-measurement-capture-plan-v6`. They add
`state_geometry`, `indexed_transform`, `operator_energy` and `root_band` to
the existing requests. Lower levels retain their previous default requests.
The v6 plan also requests the twenty [extended groups](EXTENDED_RESEARCH.md),
with qualified missing outcomes for unavailable inputs. Serialized v1/v2/v3/v4/v5
plans keep their old request sets and identities.

Ultra requests projection automatically. Supply a validated `ResearchInputs` JSON file through
`RetainedCcmRun::load_research_inputs`, then use
`CcmCapturePlan::ultra(...)`. Other levels can use `with_reference_projection()`. The file has
`schema_version`, `reference`, `basis` and `projection` fields; each reference
contains the definition and full ordered -N..N finite Fourier coefficients.
A reference is never inferred from a legacy target path. Requesting projection
without its inputs returns a qualified missing-source outcome; it cannot become a completed
measurement. The shared receipt keeps independent diagnostic results.

Applications must update their Toolkit dependency and execute the shared
plan through the adapter to acquire this coverage. Merely updating a lockfile
or retaining an old serialized plan does not add measurements. See
[capture integration](CAPTURE_LEVELS.md). This release does not alter application
repositories or their pinned versions.

## Cost and reuse

Transforms parallelize independent rows; total-energy capture parallelizes
matrix rows. Each row and final reduction use a fixed summation order. State
geometry uses fixed chunks and reusable recurrence scratch. Live energy capture
borrows the already admitted Tau matrix and checks factorization/sector ancestry
through exact manifest metadata, without rereading or recomputing a factorization.

Fresh geometry costs O(N times grid length); transforms O(N times row count);
energy O(N squared); projection O(N times basis count squared). Warm validation
checks source binding, scalar/shape constraints and recorded policy instead of
repeating those calculations. Hashing/decoding retained sources still has a
cost. No new campaign speedup is claimed.

Default transforms allow at most 100,000 rows and a conservative 256 MiB
estimated output budget. Explicit options enter the child identity. Import
and batch readers cap input bytes before loading. Source digests must be
explicitly admitted, and a matching cutoff/dimension alone is never sufficient.

## Historical data and remaining proposals

Use the [offline backfill guide](RESEARCH_BACKFILL.md) to add children from
retained sources. Existing numerical artifacts, old receipts and published
objects keep their original bytes and meaning. An old receipt is not rewritten
to claim it captured new diagnostics. Backfill itself does not publish data.

Directional-energy and conditional energy-allowance analyses are now registered,
alongside weighted reference projection, signed channel decomposition, weighted
tails, cluster comparison and per-k resolution. See [extended diagnostics](EXTENDED_RESEARCH.md).
These point products do not certify missing operator bounds, infinite tails
or convergence. The particular target formula remains an external input.
Automatic historical campaign discovery and application upgrades remain
separate from the additive backfill executable.

See [Ultra completeness](ULTRA_COMPLETENESS.md) for numerical coverage, reusable
reference preparation, local cohort discovery, finite source-qualified enclosures,
signed-band reconstruction and resumable diagnostic blocks.
