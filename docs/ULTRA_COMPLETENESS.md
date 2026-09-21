# Complete retained research capture

The shared Ultra v6 plan requests every implemented diagnostic group, including
reference projection and five additional completion artifacts. This release has
thirty new managed research artifact kinds in total. A single primary solve can
supply all applicable children. Input-dependent measurements still require the
identified reference, atom table, cohort or certificate that defines them.

## Numerical coverage

Capture receipts retain per-diagnostic `numerical_coverage`: retained rows,
resolved point rows, qualified rows, unresolved rows, outcome counts, expected
rows when known, and a reason/recovery hint. Completed execution and numerical
resolution are separate. Historical shapes without an assessment rule stay
unassessed. Conditional expressions do not become certificates through storage.

`ccm_capture_preflight` records source prerequisites and conservative working and
output estimates before the heavy extended kernels. Automatically prepared
inputs may be unavailable at that initial inventory; the later diagnostic and
receipt give the actual final outcome. Missing optional inputs do not invalidate
the primary measurement, and independent diagnostics continue.

## External reference preparation

An application may explicitly set `XC_RESEARCH_PREPARE_TARGET_REFERENCE=1` to
prepare research data from its configured runtime target evaluator. This records
uniform log-coordinate samples and a finite even Fourier projection through the
retained state's mode count, with a half-grid coefficient refinement measurement.
The samples support weighted projection; the finite projection supports Fourier
reference comparison. Their scope is numerical approximation, not a quadrature
certificate or an exact infinite reference. Signed jets describe that explicitly
finite Fourier projection with zero exterior, and never discard an infinite
target tail. Explicit research-input files take precedence. Prepared samples
are integrity-checked local checkpoints; target-dependent records stay private.

For sections with 1 through 512 modes, the same option prepares the remaining
finite arithmetic inputs directly from the authenticated run. The bundled
1,000-ordinate table supplies identified comparison geometry. State transform
squares supply zero weights; coefficient squares supply the full lattice measure,
including its origin mass. Inverse moments involving nonzero origin mass remain
undefined and are not printed as finite numbers.

The fixed polynomial space uses the first `N-min(N,64)` reference ordinates as
a prefix and degree `min(N,64)` for the remaining polynomial. The matrix forms
are `Gram=V^T V` and `Z=V^T A V/2`; the arithmetic remainder is `Z-head`.
This includes every contribution in the retained arithmetic matrix without
assuming RH or treating a finite critical-line sum as complete. The model solves
for its own energy and polynomial, then isolates that polynomial's roots.
Its constrained finite minimum does not certify the full ground or continuum.

Energy allowances use the current full Fourier matrix split at `|mode|=N/2`:
Outward Gershgorin bounds sharpened by verified interval LDL, the Frobenius cross-block bound, and a trial Rayleigh
value. The diagnostic checks `mu > U` before dividing. Retained matrix/source
accuracy remains a separate qualification; neither a weak block estimate nor a
model discrepancy is hidden as missing input. Explicit supplied values prevail.

Set `XC_RESEARCH_REFERENCE_FILE` to a non-executable JSON file using
`ccm::research_completion::ReferencePreparation`. The
[input schema](schemas/ccm-reference-preparation-v1.schema.json) and
[finite constant example](examples/reference-preparation.json) describe the format.
The example is a software fixture, not a proposed CCM target.

The file names C, precision, a definition digest and approximation scope. It can
provide a finite Fourier reference and basis, sampled values, weighted atoms,
explicit finite tail forms or a polynomial-basis tail recipe, and completion
inputs. Preparation binds the derived bundle to the actual eigenpair digest.
The original state-bound `XC_RESEARCH_INPUTS_FILE` interface is also supported.

For an even finite Fourier reference with a nonzero center, preparation supplies
center-one samples, projection data and analytic finite transform jets at the
available retained roots. The finite reference has zero exterior by definition;
this does not identify or discard the tail of an infinite target. For an infinite
target, supply its separately evaluated window/full/tail jets through the
state-bound input format. No target formula is included in the Toolkit.

A `tail_recipe` uses ascending polynomial coefficients in the declared atom
coordinate. The producer forms zero and lattice bilinear matrices from the
supplied weights. An omitted tail correction is explicitly a finite-only model
with an unknown infinite remainder. Degree, precision, definitions and coverage
remain in the source artifact.

## Independent checks and configuration comparisons

`ccm_consistency_analysis` compares the direct action of the exact retained
prime component with the prime action reconstructed from Tau, the pole term
and archimedean component. It retains signed energy differences and action
norm defects. Algebraic closure alone is not independent validation. Automatic
preparation reads exact parent components; missing components are reported
without recomputing them.

`ccm_root_transport_analysis` also retains fixed-source, support and total root
velocities from a matching u-flow source, including the additive defect. Branch,
coordinate, root value, derivative parameter and activation convention must
match. A shifted-secular velocity is not equated to an unshifted divided-state
identity.

`ccm_configuration_comparison` records independent state energies, overlaps,
small-state residual and forcing in the larger matrix, plus root/transform
changes when acquisition branches and ordinals agree. Different support lengths
are not subtracted as if they were the same basis. Duplicate policy coordinates
are flagged. Comparisons are measurements, not an automatically fitted rate.

Live retained runs register small exact-source metadata records under the
managed cache's `research-cohorts` directory. Discovery also inventories existing
local eigenpair manifests and authenticates their matrix ancestry. It reads at
most 64 comparison states per input bundle. `XC_RESEARCH_COHORT_DIR` selects an
explicit bounded cohort. Historical unknown quadrature policies stay labeled
unassessed; a controlled N/P/quadrature study needs independently varied states.
No new primary states or parameter grid are generated by cohort discovery.

## Signed-functional bands

`ccm_band_reconstruction` applies a reorthogonalized Stieltjes recurrence to
explicit signed atoms. It retains the recurrence, signed/absolute norms,
orthogonality corrections, sorted Jacobi roots, inverse moments and optional
scoring differences. It checks the positivity needed below the requested degree
and retains the partial recurrence if coverage, signs or precision fail.

The model must identify its coordinate, degree, atom coverage, hypotheses and
borrowed inputs, including any supplied energy. Positivity here is a numerical
measurement, not a ball certificate. Truncated ordinate tables are not promoted
to complete infinite measures. The algorithm accepts degrees up to 2048 subject
to explicit memory and output limits; it does not prove RH or a convergence law.

## Finite transform enclosures

With `arb`, `ccm_transform_enclosure` encloses the finite retained Fourier
transform and its derivative, including removable carrier limits. It retains
origin normalization, per-root raw/normalized enclosures and adaptive rectangle
segments. A zero count is emitted only if every boundary segment excludes zero
and the enclosed total argument identifies one integer. Precision or subdivision
failure retains the unresolved segments and withholds the count.

Without a source certificate, these enclosures concern the exact retained dyadic
coefficient function. They do not certify how close that state is to the true
finite ground state. An explicitly supplied matching schema-3 sector certificate
is replayed exactly. Its finite spectral gap and interval matrix residual give
a source L2 allowance, which is propagated through transform and derivative
bounds. The certificate's parity and assembly premises are inherited and stated.
A rejected certificate does not silently become a source-error allowance.

This route does not compute a new sector certificate automatically, establish an
infinite-limit error, identify zeros as zeta zeros, or assume RH.

## Resources, progress and recovery

The live extended producers accept these explicit environment policies:

| Variable | Default | Meaning |
|---|---:|---|
| `XC_RESEARCH_WORKING_BYTES` | 8589934592 | Estimated working memory cap |
| `XC_RESEARCH_OUTPUT_BYTES` | 8589934592 | Estimated output cap |
| `XC_RESEARCH_CHECKPOINT_BYTES` | 8589934592 | Per-checkpoint serialized byte cap |
| `XC_RESEARCH_ROOT_BLOCK_ROWS` | 128 | Root rows per checkpoint, 1 through 4096 |
| `XC_RESEARCH_CHECKPOINT_DIR` | managed cache/research-checkpoints | Optional checkpoint location |
| `XC_RESEARCH_CHECKPOINT_LOG` | quiet | Set `detail` for individual checkpoint I/O messages; stage heartbeats and errors remain visible by default |
| `XC_RESEARCH_BASIS_BYTES` | 8589934592 | Estimated cumulative band basis disk cap |
| `XC_RESEARCH_SUMMARY_DIR` | managed cache/research-summaries | Compact scalar exports |
| `XC_RESEARCH_COHORT_DIR` | managed cache/research-cohorts | Optional explicit cohort |

Backfill tasks can set their `ExtensionOptions` directly. These estimates limit
new diagnostic kernels; they are not an OS memory quota or a replacement for
older capture groups' existing policies. Long stages report start/end time and
30-second heartbeats. Independent root blocks run in parallel with stable output
ordering. Complement solves share one factorization. Root blocks, complement
factors/solutions, band eigensolutions, certificate replay and resolved contour
segments have content-bound local checkpoints. The band recurrence also retains per-degree state and chunked basis vectors for
restart; its resident basis cache and disk estimate are bounded. See
[large atom inputs and recurrence recovery](ATOM_RESEARCH.md).

Checkpoint payloads are streamed, hashed and sealed; changed sources or options
select different keys. Damaged checkpoints are rejected and recomputed. Local
execution checkpoints are not portable scientific certificates. The managed
artifact remains the final, source-bound publication unit.

An increased budget or a newly available reference can complete missing children
from retained primary artifacts. Existing objects and receipts are preserved.
New application builds must adopt the shared plan; changing the Toolkit alone
does not modify scripts that independently hard-code their diagnostic list.

## Additional model and resolution fields

Reconstructed bands retain inverse moments one through three over all model
roots, with coordinate and count. A zero denominator withholds the corresponding
moment. This differs from inverse moments of a retained root window.

Tail-form analysis retains model/reference signed and relative energy differences,
Cholesky pivot diagnostics, tail-on and tail-off energies, a lattice-normalized
model eigenvector, its generalized residual and signed zero/tail energy terms.
The same Gram whitening is reused. The additional finite solve and vector recovery
are included in the working-memory estimate; budget exhaustion is explicit.
Pivot ratios are diagnostics, not certified matrix condition numbers. Vector
failure retains the spectrum and counterfactual with an unresolved outcome.

Reference jets may supply `matched_root_ordinal` for an explicit join to retained
roots. Without it the producer never equates reference and root-window indices.
Adjacent supplied reference ordinals define the spacing used for displacement,
point Newton-correction and conditional-allowance ratios. These are different
quantities: a Newton step is not a certified root error. A missing mapping, root
or usable neighbor withholds its derived field. Ordinate identity remains the
caller's declaration, bound to the external input and retained source.

Band, tail-model and weighted-tail producers use extended diagnostic semantics
v4, including the [atom diagnostics](ATOM_RESEARCH.md). Resolution diagnostics
retain v3; other extended producers retain v2. Historical children remain readable and are not
rewritten; requesting the new fields creates new children through live capture
or additive backfill. The shared capture plan remains Ultra v6.
