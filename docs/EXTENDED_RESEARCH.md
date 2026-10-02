# Extended retained diagnostics

Ultra capture-plan v6 requests twenty extended diagnostic groups and retains
their source-bound numerical inputs. With the original v0.15.1 additions, the
release provides thirty managed kinds.

Results bind the exact input state and declared policy. Point measurements,
conditional expressions and qualified absence are distinct from certificates.
These products do not assume RH or assert infinite-limit convergence.

| Diagnostic | Artifact kind | Inputs |
|---|---|---|
| compactness | ccm_compactness_analysis | Eigenpair |
| weighted_reference_projection | ccm_weighted_reference_projection | Eigenpair and external target/basis samples |
| signed_transform | ccm_signed_transform_analysis | Eigenpair and external reference transform jets |
| arithmetic_energy | ccm_arithmetic_energy_analysis | Exact parent Tau and component operators |
| directional_response | ccm_directional_response_analysis | Exact parent Tau and roots; optional perturbations |
| weighted_tail | ccm_weighted_tail_analysis | Weighted atoms, coordinate, partitions and coverage |
| spectral_cluster | ccm_spectral_cluster_analysis | Cluster vectors, eigenvalues and assembly policies |
| resolution_budget | ccm_resolution_budget_analysis | Roots or external points; optional error allowances |
| energy_allowance | ccm_energy_allowance_analysis | Declared block bounds and hypothesis record |
| complex_transform | ccm_complex_transform_analysis | Eigenpair; all available roots |
| root_transport | ccm_root_transport_analysis | Tau, roots and automatically retained u-flow actions |
| operator_cluster | ccm_operator_cluster_analysis | Tau and automatic or supplied reference span |
| finite_section_transfer | ccm_finite_section_transfer | Tau and eigenpair; optional independent comparison |
| tail_operator | ccm_tail_operator_analysis | Supplied finite-zero, tail and lattice bilinear forms |
| observable_budget | ccm_observable_budget_analysis | Eigenpair and roots; optional declared source uncertainty |
| capture_preflight | ccm_capture_preflight | Source and prerequisite inventory, resource estimates |
| consistency | ccm_consistency_analysis | Direct retained prime action and reconstructed action |
| configuration_comparison | ccm_configuration_comparison | Authenticated independent C/N/P/quadrature snapshots |
| band_reconstruction | ccm_band_reconstruction | Explicit signed atoms, coverage and polynomial degree |
| transform_enclosure | ccm_transform_enclosure | Arb; optional exact-replay finite sector certificate |
| external_source | ccm_external_research_source | State-bound external numerical input bundle |

## External target boundary

The Toolkit does not contain or execute a particular research target formula.
Supply numerical data through ccm::extended_research::ExternalResearchInputs.
The target is identified by an opaque definition digest, evaluation policy,
precision, approximation scope and raw normalizer. The input must name the
exact retained eigenpair digest.

Sample nodes are x=j*log(C)/(2*intervals), including both endpoints. Reference
values use target(1)=1. Projection uses f(1)=1, 1<=u<=sqrt(C), and
du/sqrt(u)=exp(x/2)dx. The raw nonorthogonal Gram matrix, RHS, coefficients,
residual norm and explicit b/B2 convention are retained. Unresolved centers or
Gram systems remain unresolved. Sampled quadrature is not a continuous-target
error enclosure.

The [source payload schema](schemas/ccm-external-research-source-v1.schema.json)
describes the bundle in its data field. Optional input fields can be omitted;
unknown fields are rejected. The file is data-only: no expression or program
executes, and external campaign evaluator implementations are supplied independently.

For live capture use RetainedCcmRun::set_extended_research_inputs or
load_extended_research_inputs, or set XC_RESEARCH_INPUTS_FILE. The adapter loads
the file once on the first extended diagnostic. A bad optional file does not
prevent the primary calculation or source-only diagnostics.

Use capture_diagnostic_outcome when building a receipt. It records absent
sources as Missing. The shared cache capture executor also performs this
qualification. A missing-input record is not a completed measurement, although
its explanatory artifact may be retained.

## Numerical meanings

**Compactness:** signed F(0), F''(0), F''''(0), and resolved
-F''(0)/(2F(0)) use analytic Fourier moments. Exponential weighted squared norms
reuse coefficient autocorrelations and analytic integrals. These differ from
squared-density spatial moments. Neither infinite tails nor unconditional
ordinal counts are inferred.

**Signed transforms:** retain actual/reference window/full values and
derivatives, signed interior discrepancy, exterior tail, endpoint and fitted
parts, remaining residual, absolute channel sums, cancellation and closure
defects. Normalization is center_one. The reference closure uses the separately
supplied tail. External reference jets are not automatically certified.

**Arithmetic energy:** operators are sums of diagonal, symmetric dense and
weighted rank-one matrices in the same full Fourier basis. Their state
contractions are compared with Tau, retaining the combined action defect.
Completeness is a caller declaration with a measured defect. Supplied trial
coefficients yield a finite projected trial energy, not the full unprojected
trial energy. Ratios retain energy signs and exact-versus-approximated deficit
designation. Invalid deficit denominators are not divided by.

**Directional response:** for exactly even sources retain
tau=(t*log(C)/(2*pi))^2, projected resolvent direction norm and orthogonality
defect, rational root-condition residual, and x^T(Tau-EI)x. Perturbations add
signed forcing contractions. Response ratios remain conditional on the fixed
dimension, simple minimum, displacement and unshifted-root hypotheses.
Carrier points and unresolved denominators are excluded; visibly off-root
points withhold response ratios. No spectral gap or uniform bound is inferred.

**Weighted tails:** finite cumulative masses, absolute masses and three weighted
inverse moments remain separate by declared family and partition. Remaining
supplied mass is distinct from the unknown infinite tail. Measured power laws
are not omitted-tail bounds.

**Clusters:** retain nonorthogonal Gram projection, source overlaps/leakage and
cross-N overlap matrices with a common-cutoff Fourier embedding and match
margins. Input vector identities and assembly policies remain explicit.
Overlap matching does not certify branch continuity or select a ground state.

**Resolution:** retain actual transform channels, numerical resolution, point
Newton corrections and unassessed outcomes when source error or isolation is
unavailable. Source value/derivative allowances require error_normalization=unit_l2_dx
and refer to the unit_L2_dx actual
finite transform, unlike the center_one reference jets in signed_transform.
The source-error/curvature radius expression is conditional. A reference-tail
allowance is recorded separately. The conditional qualifying prefix is not a
minimum-N or kth-convergence formula.

**Energy allowance:** positive mu-U permits the point expressions H^2/(mu-U)
and H/(mu-U), with all declared bounds and hypotheses. Otherwise the result is
sufficient_bound_unavailable. This analysis is not a truncation certificate.
When the expression is available, `trial_energy_magnitude` records |U|.
For nonzero U, `allowance_to_trial_energy_magnitude` records
H^2/((mu-U)|U|), and `allowance_below_trial_energy_magnitude` is 1 only when
that ratio is below one. A ratio of at least one is explicitly described as
non-informative at the trial energy scale. Zero U leaves the ratio and flag
absent with an explanatory reason. This comparison describes scale, not a
relative eigenvalue error bound; a smaller allowance is not itself a
certificate. The request marker `allowance_interpretation=trial-energy-scale-v1`
gives these reports a distinct cache identity while preserving older reports.

## Cost, precision and historical inputs

ExtensionOptions::for_source uses source precision plus 64 bits, two exponential
weights, a 100,000-row cap and a conservative 256 MiB output estimate.
External/root precision may raise the live default. Extra arithmetic never
restores missing source digits.

Compactness reuses one autocorrelation across weights. Independent transform
and matrix rows run in parallel with fixed summation order. Warm validation
checks identities and report structure instead of repeating contractions.
Decoding, source hashing and input shape checks still have costs.

Ultra v6 requests every retained root, including directional energy and physical
root transport. The older standalone ExtensionOptions::for_source default is
still 16 rows for compatibility; historical plans retain that policy. Ultra's
full requests have distinct receipt IDs: arithmetic_energy_full,
directional_response_full and spectral_cluster_full. Artifact kinds remain
unchanged for those three groups.

The live adapter retains compact arithmetic actions, reuses u-flow derivative
actions, and supplies the selected parity-sector eigenvectors automatically.
Pole and archimedean actions use retained assembly primitives; the prime action
is reconstructed as Tau minus those signed components. Its closure is algebraic,
not an independent check of the prime sum. Signed conventions and parent
identities are retained. No target formula is inferred.

Root transport reuses the full directional artifact through the managed cache.
The finite-section scan computes every symmetric Fourier prefix of the one
retained matrix in quadratic work; it does not solve a new eigenproblem at each
prefix. Cluster feedback shares one complement factorization across its columns.
It estimates workspace before allocating the dense factorization. Ultra's new
producers default to 8 GiB each for estimated output and complement workspace;
explicit ExtensionOptions can override these budgets. If workspace is insufficient,
the compressed operator and coupling survive with a qualified partial outcome.
A retry can use the retained sources without repeating the primary solve.

Complex observations include the value, derivative, normalized transform and
resolved logarithmic derivative at every available root with imaginary offsets,
and on a closed counterclockwise rectangular contour. Sampled contour values are
not a certified zero count. The retained coefficients permit later refinement.

Finite-section transfer separates projections of one retained state from an
optional independently assembled comparison state and matrix. Observable budgets
transport a supplied unit-state L2 error to value and derivative bounds, with its
hypotheses retained. They do not infer an ordinal or source-accuracy certificate.

The fixed-basis tail model solves (finite_zero + tail) v = (E/2) lattice_gram v
and retains its full model spectrum and forms. It does not borrow the measured
ground energy to solve. The model definition, tail coverage and approximation
hypotheses must be supplied; absent forms yield Missing. This producer admits
fixed bases up to dimension 128, and rejects larger forms explicitly. Conditional tail error
expressions are not a proof of the omitted ordinate tail.

One configuration produces one primary state. Different N, C or precision values
are distinct configurations; a stabilization study needs that separate cohort.
A later target definition or missing external input adds a new child artifact.
It does not replace an older measurement or require a new primary solve when the
necessary original sources are present at sufficient precision.

[Additive backfill](RESEARCH_BACKFILL.md) uses these same producers. New targets
add children; old targets, source objects and receipts retain their original
bytes and meaning. Incomplete historical inputs remain explicit.

Old v1/v2/v3/v4/v5 plans retain their request sets. Applications must adopt the amended
Toolkit and shared v6 plan to request these groups. This release does not update
application dependencies or execute historical batches.

## Preparation and completeness

[Ultra completeness](ULTRA_COMPLETENESS.md) describes the v6 plan, the reusable
`XC_RESEARCH_REFERENCE_FILE` format, automatic source comparisons, checkpoints,
and the additional finite certified transform route. A source-independent
reference file is prepared and bound to the actual retained eigenpair. The
existing state-bound `XC_RESEARCH_INPUTS_FILE` remains available.

The current extended diagnostic request semantics are
`extended-retained-diagnostics-v2`. Earlier children are preserved under their
old identities; their raw sources remain reusable. The data envelope remains
schema 1, with optional additional inputs. A changed target, cohort, contour,
resource policy or source digest produces a different child identity.


## Directional arithmetic and root storage

Revision 11 evaluates the projected resolvent using homogeneous stored-point
formulas with directed arithmetic. Subtracting E from each diagonal entry before
matrix application preserves common scalar-shift invariance. Full perturbation
operators and supplied unit-state actions retain their distinct normalizations.
Every measured scalar has outward lower/upper companions and the row records
its arithmetic precision. Carrier, guard exhaustion and explicit row/memory
limits withhold measurements or ratios with a qualified status.

The reported root-condition threshold is a numerical proximity rule. It does
not establish an exact root, simple minimum, displacement identity or spectral
gap. Conditional tau-response ratios require those external mathematical
premises. Their arithmetic bounds exclude source construction and modeling
errors, and do not establish physical-root motion or convergence.

Retained adaptive root values decode at the payload precision. Solver target,
evaluation and verification precisions do not increase stored-point precision.
Target bits describe requested accuracy and may be below 64. Failed root rows
and original ordinals remain visible. Earlier results retain their historical
meaning and do not acquire this repair's arithmetic assurance.


## Conditional allowance arithmetic

Revision 12 retains U, b, mu and H at declared input precision and uses separate
binary scales for H and mu-U. Directed arithmetic avoids premature H^2
underflow and returns scalar estimates with outward lower/upper companions.
Nonpositive margins withhold allowances; explicit memory/guard/range limits
remain visible. Arithmetic bounds do not authenticate the external block
hypotheses, ground selection or a truncation model.

For nonzero U, scale_comparison_resolved indicates whether the strict comparison
with |U| is decided. If its value is 0, a below-scale flag of 0 means the strict
inequality was not established; it does not assert equality or greater-than.
The arithmetic allowances may still have narrow enclosures. U=0 omits all scale
ratio/comparison fields. Earlier revisions do not acquire these guarantees.


## Finite transform and resolution arithmetic

Revision 13 encloses exact-decimal cutoff construction, source normalization,
finite transforms and derivatives with directed arithmetic. Ordinates, external
errors and radii retain their declared precision; retained-root joins use the
root payload's precision. Equivalent spellings cannot invent spacing.

Resolution scalar estimates have outward lower/upper companions. Named
conditional root-distance allowances and the curvature expression use outward
upper endpoints. A successful row requires a positive slope lower bound and
an allowance upper bound below the advertised lower budget threshold. Ambiguity
is unresolved. A prefix requires all consecutive ordinals starting at 1.
The target error, curvature and isolation assumptions remain external; the
finite-state curvature formula does not certify them for a different target.
The report makes no zeta identification, infinite-tail or source-error claim.
Point-transform callers are outside the scope of these guarantees. Older
cached revisions do not acquire the new arithmetic guarantees.


## Observation and comparison arithmetic

Revision 14 preserves the owning precision of external state errors, comparison
coefficients/eigenvalues/roots, retained eigenvalues, and retained roots.
Observation reports enclose finite-support L2 transport with exact-cutoff
moments sqrt(L) and sqrt(L^3/12). Named error allowances are outward upper
endpoints; named lower margins are outward lower endpoints. Supplied state-error
and support hypotheses remain external. Missing errors leave channels unassessed.

Configuration comparisons use separately scaled vectors for overlap and form
A-EI entrywise before evaluating low/high residual squares. This preserves
common-scale invariance and small forcing next to a large canceled shift.
Signed energy/root/transform differences use original points and directed
arithmetic. Canonical exact cutoff equality controls same-basis comparisons
and duplicate-policy keys. Root comparisons retain branch/ordinal gates;
zero vectors, unavailable precision and explicit resource limits remain visible.
Joined-root fields participate in output estimates. All physical measurements
have outward enclosures and record the guard precision, capped at 4096 extra bits.

These calculations do not validate external source selection, source accuracy,
root identification, historical results or convergence. Other callers of shared
dot/norm/matrix helpers are outside their scope. Archived report revisions
retain their original guarantees.


## Finite-section and consistency arithmetic

Revision 15 evaluates every centered projection of a retained state using its
full-state coefficient norm. Residuals form A-EI before matrix action; retained
and omitted masses are summed directly. Matrix, shifted-action and coefficient
scales are separated. All prefixes retain quadratic arithmetic complexity.
Optional comparison block norms and signed defects preserve comparison-point
precision and nonzero common-scale invariance. A zero comparison state leaves
the defined block norm and prefix measurements available with a qualification.

Independent action consistency decodes both actions at their declared input
precision, scales their difference before taking norms, and encloses the signed
contraction on the original center-oriented unit source. Common normalization
and source independence remain external premises. These two diagnostics report
outward finite-expression intervals and the actual arithmetic precision, with
4096 maximum guard bits and explicit resource/precision failure behavior.

Operator-cluster reference columns now preserve their own input precision and
are normalized before numerical rank selection. Its factor checkpoint identity
changes accordingly. This repairs arbitrary column-scale dependence; it does
not certify the remaining cluster compression, solve, feedback or rank logic.
No primary eigenstate, infinite-limit, historical or scientific clearance is
implied by these repairs. Archived reports retain their original guarantees.


### Finite operator cluster arithmetic (retained observations revision 16)

The operator-cluster producer decodes each source point at its declared precision.
Reference columns are normalized before numerical rank selection. The fixed
requested-precision threshold is enclosed during twice-reorthogonalized Gram–Schmidt;
an ambiguous decision retries guard precision, then reports unresolved. This selects
a numerical subspace and does not certify the full declared span or spectral selection.

For its ideal orthonormal basis U, P=UU^T, Q=I-P, D=A-EI, and K=QDU, the producer
encloses U^T A U, K^T K, K^T[QDQ+P]^-1 K, and their effective-operator difference.
It removes E before projection and rescales D by an exact power of two before the
complement solve. Directed interval elimination requires pivots excluding zero;
unresolved or budget-limited inversion retains enclosed compression and coupling
while withholding feedback. The reported residual concerns the solution midpoint
in the scaled system; the feedback enclosure comes from interval elimination.
Each value carries outward decimal bounds and the used precision. Guards are
limited to 4,096 extra bits. Checkpoints bind original inputs and their precisions,
source identities, options, selected columns, scale, guard precision, and method.
These are finite stored-point arithmetic bounds, not source-error, continuum, or
ground-state-selection certificates.


### Complex finite-transform arithmetic (retained observations revision 17)

Complex samples use the exact decimal cutoff and original stored coefficient
and ordinate points. Their transform convention is `exp(-i*z*x)`, matching retained secular roots; the derivative is taken with respect to the same original coordinate `z`. Directed rectangle arithmetic encloses the entire finite
Fourier transform and its derivative. Small-argument series keep the leading
derivative term separate and bound the omitted tail; integer-pi carrier shifts
are removed analytically away from a removable carrier. Binary scaling protects
complex magnitudes and quotients from avoidable squared-norm range failures.
Exact Fourier orthogonality fixes F(0); exact evenness fixes the imaginary-axis
zero channels. Every measured scalar carries outward bounds and used precision.

Five offsets are retained at every root ordinal and the origin. Missing ordinates
have explicit missing rows. The 65-point closed counterclockwise contour uses
exact affine coordinate expressions in the stored maximum ordinate. These
samples do not certify a contour count or root. F/F(0) and F'/F are emitted only
when their denominators and arithmetic are resolved; a failed ratio retains
enclosed F and F' when available. Guards are limited to 4,096 extra bits, and
working/output budgets and qualified arithmetic failures remain explicit.
The bounds exclude source-construction and infinite-tail errors. Revisioned
method, coordinate, point-precision and output contracts prevent reuse of the
old unbounded complex producer's cached results as new arithmetic evidence.
