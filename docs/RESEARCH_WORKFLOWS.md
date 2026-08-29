# Research workflow entry points

Version target: `0.14.1`

These examples are stable, nonmutating entry points for the three release-required research workflows. Run them from the repository root. Successful commands write machine-readable JSON to standard output and return status 0; failures write Rust error context to standard error and return nonzero. None reads credentials, publishes cache data, or mutates a remote repository.

## CCM observation planning

The high-precision CCM API uses the managed cache without consumer setup. A full tau hit is loaded directly. On a miss, the toolkit resolves or computes compact archimedean integrals and the prime-component matrix, assembles and validates the full tau matrix, and records the dependency chain. `CcmParityPolicy::EvenSector` remains the default and reuses the established reduced even-sector operator and LU identities. `Natural` performs an unrestricted full-space solve. `AdaptiveEven` performs the original full-space inverse iteration and conditionally projects only an iterate that materially drifts away from even symmetry. The three policies have disjoint selected-state and downstream artifact identities, while sharing compatible Tau and full-factorization dependencies. Normal runs retain the selected ground eigenpair, secular source, one bundled and one-based indexed root range, and configuration evidence; evenness runs retain both natural and even-sector states plus their validation record. Author-mode publication stages the same objects to their configured public/private family shards without requiring the consuming application to implement a publishing pipeline.

Parity-sector research is an explicit operation because dense low-spectrum recovery is unnecessary overhead for an ordinary reproduction. Calling `xc_spectral::ccm::hp::analyze_sector_gap` derives and caches both parity matrices from the same exact Tau dependency, retains the requested lowest even and odd eigenpairs as `ccm_sector_spectrum` artifacts, and retains the derived `ccm_sector_gap` evidence. The odd basis matches the historical `(e_k-e_-k)/sqrt(2)` convention. The result reports GapLog, the direct eigenvalue difference and ordering, and an even-sector simplicity margin as separate values.

`xc_spectral::ccm::hp::run` is reference-free by default. It discovers a positive prefix from the finite secular source and then performs HP refinement; it does not accept a zero table. `run_independent` accepts prefix, one-based index-range, and height-window targets. Computed assurance uses pole-aware MPFR discovery directly on the full-precision secular source to obtain starting points and therefore makes no rigorous completeness claim. Certified production discovery uses exact cumulative finite-source counts, FLINT/Arb complete root isolation, and interval Newton through `certify_production_independent_ccm_roots`. `build_source` explicitly requests a source-only computation.

Halley's method is the ordinary HP refinement route; Newton remains an explicit comparison option and is never an automatic fallback. Root refinement and CCM inverse iteration each default to a 2,000-iteration ceiling. A root is converged only after meeting the requested-accuracy correction target. `RootPrecisionPolicy::FixedGuard` remains the default and preserves the historical v6/v7 identity and arithmetic exactly. Call `HighPrecConfig::with_adaptive_root_precision()` to opt into v9. Adaptive refinement begins with the established 64 MPFR guard bits and widens only the secular-root arithmetic when cancellation prevents a higher-precision replay of the exact stored point from confirming that target. The default resource ceiling is 4,096 extra root bits and the independent check uses a 64-bit wider precision; neither ceiling nor a precision-floor heuristic can substitute for the requested target. An unchanged MPFR point, a two-cycle, or 128 consecutive iterations without a smaller correction triggers another precision tier under the adaptive policy and remains stagnation at the ceiling. Slow monotone improvement may use the full iteration budget. Computed runs retain finite, ordered precision-limited or iteration-limited values with their correction, residual, iteration count, achieved digits, evaluation/verification precision, escalation count, and stopping reason; they are never relabeled as converged. A failed root with no value remains fatal, and cross-checked or certified assurance requires every root to converge. A v9 miss never uses a v6/v7 root as a warm start: this keeps path-dependent iteration evidence and payload bytes canonical across reuse, refresh, and verification. Inverse iteration separately records its configured limit, unshifted steps, convergence flag, final Rayleigh change, shifted-refinement outcome, and replayed relative Tau residual. Reaching the unshifted limit remains visible in ordinary output and run evidence even when shifted refinement successfully rescues the eigenstate. The CCM eigenstate must pass its Tau residual check before root refinement begins.

`run_indexed_seeded` and `run_indexed_seeded_via_cache` are explicit comparison/refinement APIs. Their supplied values are reference seeds, their artifacts are `ccm_root_refinement`, and they cannot satisfy an independent-discovery request. Reference datasets may be attached only through a separate post-discovery comparison artifact after the independent root window has been frozen. The lowest Tau eigenpair remains the standard CCM secular source; higher Tau eigenpairs must not be treated as later zeta zeros.

Independent and seeded refinement have the same final HP refinement and normally converge to the same finite-source roots when the computed discovery finds the complete requested sequence. Refinement is faster because it skips discovery and begins close to each root, but it is circular evidence and cannot establish that CCM found the roots. Computed independent discovery adds a pole-aware MPFR scan whose cost remains small relative to Tau construction. Certified independent discovery is substantially more expensive because it builds the exact finite numerator and proves count, completeness, existence, and uniqueness. Certification remains selective rather than the default assurance.

```powershell
cargo run -p xc-spectral --example ccm_window_plan --locked
```

The result is a serialized `ObservationPlan` for the first 100 zeros. It reports the estimated height, minimum mode reach, recommended precision, guard digits, and assumptions. This is a feasibility observation, not a proof of continuum convergence.

To replay a complete finite f64 observation previously created by `run_saved_ccm_f64_observation`, supply the saved record and a freshly captured execution fingerprint:

```powershell
cargo run -p xc-spectral --example ccm_reproduce --locked -- saved-observation.json current-fingerprint.json
```

Replay reconstructs the Weil matrix, smallest eigenpair, normalized even source, and rational spectrum from the saved configuration. It refuses an execution-fingerprint mismatch and compares the timing-independent numerical payload exactly. The saved answer is comparison evidence only and is never supplied to the computation. Success reproduces a finite binary64 observation, not an HP certificate or an infinite-dimensional conclusion.

## Finite certificate construction and verification

```powershell
cargo run -p xc-certify --example finite_certificate --locked
```

The example constructs a finite-dimensional positive-definiteness certificate, recomputes its canonical certificate identity, independently runs the bundle verifier, and emits the verified `CertificateBundle`. The JSON separates its finite claim, achieved assurance, backend, precision, matrix identity, inertia evidence, assumptions, and provenance.

## Exact Maynard–Tao lower bound

This route needs the `hp` feature and therefore a supported GNU/Linux toolchain with GMP/MPFR, such as the project's existing HP WSL environment:

```bash
cargo run -p xc-variational --example mk_constant --features hp --locked
```

The result is a serialized `MkRayleighCertificate` for the constant degree-zero candidate at `k = 2`. Its exact numerator, denominator, and quotient establish a rigorous lower bound within the declared finite polynomial search space. For the larger symmetric discovery-plus-exact-evaluation example, run:

```bash
cargo run -p xc-variational --example mk_symmetric --features hp --locked
```

The exploratory eigensolver in `mk_symmetric` proposes coefficients, but the reported lower bound is recomputed after exact rationalization and does not inherit assurance from floating-point discovery.

## Target-distance measurement

The CCM target-distance program measures
`d(N, lambda) = integral_1^lambda |f(u) - target(u)| u^(-alpha) du`, where `f` is
the even CCM ground-state eigenfunction normalized to `f(1) = 1` and the
normalized target profile is supplied privately at runtime. Set
`XC_TARGET_SPEC_FILE` to the JSON specification path before target-dependent
work. The public toolkit retains only the specification's SHA-256 digest. The program's objective
takes the limits in a fixed order: stabilize in `N` at fixed `lambda` first,
then study the stabilized distance as `lambda` grows.

`xc_spectral::distance::hp::ccm_distance_to_target_hp` performs one such
measurement end to end. The eigenvector resolves through the ordinary
reuse-first cache routes, so sweeps over already-cached configurations reuse
the expensive spectral artifacts. For measurement campaigns over cached
states, run with `XC_CACHE_MODE=require_reuse` so a mistyped configuration
surfaces as an immediate miss instead of a silent cold recomputation, and keep
publication disabled when the campaign is analysis-only. In an author
publication run, the same mode stages each validated reuse hit and its complete
dependency closure without recomputing it; the remote destination is verified
before any Git mutation, so already-published identities remain no-ops.

The state source is the same canonical managed `ccm_weil_eigenpair` used by
the claim pipeline. Its content digest is part of every profile/distance
semantic identity and the exact eigenpair is retained as a dependency. This is
not interchangeable with a separately approximated sector-spectrum midpoint:
near the precision floor that midpoint can have the wrong sign while still
lying inside its absolute Sturm tolerance. Legacy unbound distance identities
are therefore not reused by the corrected route.

The measurement path is validated against an independent implementation. At
`N = 150`, 500 decimal digits, matrix quadrature `Q = 600`, Gauss-Legendre
`Q = 600`, and `alpha = 1/2`, `ccm_distance_to_target_hp` reproduces every
digit of the values reported independently for `c = 5, 13, 17`:

```text
c = 5    0.0269735313324961574...
c = 13   0.00988258128277552575...
c = 17   0.00750657880432477674...
```

Only the eigenfunction is an approximation here; the agreement is to the full
precision the comparison values were quoted at.

Report every distance together with its recorded convention: integration
rule, grid variable, resolution, `alpha`, and precision. Both the uniform-grid
family and Gauss--Legendre are available, so an external implementation's rule
can be reproduced exactly instead of approximated; neither family is the
toolkit's default authority. Two values computed
under different conventions differ at finite step even for identical
eigenfunctions; the `target_distance` example demonstrates the spread for the
runtime-supplied target.

Measurements are retained only when retention is requested. Distance capture is
absent from the `claim`, `research`, and `gap` capture levels, is requested
explicitly through `CcmDistanceCaptureOptions`, and is included by
`CcmResearchCaptureOptions::maximum`. Maximum capture retains the eigenfunction
profile, target distance, `ccm_distance_resolution_evidence`, and
`ccm_target_residual_analysis`, which are properties of one configuration. The
resolution evidence records effective bandwidth and discarded-tail diagnostics
at thresholds `1e-15`, `1e-30`, and `1e-45`, plus same-rule Q/2Q refinement for
every uniform-grid rule. It continues to 4Q only when the Q/2Q relative
difference exceeds `1e-8`; Gauss--Legendre stays the independent-family
cross-check in the target-distance artifact and is not doubled. The residual
analysis records signed and one-sided residual mass under those same rules and
uses the already-retained profile samples for signs, extrema, and strict
sign-change brackets. It does not change the integration rule or perform
piecewise integration. An ordinary explicit distance capture remains profile
plus distance unless `with_resolution_evidence()` and/or
`with_residual_analysis()` is selected. It
does not retain `ccm_discretization_distance`, and cannot: `D_alpha` compares
two mode cutoffs, while a run resolves one. Retaining it is a separate explicit
call, `capture_ccm_discretization_distance_via_cache`, which takes both
configurations. Because `D_alpha` is symmetric, the cutoff pair is stored in
ascending order so the measurement has one artifact identity rather than two.

Target distance, resolution evidence, residual analysis, and deviation
decomposition are private-only artifact kinds. Managed publication routes them
to the private leg: under `Both` they are withheld from the public destination
while public-eligible kinds publish to both, and an explicit `Public`-only
request fails only when nothing staged is public-eligible. Public bootstrap
layers ignore them. Eigenfunction profiles and inter-discretization distances
contain no runtime target definition and remain public-eligible.

The same maximum route retains `ccm_root_conditioning_analysis` separately in
`ccm-evidence`. Its per-root records bind to the exact root-range and secular-
source parents and contain the signed secular derivative, reciprocal derivative,
condition magnitude, absolute secular-term sum, neighboring poles, nearest-pole
distance, normalized isolation margin, and normalized position in an enclosing
pole interval. This is inexpensive point-source analysis rather than
certification. On a reuse-first rerun, a missing child is populated from the
retained parents; reuse replays the term sum and derivative from the exact
secular source before accepting the child.

`ccm_prime_power_response_analysis` is also retained in `ccm-evidence`, but only
when `capture_prime_power_response` is explicitly true (the builder
`.with_prime_power_response()` sets it). It is intentionally excluded from
`maximum` because it factors one reduced-even-sector bordered
eigenstate-response system and solves one right-hand side per active prime
power. Before doing so, response schema v2 resolves indexed HP Sturm enclosures
for the first two even-sector eigenvalues, requires their same-sector gap to be
positive, verifies that the selected state belongs to the isolated lowest
branch with residual small relative to that gap, and retains that evidence in
the payload. For each event it retains the exact
von Mangoldt weight, current reduced position, analytic `dQ/du` coefficient,
edge-jump coefficient, Hellmann--Feynman eigenvalue response, projected forcing
norm, complete L2 eigenvector response, CCM normalization response, solve
residual, and the implicit response of every retained root. The response is the
event's additive prime contribution at fixed observation geometry; nonprime and
pole-motion derivatives are outside the artifact. When the observation cutoff
equals the event power, it reduces to the exact rank-one right-minus-left jump.
The Tau, selected eigenpair, root range, secular source, even-sector matrix, and
indexed even-sector eigenvalues are exact parents, so a later opt-in run can
create a missing child from retained parents without changing their semantic
keys. It still performs the new bordered factorization and event solves. An
unresolved same-sector crossing stops capture with
`unresolved_near_crossing`; a small solve residual cannot override the missing
simplicity precondition. Natural and adaptive-even state routes are rejected.
The v2 response identity cannot reuse an unguarded v1 payload.

`ccm_deviation_decomposition` is an explicit opt-in through
`capture_deviation_decomposition` or `.with_deviation_decomposition()` on
`CcmDistanceCaptureOptions`, and is excluded from every named capture level.
It records the amplitude of the auxiliary profile supplied by the same private
runtime specification, together with the deviation, auxiliary-profile, and
residual norms and the relative residual left after the projection. The solved
auxiliary parameter and target-definition digest travel in the payload, so the
amplitudes remain identity-bound and reproducible when the private definition
is available.

Two inner products are defensible readings of the distance functional's
`u^(-1/2)` weight -- applied to each factor, or once to the product -- and
they are not equivalent. Both are always retained, each labeled with the
metric that produced it, because an amplitude without its metric is not a
recoverable number. The projection integrals use the retained profile grid
rather than an independent quadrature.

The amplitude can pass through zero at a cutoff-dependent `N`, where the
deviation is carried by other components instead. Such a configuration is
recorded, not rejected: unlike first-order eigenpair perturbation, a
projection onto a fixed basis stays well defined at a crossing, and those are
precisely the configurations that locate it. The artifact reads only the
retained canonical-eigenpair-bound profile, so a missing child can be
backfilled without repeating an eigensolve. A legacy unbound profile is not
accepted as that parent. It states amplitudes and residuals only; no law
relating them across configurations is computed or implied.

`ccm_u_flow_response_analysis` is a separate explicit opt-in through
`capture_u_flow_response` or `.with_u_flow_response()` and is likewise excluded
from `maximum`. It records the complete first derivative with respect to
`u=log(lambda_squared)` under the toolkit's right-continuous active-prime-set
convention. Four channels are retained: the Tau pole derivative, the signed Tau
archimedean derivative, the aggregate active-prime derivative, and their total.
For each channel the artifact stores the analytic matrix action on the selected
state, its norm, Hellmann--Feynman eigenvalue response, projected forcing,
complete L2 eigenvector response, CCM normalization response, bordered-solve
residual, and fixed-secular-pole response of every retained root. Separate
per-root records isolate secular-pole motion
`d(2*pi*n/u)/du=-2*pi*n/u^2` and give the final
combined moving-pole response. This makes prime/nonprime reinforcement and
cancellation directly researchable while retaining the full selected-state
transport needed by convergence studies. The exact Tau, eigenpair, root range,
secular source, even-sector matrix, and indexed even-sector eigenvalues are its
parents; a missing child can be added later without changing those identities,
although analytic archimedean differentiation and four response solves still
run. The same schema-v2 isolation evidence, reduced-even-sector solve,
non-even-route rejection, explicit `unresolved_near_crossing` failure, and v1
cache separation described for prime-power response apply here.

Finite parity/order certification is a third explicit opt-in and is excluded
from `maximum` because it performs a cutoff-free Arb interval assembly and
exact-rational shifted-inertia replay. Compose it with any sector analysis that
retains at least two eigenpairs; `maximum` already supplies that numerical
guide layer:

```rust
use xc_spectral::ccm::hp::CcmResearchCaptureOptions;
use xc_spectral::ccm::sector_gap_certificate::CcmSectorGapCertificationOptions;

let capture = CcmResearchCaptureOptions::maximum(8)
    .with_sector_gap_certification(CcmSectorGapCertificationOptions::default());
```

This creates `ccm_sector_gap_certificate` in `ccm-evidence`. The ordinary even
and odd spectra remain exact manifest parents and are retained as research
guides. The certificate first proves the raw full cutoff-free matrix's inertia
with interval LDLT, so its positive-definiteness conclusion needs no parity or
centrosymmetry assumption. It then independently derives native high-precision
guides from a reflection-orbit canonicalization and proves enclosures for the
lowest two even eigenvalues and lowest odd eigenvalue with exact shifted
inertia. The reported parity (`even`, `odd`, or `unresolved`), sector ordering,
and sector simplicity explicitly depend on the premise that the exact
closed-form CCM matrix is centrosymmetric. Offline replay binds the conditional
parity matrix to the raw inertia-certified Tau records and trusts neither guide
family.

A prior maximum run has the numerical parents but not the interval certificate.
A later opt-in run reuses those parents and creates the missing child. On a
certificate cache hit, exact replay uses the stored payload and does not
reassemble Tau. On a miss, the cutoff-free matrix and proof must be computed;
this cannot be backfilled by merely copying manifest metadata. The claim is
only for the recorded finite `(c, N)` matrix and does not establish a continuum
parity theorem or CCM convergence.

A requested capture writes its selected `ccm-distance` artifacts through the
managed cache and fails rather than silently discarding them when no cache
context is available. Gauss--Legendre target-distance capture also resolves
its nodes and weights through the managed `gauss_legendre_rule` family.
Configurations with the same order and working precision therefore reuse the
same exact table; ordinary non-capture measurements remain cache-off. Because
the convention is part of the semantic key, running the same configuration
under a second scheme adds a distinct artifact instead of replacing the first.
Published canonical-eigenpair-bound profiles are the useful unit for
downstream work: norms, inter-discretization distances, and pointwise ordering
checks can all be recomputed from a retained profile without access to the
original eigensolve. A current profile plus target distance can backfill any of
the three diagnostic artifacts without rerunning the eigensolve. Legacy
unbound identities remain historical and are not accepted as current parents.
Residual backfill still
performs the requested signed quadrature against the retained coefficients.
During a fresh maximum capture, resolution evidence reuses Q eigenfunction
values in 2Q and 2Q values in conditional 4Q only when the MPFR abscissae are
exactly identical. This applies to nested left, right, and trapezoid grids in
either `u` or `log(u)`; midpoint grids are evaluated independently.

## Library-level normal use

Every production workspace crate has a compiled normal-use target recorded in `EXAMPLE_INVENTORY.json`. The smaller library examples can be run independently:

```powershell
cargo run -p xc-numerics --example quadrature --locked
cargo run -p xc-operator --example matrix_action --locked
cargo run -p xc-root --example bracketed_root --locked
cargo run -p xc-solver --example plan --locked
cargo run -p xc-core --example publication_export --locked
cargo run -p xc-zeta --example reference_zeros --locked
```

The cache examples separately demonstrate overlay resolution, permission failure, publication planning, and verification without executing remote mutation. `xc` is a binary-first crate, so its compiled normal-use target is the binary rather than an artificial example wrapper.

## Validation commands

The bounded default examples are exercised directly with:

```powershell
cargo run -p xc-spectral --example ccm_window_plan --locked
cargo run -p xc-certify --example finite_certificate --locked
```

The HP examples are compiled and run on a supported GNU/Linux HP environment. Windows MSVC is intentionally not presented as an HP route because `gmp-mpfr-sys` does not support that target.
