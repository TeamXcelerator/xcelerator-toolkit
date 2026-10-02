# Numerical compatibility and existing artifacts

v0.15.0 separates corrected numerical routes from historical results through
versioned semantic identities. An upgrade does not rewrite stored artifacts or
establish the accuracy of an earlier experiment. The changes below apply to
the library; applications must update their dependencies and rebuild to use them.

## v0.16.0 clean slate

Toolkit 0.16.0 does not reuse any artifact produced by an earlier release.
Every managed family and kind has producer and reader floor 0.16.0: earlier
entries in local caches or shards are inadmissible hits and are recomputed.
The artifact repositories were restarted for this release. Results reported
from earlier releases are assessed by recomputing them with 0.16.0; the
guidance below for comparing earlier results remains applicable.

## Changes after v0.15.2

Adaptive root results must replay their directed local root witness at the
verification precision recorded in their validated precision schedule. The
fixed-precision path still uses its fixed guard. Stored correction diagnostics
must equal the verified correction rounded upward to storage precision. Root
existence, precision ceilings, and correction thresholds remain enforced.

Automatic preparation now streams each exact head coefficient into the same
directed moment sums. Successful point values and head/tail semantics are
unchanged; the implementation no longer retains every large exact coefficient
simultaneously. Per-coefficient and input-coordinate resource limits remain.
These source changes follow the published v0.15.2 tag.

## Changes in v0.15.2

This release repairs finite arithmetic, input-domain, result-acceptance, and
cache-identity defects.
Old results remain historical evidence; upgrading does not validate earlier
scientific conclusions or rewrite retained payloads.

Affected matrix, integral, prime-component, root, conditioning, response, and
research requests have new arithmetic identities. Current production of those
artifacts requires reader v0.15.2. Standalone Tau uses schema 3 and its arithmetic
stamp; affected point-root certificates use schema 2. Legacy response repair
preserves the original pole and derivative conventions. It does not silently
convert an old result into a current one.

HP point-root geometry is explicitly staged: `L = RN_p(log(C))`,
`spacing = RN_p(2*pi/L)`, and `pole_k = RN_p(k*spacing)`. Logarithms start from the
exact declared decimal cutoff. A floating-point caller has already rounded its
input; `CcmParams` uses that input's canonical decimal identity. Use the exact
cutoff interface when the original decimal, including a very small distance from
one, must be preserved. Root refinement, conditioning, and current point-source
certificates share the same stored geometry.

Matrix components use their documented stored-length and quadrature-point
stages. Exact-cutoff aggregate research forms have a separate contract; a prime
edge that is exactly zero in that contract need not be exactly zero in a
rounded-length matrix model. Directed rounding and exact finite sums establish
point-stage results, not quadrature truncation or continuum accuracy. The
separate cutoff-free Arb assembly/certificate route encloses its stated source
model.

Automatic polynomial preparation uses convention v4 and an exact stored
head-plus-tail decomposition. Additional representation bits can preserve that
identity without increasing source accuracy. Finite tail-model products retain
cancellation. The conditional energy perturbation bound is rounded upward from
`2*epsilon*trace(G^-1)` for an exactly positive-definite stored Gram matrix. It
assumes the supplied tail spectral-norm error; it excludes numerical eigensolve
and source-assembly errors. Approximate eigensolve outputs remain point diagnostics.

Unresolved rounding, unsupported domains, and resource or exponent exhaustion
return errors where the contract requires a reliable result. Passing tests or
source inspection is not a universal proof that every library input is defect-free.

## Additive changes in v0.15.1

Ultra capture-plan v6 adds completion diagnostics; serialized v1-v5 plans keep
their original request sets. New extended diagnostics use v2 request semantics,
so earlier child measurements are preserved rather than silently reinterpreted.
That capture-plan addition preserves primary source identities. Separate
mathematical corrections and arithmetic identity changes are described below.

Capture receipts with the optional numerical coverage summary require reader
v0.15.1. Historical receipts remain readable without that summary; unknown
legacy outcome shapes are reported as unassessed, not numerically resolved.
Changing an external target or improving a child producer calls for additive
backfill from exact retained sources. It does not establish corruption in those
sources or require deleting previous target measurements.

See [Ultra completeness](ULTRA_COMPLETENESS.md) and [backfill](RESEARCH_BACKFILL.md).

### Gaussian target-series termination

Generic series sum finite partial sums only after bounding their omitted
absolute monomial tails. Each base/parameter component must separately meet
its relative budget. A polynomial zero cannot establish convergence; exhausting
`maximum_terms` returns an error in both backends. Binary64 base normalization
cancels constant scales before evaluation and uses a normalization-aware
absolute budget when the normalized value approaches underflow.

This corrects HP scale-sensitive early termination, HP silent exhaustion, and
binary64 premature termination at a small or zero polynomial term. Gaussian
target-definition digests now bind `gaussian-series-log-range-checked-v3`
(the earlier tail-only repair used `gaussian-series-relative-geometric-tail-v2`).
External-only targets retain their existing protocol identity; external targets
with Gaussian auxiliary series also bind the new summation semantics. Old
target-dependent cache results remain historical and cannot satisfy requests
using the new definition digest. An unchanged descriptor file therefore has a
new evaluation identity. No original payload is silently relabeled.

The tail comparison is computed floating-point arithmetic, not a directed
interval certificate for the complete evaluator. Cancellation and source
accuracy remain separate concerns. Independent Arb checks verify selected
values, parameters and unequal scales using incomplete-gamma integral tail
bounds. Historical distance results are not revalidated by this change.

### Stable cutoff-flow derivative identity

Stable cutoff-flow derivatives introduced `ccm-u-flow-response-v0.15.1-v4`.
The v0.15.1 production route preserved that calculation under the source-isolation v5
identity described below, with reader floor 0.15.1. The stable gamma difference used in Archimedean derivatives
changes low-order action bits even when both formulas are accurate. Reusing
the old v3 identity would make its required exact numerical replay fail.
The new identity requires fresh derivative actions, bordered tangents and
dependent responses. Existing matrix/state inputs keep their own identities
and require their separate validation; no old numerical payload is relabeled.

The offline root-only repair continues to map legacy v2 responses to v3.
It preserves their derivative arithmetic and cannot upgrade them to v4.
Already-v4 sources may be checked by exact retained replay without changing
their identity. An arithmetic identity change alone does not establish that
old numerical values were inaccurate at a scientifically relevant scale.

### Finite binary64 root ranges

The generic `xc-root` bisection, safeguarded Newton fallback and pole-aware
discovery now support finite brackets whose full width overflows binary64.
Previously, a bracket such as `[-1e308, 1e308]` could produce an infinite
midpoint labeled refined, or be skipped by discovery. Midpoint, width and
subdivision arithmetic now avoid that overflow. Discovery rejects nonfinite
function values, and optional derivative diagnostics omit nonfinite values.
These point approximations require the stated continuity/domain assumptions;
they are not certified root enclosures or a completeness proof. The HP CCM
root algorithms are unchanged by this generic binary64 correction.

## Changes in v0.15.0

| Product | Current behavior | Existing artifacts |
|---|---|---|
| Quadrature, components, Tau matrices, eigenstates and roots | Existing kind-wide compatibility floors remain; input and reader validation is stronger | Compatible, valid entries remain eligible for reuse |
| Sector tridiagonal T, transform Q and selected spectrum | Scaled opposite-sign Householder reduction and interleaved-pivot solves use new semantic identities | Earlier identities remain available for historical replay and are not reused as the corrected products |
| Managed prolate FD spectrum | Schema 2 keys exact cutoff text, source precision, grid and working precision | Historical integer-key spectra are not substituted for the new request |
| Eigenfunction profile and target distance | Declared-precision round-trip decimal exports use new semantic identities | Earlier payloads remain historical objects; current production requests the new identity |
| Resolution, residual and decomposition diagnostics | Children use the actual retained profile/distance values and bind exact parent dependencies | Children of earlier parent semantics are not reused under the new identity |
| Prefix analysis | Legacy two-moment requests retain v2; extended policies use v3 for third moments or omitted cancellation | New Ultra plans request v3; old resolved plans keep their policy and compatible v2 children remain reusable |
| Capture receipts and hypothesis evaluations | New private-only managed kinds, with exact outcome/observation identities and floor 0.15.0 | Existing local reports are preserved; new records are produced through the capture or evaluation APIs |
| Retained reduction check | New managed `ccm_retained_reduction_check` kind, with reader/producer floor 0.15.0 | A diagnostic child is added without replacing its matrix or eigenstates |
| Cutoff-free sector certificates | Corrected zero-mode endpoint treatment has separately versioned semantics | Earlier certificate meanings are not promoted to the corrected meaning |
| Prime-power and cutoff-flow root responses | v3 semantics evaluate root motion directly from the L2 state and tangent | Earlier response children and dependent receipts require new identities; their matrix, eigenstate, root and prefix parents remain reusable |

Both `Banded` and `BandedInterleaved` now select the corrected interleaved-pivot
algorithm and the same current semantic identity. The ordinary
`tridiag_lu_solve_hp` entry point is corrected too. Historical defective solve
arithmetic, like historical same-sign Householder arithmetic, requires a pinned
earlier revision; it is not exposed as a current numerical route.

Other corrections reject nonfinite HP root evaluations, detect brackets that
collapse at the working precision, handle exact safeguarded-Newton endpoint
roots, and require larger seeded-window reuse to match the supplied reference
values. A dataset label alone does not authenticate a seed sequence.

## Assessing earlier results

Distinguish four questions when reviewing a stored result:

1. **Byte integrity:** do the manifest, transport package and logical payload
   match their recorded digests? A checksum detects changed bytes, not a
   numerical algorithm defect.
2. **Identity and provenance:** do exact parameters, solver semantics,
   precision and dependency closure match the intended experiment?
3. **Numerical accuracy:** do original-operator residuals, independent checks
   and controlled precision/quadrature studies support the required accuracy?
4. **Capture completeness:** were all requested measurements and their outcomes
   retained? Missing diagnostics do not by themselves make a source corrupt.

The legacy later-pivot solve, cancellation-sensitive reflector, prolate key
aliasing and precision-losing decimal exports were real defects. Their impact
on a particular result depends on its route, inputs and retained provenance.
Do not infer that every old result is affected, or that a result is accurate
merely because the current reader accepts it. Higher analysis precision cannot
recover construction digits absent from the stored source.

Stable dense eigenvalues are point estimates, with no promised relative error
for eigenvalues tiny compared with the matrix scale. A small positive value
does not certify its sign or accuracy. Check arithmetic sensitivity on the
same retained source separately from comparisons of newly assembled sources;
use the [prefix precision guidance](PREFIX_CONVERGENCE.md#precision-planning-and-observed-cancellation)
to plan those studies. Effective inverse rank and cancellation are diagnostics,
not automatic proofs of sufficient precision.

Artifacts carrying the known cutoff-free zero-mode certificate semantics, their
recorded descendants, and artifacts whose identities change under the v0.15.0
sector/distance routes require new calculation. Include every relevant
dependency shard, including rollover shards and public parents of private
children, when determining that closure; missing dependency manifests can
undercount descendants. Absence from that closure does not make an artifact a
validated numerical result.

There is no universal per-artifact cost: a sector reduction is cubic in its
dimension, while a cached scalar child can be cheap. Use representative
dimension/precision measurements and the full dependency closure rather than
multiplying every artifact by one timing. Historical readers are retained as
explicit legacy numerical APIs and pinned revisions; new production does not
alias earlier sector products into corrected identities.

For an affected scientific observable, preserve the original artifact and
compare it with a newly identified computation. Record signed differences,
original-matrix residuals, precision, construction quadrature and source
digests. Root, branch, positivity and continuum claims require their own
evidence; a small factorization or reduction residual does not establish them.

## Reuse and diagnostic backfill

### Source-matrix response isolation

Current responses use `ccm-prime-power-response-v0.15.1-v4` and
`ccm-u-flow-response-v0.15.1-v5`. Both require the stored even-sector source
matrix, tridiagonal, full Householder transform, and the first two indexed
Sturm enclosures. The transform is an explicit cache dependency and identity
input. Every Gram product and every similarity residual is checked with
directed arithmetic. Checks are homogeneous under common binary scaling;
there is no unit floor or sampled-column substitute.

If eta bounds ||Q^T Q-I||_infinity <= 1/2, rho bounds ||AQ-QT||_infinity,
and tau bounds ||T||_infinity, the polar decomposition Q=U H gives
||U^T A U-T||_2 <= delta = 2*n*rho + 4*tau*eta. Weyl's inequality transfers
each ordered tridiagonal enclosure to the actual stored symmetric source
matrix by widening both endpoints by delta. The response's optional
`source_matrix_eigenvalue_allowance` field records that upper bound. New
production always writes it; historical payloads may lack it. The original
Sturm counts remain attached to the original tridiagonal intervals.

The legacy field `sturm_gap_lower_bound` now bounds the source-matrix gap
after widening. `selected_state_absolute_residual` bounds
||Av-lambda*v||_2/||v||_2, in eigenvalue units, for the actual stored restricted
sector vector. `selected_state_relative_residual` bounds
||Av-lambda*v||_infinity/||v||_infinity; its historical name does not imply
division by an operator norm. The ratio field is an upper bound using the
source-gap lower bound. Bound calculations and decimal endpoints are directed;
rounded vectors are not assumed to have exact unit norm.

These checks require finite symmetric stored matrices and 64..=1,000,000 bits,
and reject unsupported exponent spans or numerical buffers beyond 8 GiB.
Full transform validation takes O(n^3) work. These finite matrix bounds do not
certify continuum discretization, full natural-sector ground selection,
derivative actions/tangents, or historical response artifacts. The retained
root-only repair cannot establish this new isolation identity: source matrices
and transforms must be resolved and revalidated through current production.

### Concentration quadrature

The sinc kernel uses its removable limit only at exact zero. The former fixed
1e-40 branch erased corrections resolvable at high precision. Public
concentration assembly now checks dimensions, precision, C>1, finite bandwidth
Omega>=0, numerical allocation budget, quadrature success and finite outputs.
Zero bandwidth returns the exact zero matrix. The result is still a computed
quadrature matrix; its continuum operator's [0,1] spectrum and truncation error
are not automatically certified by these input and arithmetic checks.

### Root-response normalization

The root-normalization correction introduced
`ccm-prime-power-response-v0.15.0-v3` and
`ccm-u-flow-response-v0.15.0-v3`. Current cutoff-flow production additionally
uses the stable-derivative calculation and the source-isolation v5 identity described above. Payload schema 2 and the artifact kinds are
unchanged. Earlier v2 computations differentiated the CCM-normalized weights
before evaluating the secular derivative. A tiny eigenvector boundary sum can
make that normalization derivative enormous. Its contribution vanishes at an
exact secular zero, but finite arithmetic can leave a spurious root velocity,
including an incorrect sign or exponent. A successful bordered residual check
or a complete capture receipt does not detect that cancellation.

The corrected computation evaluates root motion with the L2 state and its
retained tangent, including pole motion in the same sum for total cutoff flow.
The CCM normalization-scale derivative remains separately available. This
removes the avoidable gauge cancellation; it does not certify an underresolved
source or guarantee relative accuracy for every very small response.

Preserve historical response artifacts. Correct response children and their
receipts under the new identities before relying on their root velocities.
The [retained response repair tool](CCM_RESPONSE_REPAIR.md) can do this from
schema-2 tangents and exact original eigenpairs without rerunning claims or
repeating the expensive bordered solves. Missing or incompatible retained
inputs are explicit errors, never a reason to silently replace a source.
The correction does not change captured eigenvalues, roots, eigenvector
tangents, prefix moments or matrix assembly. It requires no new shard layout
or artifact kind. Older response identities and their descendants require
recomputation.

### Published research record reuse

Managed receipts and evaluations retain their existing identities and payloads.
Their shard adapters carry dependency identities in a canonical manifest and
leave the local key-based list empty. Earlier v0.15.0 builds incorrectly compared
that empty list with the record's sources and could report
`managed research dependency closure mismatch` for a valid published record.

The corrected reader validates the canonical graph against each recorded source's
semantic identity, content digest and quality using metadata only. Missing,
mismatched and insufficient-quality sources remain errors. Update the executable
to use this correction; flushing caches, republishing receipts or repeating
numerical calculations is unnecessary for this reader defect. A failed invocation
remains an unsuccessful attempt even when its retained measurements are valid.

### Publication aliases

An artifact can be reached through a local draft and published private/public
manifests that differ only in their publication visibility marker. Earlier
v0.15.0 builds rejected that valid combination as an ambiguous publication
closure, even after every requested diagnostic and receipt had completed.

The publisher now resolves these exact aliases and publishes one equivalent
artifact per destination. It canonicalizes dependency ordering after remapping
and retains distinct historical dependency closures, all evidence digests and
the strongest assurance requirement. It still rejects an unbound transport or
a missing dependency. This changes publication bookkeeping, not numerical
payloads or semantic versions. No cache flush or artifact repair is needed.
A failed invocation still requires a successful retry before it can be reported
as complete; its earlier logs remain unsuccessful historical evidence.

### Backfilling requested children

A current application can reuse an accepted source and compute missing
diagnostic children when it explicitly requests them and its child cache mode
permits computation. `RequireReuse` cannot create a missing child. Use separate
source and diagnostic contexts when sources must be reused but new child work
is allowed. The retained-source APIs never acquire a replacement source.

New semantic identities can require new computations even when older artifacts
exist. Reader rejection must be handled explicitly; a numerical validation
failure is not a promise of automatic repair. See [capture levels](CAPTURE_LEVELS.md)
for the two capture phases and receipt requirements, and [cache behavior](CACHE_SCHEMA.md)
for publication and dependency validation.

## Validation cost and interpretation

Ordinary Gauss--Legendre reads use O(n) structural and selected-moment screens.
`check_gauss_legendre_rule_hp` explicitly checks all Legendre residuals and
derivative-weight identities in O(n²), under an order budget. LU validation
uses three deterministic solve probes in O(d²); it is not a full PA=LU residual
or a condition-number bound.

Managed retained-reduction reports validate identity, exact dependencies,
finite values, shape and verdict consistency on a warm hit. Refresh/Verify
performs numerical replay. The initial check is cubic and requires an explicit
dimension and working-precision budget. See [research evidence](RESEARCH_EVIDENCE.md)
for its residual definitions.

### Extended prefix policies

New Ultra plans capture the third inverse moment under prefix semantics v10;
the two-moment policy uses v8. The optional cancellation flag
remains part of the explicit diagnostic policy. Earlier phase scheduling kept
v2/v3 arithmetic unchanged; the present export repair requires new child
identities. Neither policy changes source, root, eigenstate or retained-reduction
identities. Old serialized plans keep their requested diagnostics. Older prefix
children require fresh calculation; this does not establish which historical
payloads exhibit the defects described below.

### Capability-dependent retained diagnostics

Transform-enclosure and band-reconstruction requests bind the availability of
the Arb backend into their semantic cache identity. A build without Arb can
produce a qualified feature-required absence; that record cannot satisfy a
later Arb-enabled request. Legacy requests without a capability declaration
also require a fresh calculation under the new identity. Historical bytes are
preserved. This identity repair does not invalidate an independently verified
finite enclosure or claim that a missing result contained a false certificate.

### Grid and state-selection repairs

Uniform-grid HP integration validates Float bounds directly, constructs exact
integer/half-integer offsets and divides by the full usize cell count. It no
longer rejects valid HP ranges through binary64 conversion or truncates counts
above u32. Both backends reject nonfinite samples and arithmetic. These are
computed finite quadrature sums; discretization and callback accuracy are separate.

Native tridiagonal Sturm counts now use exact integer arithmetic on the stored
binary64 dyadics. An epsilon pivot floor previously changed inertia. Bisection
uses outward initial bounds, exact counts, safe midpoints and an exact endpoint
width comparison. Unrepresentable tolerances fail explicitly.

HP tridiagonal QR rejects nonfinite inputs and nonrepresentable intermediates,
and initializes working copies at the requested precision. HP Sturm counts now
use directed determinant intervals with bounded guard-precision escalation.
Finite cancellation formerly rounded a negative determinant to zero. An interval
must establish each sign or exact zero; unresolved cases return errors. Selected
HP eigenvalue endpoints use these counts and an upward-rounded width test.
These finite-matrix enclosures do not establish matrix-assembly accuracy,
eigenvector accuracy, continuum claims or eigenvalue correspondence to zeta zeros.

Ordinary same-precision arithmetic and identities are preserved where the new
guards are inactive and previous sign decisions were correct. Previously wrong
or unresolved requests can now change or fail. No historical payload is relabeled
or cleared by this source repair. Historical callers require their own revalidation.

### Interval, vector, restriction, and operator contracts

Invalid MPFR interval arithmetic is no longer usable as evidence for a sign,
subset, or empty Newton intersection. Fallible APIs validate finite ordered
bounds and compatible precision. Infallible arithmetic propagates an invalid
sentinel that proof-producing consumers reject; constructing a nonfinite point
panics. Directed precision conversion encloses the original stored Float.
Finite secular certificates remain certificates of the declared stored poles
and residues; integer CCM pole construction does not enclose exact pi/log(c).

Native Gauss-Legendre has checked rule/evaluation APIs, safe affine parameters,
and explicit invalid-order, nonfinite, and underflow-to-zero failures. Its
convenience evaluation wrapper returns NaN on those failures. Hypothesis
enclosure export and reduction artifact construction reject nonfinite results.
Reduction comparisons decode at source precision and compare an upward
difference against a downward tolerance.

Checked L2 normalization scales extreme vectors when the ordinary norm is
unrepresentable. Eigenvector perturbations scale down with small matrices;
final computed residuals screen recovered vectors. The default is now the
corrected interleaved tridiagonal solver. Its semantic ID, the dense recovery
ID, and the CCM sector-spectrum recovery identity advance to v3. Prior vector
payloads cannot satisfy these new semantic requests. Unshifted inverse
iteration selects by inverse magnitude and seed overlap; it is not a proof of
algebraic ground-state ordering.

Independent Jacobi computations use the actual matrix scale, stable plane
rotations, checked outputs, and optional eigenvector accumulation. They remain
computed estimates with no tiny-eigenvalue relative-accuracy or sign guarantee.

`weil_spectrum_sonin_hp` now returns the actual compressed finite matrix
spectrum Q^T A_arch Q, of length dim-n_drop. The old length-dim penalized
spectrum was not an exact subspace restriction. An unresolved concentration
cluster crossing the requested cutoff is rejected. The new result does not
certify continuum Sonin positivity or quadrature error.

HP point-root refinement uses safe midpoints, representable decimal stopping
parameters and conservative comparisons. `RootApproximationHp.newton_steps`
counts accepted derivative-based interior updates, with a zero deserialization
default. A terminal derivative diagnostic alone no longer makes a root
cross-check independent or accepted. Independence here means separate executed
algorithms; it does not establish an independent function implementation.

Dense symmetric operators now require exact symmetric storage; the retained
tolerance argument cannot turn asymmetric data into a symmetric map. Norm
upper bounds use upward arithmetic. Nonfinite actions fail, and an overflowed
norm bound is unavailable. Matrix-free callback promises remain the caller's
mathematical responsibility. The historical `rayleigh_quotient` helper still
computes x^T A x; its unit-vector requirement for a quotient is now explicit.

These repairs do not clear historical cache payloads.


### Additional contracts

Maynard matrix entries now reject exponents outside the constructed degree;
native and HP actions reject nonfinite data. Exact spectral-gap structure checks
use decimal arithmetic, require ordered indices, and check the claimed lower
bound against the cluster endpoints. `VerificationReport` adds the default-false
`mathematical_claim_verified` flag. Structural bundle validation and gap-algebra
replay do not set it; actual matrix-bound exact/interval evidence replay does.

The sign diagnostic enum adds `Nonfinite`. Diagnostic ratios use scaled fallback
arithmetic and return an unavailable result when a nonzero value is not
representable. Native operator-batch differences are upward bounds. Solver
cross-checks require finite converged reports, compatible dimensions, nonzero
vectors and distinct route identities. Actual route independence and same-state
identity remain caller obligations. HP overlap is now a conservative lower
bound and one-minus-overlap an upper bound. Neither scalar agreement nor small
residuals prove the selected eigenstate index.

Dense `try_lu_solve` and `try_lu_solve_with` validate the public factor record,
permutation, precision and finite output. Legacy infallible wrappers panic on
invalid data; Toolkit calculations propagate checked errors. Historical
tridiagonal v1 replay remains explicitly separate from corrected new analysis.

Fresh and reused HP Gauss-Legendre tables must pass every root/weight identity,
in addition to low-order moments. Cache validation therefore uses O(n^2) point
arithmetic. `try_gauss_legendre_nodes` offers explicit failure propagation;
legacy tuple APIs require valid input and panic on failure. These checks are not
interval quadrature certificates, and correct node/weight payloads are unchanged.

The screw kernel semantics are `suzuki-sqrt-prime-cusp-v2`: Suzuki's prime weight
is Lambda(n)/sqrt(n), not the old Lambda(n)/n. Old screw reference values above
the first prime threshold are wrong. Checked construction/evaluation validates
support and convergence, and a local expansion handles the cusp near zero.

The Mellin semantics are `eta-stable-real-crossings-v2`. Stable eta evaluation
uses exp(-abs(t)) without a fixed HP asymptotic cutoff. Checked crossing APIs
include sampled zeros, preserve exact midpoint zeros, reject invalid evaluations
and use safe grid/midpoint arithmetic. The legacy `scan_critical_line_zeros_*`
names still return real-part crossing candidates: the imaginary part need not
vanish. Such candidates are not complex-zero certificates or complete zero lists.


Checked `try_truncated_lambda_*` and `try_xi_weighted_mellin_*` APIs now reject
invalid parameters, empty or malformed quadrature, and asymmetric full Fourier
coefficient vectors. Legacy tuple wrappers return NaNs on checked failure.
Unweighted integration permits lambda=1 (zero length); weighted reconstruction
requires lambda>1 and a finite exactly even vector of length 2N+1. HP inputs
must have the working precision, and supplied GL rules undergo O(n^2) validation
per call. Callers performing large scans should account for that validation cost.
The integrand combines the power and eta kernel in logarithmic form to avoid
losing a finite product to independent overflow and underflow. These remain
point quadrature calculations, without a certified integration-error bound.


### Generalized and projected solver contracts

Generalized HP cross-checks now validate finite compatible reports, compare an
upward eigenvalue difference with a downward literal tolerance, and normalize
the computed metric overlap. The interval arithmetic bounds the stored metric
images; it does not enclose an arbitrary callback's internal rounding error.
Same-problem identity, positive definiteness, and route independence are caller
premises. The overlap defect bounds its absolute departure from one.

The projected HP eigensolvers use the shared scale-relative Jacobi core. The
scalar two-by-two generalized solve uses Cholesky whitening and Jacobi instead
of a cancellation-prone quadratic formula. Whitening helpers are shared with
the dense generalized reference; its Householder/QR eigenalgorithm is distinct.
Agreement is not independence of every operation. Native whitening checks
finite intermediates, averages computed symmetry roundoff, and uses a bounded
library eigensolve. Both lanes revalidate public dense records and require
finite, exactly symmetric stored matrices. The legacy native symmetry-tolerance
parameter must still be finite/nonnegative but cannot admit asymmetry.

Generalized, block, restarted and shift-invert HP diagnostics use stable norms
and reject nonfinite/overflowed denominators or nonzero ratios rounded to zero.
Native generalized diagnostics preserve subnormal scale instead of flooring the
denominator. An unrepresentable diagnostic returns an error. These remain point
diagnostics, not rounding or operator-error certificates. A zero operator with
a nonzero metric image has zero relative residual by explicit convention.
HP solver precision is limited to 33..=1000000 bits; a nonzero decimal shift or
approximation bound that underflows is rejected rather than treated as zero.

Exact residual-converged warm starts now receive the required additional
stability observation. Work counters count actual applications and factorizations.
HP scalar generalized reports add the same
`target_ordering_established_by_full_space_projection` field as the native
route. A false value explicitly leaves global ordering unestablished. A true
value records a computed full-space projection, not an index certificate.
For A=diag(1,2), B=I, an e1 warm start can converge at 1 while the requested
largest state is 2. The field is false, and a cross-check against the full dense
route rejects that missed target. Local convergence and guard Ritz values do
not prove that an unvisited invariant subspace contains no requested state.
Use independent full-spectrum or exact/interval index evidence when global
ordering is required. Generalized report algorithm identifiers carry `_v2`.

Historical artifacts require their own validation; these repairs do not clear them.


### Reconstruction and finite prolate contracts

Even Fourier reconstruction now checks every coefficient, finite lambda>1,
exact reflection symmetry, and checked 2N+1 dimensions. HP precision must be
32..=1000000 bits. Safe logarithm identities avoid overflow in lambda squared
or lambda*u. Invalid evaluation points produce NaN through the scalar API;
constructors return errors for invalid data or normalization. Exactly even,
ordinary-range inputs retain the existing ordered evaluation arithmetic.

The prolate matrix implements a centered-flux finite Dirichlet discretization
of -d/dx((lambda^2-x^2)d/dx)+(2*pi*lambda*x)^2. Integer dimensionless fluxes
preserve reflection symmetry; HP arithmetic uses the requested precision even
when lambda was supplied at lower precision. Checked matrix builders reject
invalid precision, grid shape, and nonfinite arithmetic. Compatibility tuple
wrappers panic on invalid input. Finite-matrix calculations do not certify
continuum boundary behavior or discretization error.

Parity includes the center coordinate and uses scale-relative deviations.
Node counting removes a relative 1e-6 noise floor, without an absolute zero
cutoff. Mode selection also requires ordered indices 0 and 4; parity and
thresholded node counts alone are not an index certificate. The integral
constraint uses h*sum(v), the trapezoidal integral of the piecewise-linear
interpolant with zero endpoint values. Checked interpolation requires the
canonical grid spacing 2*lambda/(N+1) rounded at working precision, finite
data and evaluation points. Legacy interpolation/node wrappers panic on
invalid input; try_interp_grid and try_count_nodes return errors. Projected
forms reject nonfinite computed entries; the generalized solve establishes
positive definiteness of the computed Gram matrix, not of a continuum form.

The least-squares comparison scales both sample vectors before products,
rejects nonfinite/zero candidate data, and computes residuals using the actual
returned scalar. Its L2 field is an unweighted discrete Euclidean norm; its
L-infinity field is a maximum on the supplied positive sample points. Neither
is a continuum bound. Unrepresentable scalars, residuals and norms fail closed.

Prolate shared/exact-source caches use arithmetic semantics v2 and payload
schema 3. Integer compatibility payloads require the new arithmetic stamp,
exact cutoff identity, mode, grid and precision metadata. Old entries become
cache misses. A rounded square root of an integer uses exact-source identity;
only an exactly integral source lambda can take the integer shortcut. No
historical cache file is repaired by these source edits. Structural cache
validation remains weaker than an independent spectrum calculation.

The prolate/Weil sampled comparison is not a test of Lemma 7.2. In Zeta Spectral
Triples, that lemma concerns normalized prolate-to-Hermite convergence; relating
the resulting construction to the lowest Weil state is an additional question.
Reference: https://arxiv.org/html/2511.22755v1#S7 . The independent development
fixtures check 25 finite matrices and six complete finite state/sample cases.
They do not establish universal correctness or validate historical results.


### Meromorphic kernels and exact-polynomial contracts

Yakaboylu matrix elements now evaluate epsilon^2/(epsilon^2-a^2),
a=conjugate(s)+s'-1, using scaled arithmetic and compensated real cancellation.
Finite epsilon>0 is required. Checked `try_v_r_matrix_element_*` functions
reject invalid parameters, poles and unrepresentable arithmetic; legacy tuple
functions return NaNs on failure. The HP checked entry accepts explicit output
precision in 32..=1000000 bits and uses 32 guard bits above the largest source
or output precision. A lower-precision epsilon no longer downgrades matrix
construction. This remains point arithmetic, without a rigorous near-pole error
bound. Source inputs retain their stored binary values.

Checked Lorentzian matrix builders require finite nonempty ordinates and
epsilon>0; legacy vector builders panic on invalid input. Eigenvalue helpers
revalidate finite exactly symmetric square storage and positive dimensions.
Native eigenanalysis scales the matrix and uses a bounded library iteration.
The discrete delta diagnostic uses exact equality of ordinates, without an
absolute coincidence threshold that identifies distinct small values.

The positivity reports now distinguish a strict computed positive margin
(`positive_definite`) from `positive_semidefinite_with_tolerance`, and record
the actual tolerance: 64*N*binary64 epsilon or N*2^(16-prec) in HP. Duplicate
ordinates cannot be labeled positive definite; their condition number is
infinite. The HP finite 1e300 sentinel is removed. A zero computed minimum or
unrepresentable ratio also produces infinity. These computed diagnostics are
not rigorous eigenvalue lower bounds or certified condition numbers. The
finite critical-line Lorentzian kernel is positive semidefinite for arbitrary
real ordinates, so its positivity does not independently verify zeta zeros,
RH, or the full operator framework. The formula and its meromorphic continuation
are from arXiv:2408.15135v15, Eq. (52), not the different 2024 journal article
previously conflated with that reference.

Exact rational contour certification revalidates the public rectangle fields
at its entry point. Degenerate or inverted rectangles cannot receive a rigorous
result. Subdivision uses an explicit stack, preserving traversal order without
recursive stack growth; exact boundary vertices return Inconclusive immediately.
Dyadic square-root precision is limited to 0..=1000000 before computing twice
the bit count or allocating big integers. Sturm isolating intervals have
disjoint interiors; adjacent closed intervals may share a non-root endpoint.

The native bisection helper now requires opposite endpoint signs before using
a nonzero residual tolerance. Exact endpoint zeros remain valid. Previously a
tiny nonzero constant could be returned as a root. Continuity remains a caller
premise, and exhausted iteration budgets still return a point approximation,
as documented. This helper is not a convergence or root-count certificate.


### Weighted measurements and normalization range

Weighted distances evaluate the stated finite quadrature sum; they do not
certify continuum quadrature error or callback accuracy. Finite callback
values are required. Exactly equal stored values give a zero residual before
forming a potentially overflowing weight. For nonzero residuals, a logarithmic
fallback combines log(abs(f-g)) with -alpha*log(u) when separately forming the
weight would lose range. MPFR differences at the exponent floor are recovered
with an exact common power of two before subtraction. An unrepresentable
nonzero sample, nonfinite accumulation, or total nonzero positive sum rounded
to zero produces an error. Some finite final integrals can still be rejected
when the chosen finite rule has unrepresentable intermediates. This is an
explicit arithmetic range limit, not evidence of a zero distance.

HP weighted measurements require 32..=1000000 output bits. They operate at
output precision plus 32 guard bits and round the result to output precision.
Low-precision callback objects no longer determine the precision of subsequent
subtraction, weighting, Jacobian multiplication, or trapezoid endpoint averaging.
Promotion does not recover information already lost inside a callback. Generic
HP uniform-grid integration likewise promotes returned samples before arithmetic.
Gauss-Legendre requests now use checked table constructors; invalid precision,
unresolved HP grid bounds and invalid table requests return errors.

Normalized even cosine reconstructions rescale extreme coefficients by a common
power of two before raw summation. That common scale cancels in f_raw(u)/f_raw(1),
preventing overflow of a raw sum whose normalized value is finite. A nonzero
coefficient lost during rescaling is an explicit error. Ordinary-range serial
evaluation order remains unchanged. Rounded normalized coefficients support
evaluation at new abscissae without sampled-profile interpolation; dividing and
renormalizing them can change low bits. They are not a bit-identical replay of
the original evaluation. Documentation formerly claiming lossless exact replay
has been corrected.

Affected managed profile, target-distance, resolution, residual, decomposition,
retained-parent and discretization-distance semantics are versioned. Old
artifacts cannot satisfy requests for the repaired arithmetic. Historical cache
files were not edited or independently revalidated by this source repair.


### CCM finite rank-one quotient contracts

The native CCM state normalization no longer treats every eta pairing below
binary64 epsilon as zero or rejects a common state scale solely because the raw
sum overflows. A common power-of-two scale and a floating expansion preserve
cancellation terms before division. A zero pairing, lost nonzero coefficient,
or unrepresentable normalized value is an explicit error. The HP route uses
the same projective normalization principle with MPFR arithmetic.

For a pivot p, the coordinate-section quotient is assembled directly as
Q_ij = delta_ij*d_i - (d_i-d_p)*xi_i, with xi normalized to sum one. This cancels
the pivot symbolically before arithmetic; two separate overflowing image terms
can no longer contaminate a finite final entry. Native entries use fused
multiply-add. The dense native Schur solve is scaled, iteration-bounded and
checked for finite output. Imaginary tolerance remains an absolute pointwise
tolerance, not a proof of a real spectrum. Physical ordinate conversion checks
range and recovers representable products through combined logarithms when
forming 2*pi/L alone loses range; nonzero unrepresentable outputs are errors.

HP rank-one eigenvalue precision is state[0].prec(), in 33..=1000000 bits.
Inputs may have different precision, but factorization and form arithmetic use
32 guard bits above the largest input precision (inputs above 1000000 bits are
rejected); the symmetric eigensolve returns the requested output precision.
The source Weil matrix must be exactly symmetric before projection. Validation
tolerance must be finite and positive. Shifted metric entries, state residuals,
projected forms, Cholesky pivots/factors, congruence solves and reported spectra
receive explicit finiteness checks. The method identifier advances to
finite_rank_one_weil_metric_guarded_cholesky_congruence_hp_v2.

These remain finite point computations. Native quotient construction alone
does not establish the positive-metric/radical premises of CCM Lemma 5.4. HP
residual/symmetry and positive-pivot checks use numerical tolerances and do not
constitute interval certification. Three-route producers now reject empty route
and independence-class identities before evaluating a peer. Distinct labels
record the claimed route separation; they do not prove implementation independence.


### Generic diagonal-plus-rank-one scalar reference

`diagonal_rank_one_spectrum_f64` now treats stored finite binary64 inputs as
exact dyadic rationals for secular signs and outer bounds. The conditions remain
strictly increasing diagonal entries, nonzero update components and alpha,
finite positive tolerance, and a positive iteration budget. At a pole, the
one-sided sign is supplied analytically instead of evaluating an overflowing
reciprocal at the adjacent float. Exact integer arithmetic forms the weighted
updates alpha*u_i^2 and the outer Weyl bound, including cases where a separate
u_i^2 or update term exceeds binary64 range but the final eigenvalue is finite.

Each returned point is either an exact stored-data secular zero or belongs to
a root bracket whose exact width is at most tolerance*max(1,abs(point)). Width
and tolerance-product acceptance use exact dyadic comparisons. Unresolved
binary64 pole gaps, unrepresentable finite output/residual bounds and exhausted
iterations now return errors. An infinite midpoint can no longer pass because
both sides of a floating comparison overflow. Earlier versions could also return
unconverged points after the iteration budget expired without an explicit failure.

The `residuals` field is strengthened to contain the least finite binary64 upper
bound on each absolute stored-data secular residual. A nonzero residual below
the binary64 range gets the least subnormal upper bound, not a zero sentinel.
The existing record shape is retained; the algorithm identifier is now
diagonal_rank_one_exact_sign_bisection_f64_v2. Roots and residuals retain the
same interlacing order. This exact-sign reference route can cost more than the
old floating-only arithmetic. It uses the existing num-bigint dependency and
does not use a dense eigensolver, preserving its independence from that route.
Finite matrix evidence does not identify a continuum operator or zeta zeros.


## Cutoff-free certificate assembly binding

`CutoffFreeMatrix` certificate and component-evidence methods now validate the
configuration, component shapes, exact symmetry, full rational reconstruction
containment, and a private binding recorded by `assemble`. Previously a caller
could replace tau while keeping the original component digest, remove components,
or change the cutoff and still receive a portable inertia certificate carrying
a CCM assembly note. The inertia result described the changed endpoints, but its
claimed assembly provenance was unsupported. Mutated assemblies now fail.

Public field reads and cloning remain available; external struct-literal
construction is intentionally prevented by the private assembly binding.
Generic exact-inertia APIs remain available for arbitrary interval matrices.
The ordinary assembled numerical entries and component digest convention are
unchanged. Generic portable inertia replay alone verifies the supplied matrix
endpoints, not the special-function assembly derivation or historical provenance.

Cutoff-free precision is checked in 64..=1000000, dimension/capacity overflow
fails before arithmetic, and prime enumeration propagates Result errors.
`CutoffFreeConfig::checked_dimension` and `try_prime_powers_up_to` provide checked
alternatives to the retained convenience APIs with documented panic contracts.
The shared checked sieve now reserves its output list fallibly as well.


## Retained-state geometry arithmetic

Retained-state decimal parsing rejects nonzero values that underflow to zero.
Geometry normalization and reflection defects use guarded hypot accumulation;
the center uses correctly rounded summation. These changes repair false evenness,
zero defects caused by squaring underflow, and a center-cancellation case that
selected the wrong orientation. Common state scales near the MPFR exponent
limits now work when the reported scalars remain representable. Unsupported
range loss fails explicitly. Arithmetic uses 32 guard bits and rounds reported
scalars to the requested precision; these remain computed point diagnostics.

Calculation semantics advance to `ccm-retained-fourier-state-geometry-v2` and
`periodic_trapezoid_guarded_hypot_normalization_v2`. The v1-shaped JSON schema
admits historical v1 and current v2 semantics, while the current managed key and
reader require v2. Historical artifacts are unchanged. Worker-count determinism,
the Fourier convention and absence of any ground-selection/global-sign guarantee
are preserved. The change can alter low bits in ordinary geometry reports.


## Native CCM discovery and domains

`solve_spectrum_f64` now requires an exactly even finite state with nonzero
component sum and positive tolerance/budget. It normalizes the common state
scale, uses the checked root solver, and propagates nonconvergence and evaluation
errors. Tolerance controls absolute/relative squared-coordinate bracket width;
only zero or least-subnormal point residual short-circuits it. This replaces the
scale-sensitive absolute-residual exit and successful exhausted-budget midpoint.
Physical ordinate conversion reuses the checked combined-product implementation.
The routine remains an exploratory point search with no completeness guarantee;
deflated carriers, near-pole/zero roots, multiple signed-gap roots and the finite
exterior window require separate treatment. Its tolerance is not a physical-error
or Riemann-zero accuracy certificate.

Native assembly rejects inconsistent integer/float cutoff fields, propagates
prime-sieve failures, checks size/order overflow, reserves matrix storage fallibly,
and requires finite matrix/eigenpair/state values. The even eigensolve is bounded.
The minimum remains an even Ritz value, not certified full-space ground selection.
No managed cache lookup occurs in these native entry points; historical retained
observations still need source-specific replay.


## Validated real Dirichlet characters

`LFunctionSpec::new` and deserialization now check the complete real-character
contract, including multiplicativity, nonunit zeros and parity. Mathematical
fields are private; use `modulus()`, `values()` and `parity()`. This intentionally
replaces unchecked struct literals/mutation with checked construction. Valid JSON
keeps the same shape. The label remains descriptive and mutable.
`chi_at_prime_power(p,0)` now returns one, including when p divides the modulus.
Built-in tables and positive-exponent values are unchanged. Character data do not
constitute a generalized CCM operator, and the modulus is not necessarily a
primitive conductor. `is_trivial` means the modulus-one zeta character; principal
characters of larger modulus omit Euler factors and return false.


## HP L2 normalization

`try_normalize_l2` uses exact common binary scaling before the canonical squared
sum, at the maximum input precision. This fixes mixed-precision reduction and
partial-square-underflow cases. Outputs share that precision; nonzero component
loss fails explicitly, and errors preserve the input. Uniform ordinary-range
rounding is preserved. It remains a computed point normalization, not a certified
norm or a proof of state selection. The infallible wrapper still emits NaNs on
failure. Managed CCM eigenpair/sector-spectrum/tail-operator identities and
standalone Weil-state admission carry `power_two_scaled_max_precision_l2_v2`.
Historical files are retained but are not thereby revalidated.


## Outward finite tail budgets

`ArchimedeanTailBudget::explicit` now encloses the elementary tail formula with
directed rounding and verifies its strict theorem domain. Mode counts are no
longer truncated to u32. It requires finite T and 64..=1,000,000 precision bits.
Mathematical fields are private; read-only getters replace direct field access.
The log-cutoff and rho getters return intervals. Invalid or unresolved domains
return errors. `finite_cutoff_interval_decision` uses separately justified
eigenvalue endpoints; the checked point helper rejects nonfinite values and
the legacy wrapper returns inconclusive for them. A bare approximate eigenvalue
is not certified by this budget. The type has no persisted cache schema here.


## Discrete weighted projection arithmetic

Deviation projection now validates the grid and precision domain, promotes all
inputs to guarded working precision, uses exact binary scaling and hypot norms,
and returns errors for nonrepresentable intermediates/results. This prevents
successful NaN outputs, false zero residuals and avoidable norm overflow.
The retained-payload caller promotes f before subtracting the target and rejects
nonzero decimal underflow. Managed decomposition semantics advance to v6, with
explicit projection and subtraction arithmetic stamps; algorithm v3 and portable
schema 3. Old payloads remain preserved. Results are discrete point measurements,
not continuum quadrature bounds or certified explanatory models.


## Prolate evidence and residual separation

CertifiedProlateDeficiency's mathematical fields are private with read-only
getters. Their exact algebraic relationship can no longer be changed before a
comparison consumes them. Physical matrix/mode identification remains external.
Checked and logarithmic asymptotic APIs distinguish positive values from exponent
underflow; the legacy point wrapper returns NaN for unsupported calculations.
Finite-state comparison now proves its stored form positive definite by interval
LDL, checks symmetry/domains, and uses scaled guarded arithmetic and hypot norms.
That admission check has cubic cost; the resulting diagnostic values are points,
not forward-error certificates. Supplied truncation labels remain premises.

The ambiguous residual_to_certified_gap helper now always errors. Use
residual_to_complement_separation with distance from mu to the unwanted spectrum,
or residual_to_eigenvalue_gap with both an eigenvalue gap and a certified
eigenvalue error. A raw eigenvalue gap cannot justify the old r/g angle claim.
These APIs have no persisted payload schema or managed cache consumer here.


## Retained sources, transforms and energies

Authenticated RetainedReference/RetainedDataset specs are private with read-only
spec() getters. Edit a cloned definition and capture/admit it again to change it.
Scalar admission rejects nonzero decimals rounded to zero. Managed retained
research semantics advance to v2, so prior computed reports have different keys.
Legacy v1 input definitions may be re-admitted after full current validation;
this does not admit old calculated reports as current results.

Finite Fourier transforms now normalize after reversible binary scaling, orient
by the exact stored center sum, and retain representable small-argument derivatives.
Energy diagnostics scale before products and residual norms, sum stored products
before rounding, and decode eigenvalues at their declared source precision before
promotion. Required physical outputs outside MPFR range cause explicit errors.
Default transform precision stays within the public cap; guard bits are internal.
These are finite point diagnostics, without a source or continuum error enclosure.
Other formulas are outside the scope of this change; shared v2 keys are not a
blanket correctness certificate.


Energy accumulation retains only row scratch space and exact signed/absolute
accumulators. The exponent-span precision budget is at most 3p+1,000,000 bits;
larger requirements fail explicitly. Dense input and arithmetic remain quadratic
in dimension, while additional enlarged-precision storage is linear in dimension.


## Retained projection and finite statistics

Managed retained-research semantics advance to v3. Strictly revalidated v1/v2
reference and explicit-dataset definitions remain readable; prior calculations
have different current keys. Reference coefficient points are decoded at declared
precision before promotion, and projection supports must match exactly as canonical
decimal parameters. Equivalent encodings such as 9 and 9.000 remain compatible.

Projection uses guarded normalization, exact centers, independent binary column
scaling and checked residual arithmetic. ProjectionData adds pivot_metric: minimum
pivots now refer to the column-scaled Gram matrix and do not certify numerical rank.
Required unrepresentable raw fields fail explicitly. The fixed-component ratio is
still a point ratio without a coefficient-error enclosure; a small denominator
can make it highly sensitive.

Finite root-window moments/spacings use checked guarded arithmetic; positive moments
cannot silently become zero after underflow. Stabilization decodes original source
points and scales before relative subtraction. It remains a frozen finite rule,
not evidence of branch comparability or a limiting theorem. Shared exact-center
errors propagate into three consumers. These repairs do not clear historical artifacts.


## Extended compactness and shared normalization

Managed retained-research semantics advance to v4; strictly revalidated v1/v2/v3
input definitions remain readable, while prior calculated reports have distinct
keys. Extended requests explicitly name the new normalization and compactness
arithmetic and the guard cap. Compactness reports arithmetic_precision_bits.

Source normalization uses reversible binary scaling, guarded hypot and exact
stored-point orientation; unrepresentable components fail explicitly, and all
callers propagate errors. Compactness directly encloses finite-state origin
moments and exponential weighted norms, accepts a value only when both directed
bounds round to the same point, and uses expm1 plus the exact zero-rate limit.
Unresolved cancellation fails after at most 4096 additional guard bits. Sigma is
absent when the origin is exactly zero. These are finite stored-state values;
source construction error and continuum claims are excluded. The optional memory
limit checks an additional scratch estimate, not total host process memory.
Default public precision is capped at 1,000,000 bits. Shared normalization fixes
do not validate all consuming formulas or clear historical artifacts.


## Finite atom arithmetic

Managed retained semantics advance to v5; strictly revalidated v1..v4 definitions
remain readable. Weighted-tail requests and the signed-kernel local checkpoint
namespace name the new arithmetic, preventing reuse of prior computed values.

Atom coordinates, weights, cutoffs and evaluation points are decoded at the
external input's declared precision before promotion. Exact dyadic accumulation
preserves signed cancellation and finite suffix mass. Directed moment/kernel
arithmetic accepts values only when enclosure endpoints agree after rounding.
Unresolved or unrepresentable moments are withheld; valid mass/count fields remain
where available. Kernel failures withhold all sums. used_atoms is zero for an
unresolved kernel. Nonzero origin weights remain an explicit inverse-moment
obstruction, and excluded atom identities/counts retain their meaning.

An exact accumulator's exponent span is capped at one million bits; interval
retries use at most 4096 extra guard bits. Optional working-byte limits apply to
conservative per-group/per-kernel additional scratch estimates, not the full host
process or all concurrent work. The results are finite stored-point expressions,
excluding source construction error and unprovided infinite tails.


## Finite cluster geometry and source-point reporting

Managed retained semantics advance to v6; strictly revalidated v1..v5 definitions
remain readable. Cluster arithmetic v3 records a bounded pivot-based precision
policy and at most 4096 guard bits. Guarded homogeneous normalization, checked
products/sums and the existing checked Gram solver replace the cluster's unchecked
normalization and residual paths. Declared cluster points are decoded at their own
precision. Squared-residual range loss fails explicitly. Original signs, cross-N
embedding and point tie semantics are retained; ties explicitly disclaim a unique
match. Minimum pivots refer to the unit-column Gram and are not rank certificates.

Near-dependent columns trigger precision escalation or an unresolved result. The
computed pivot is a precision-selection proxy, not a certified eigenvalue bound;
outputs remain point measurements without source-construction or forward-error
enclosures. Explicit byte limits estimate additional scratch at the maximum guard.

An atom coordinate-echo follow-up promotes the declared source point to report
precision before decimal serialization. The atom calculation itself is unchanged.
Its request and local kernel checkpoint identities advance to prevent reuse of
the earlier echo. Other fixed-guard projection callers and formulas are outside
the scope of this change. Historical artifacts are not cleared.


## Finite weighted-profile arithmetic

Retained semantics v7 and a new weighted-profile arithmetic/output identity
replace unchecked near-constant subtraction, raw Gram elimination, and input
precision reinterpretation. Strictly revalidated v1..v6 definitions remain
readable. Reference and basis decimals are decoded at declared input precision.
The first stored target point must equal one exactly.

The Fourier difference is formed algebraically before evaluation; homogeneous
source normalization and independent binary column scales preserve small signals
and raw fit units. Directed finite-grid intervals and positive interval Gram
pivots support bounded precision retries up to 4096 guard bits. Every measured
field has outward real-decimal lower/upper bounds; the original scalar name is
an enclosure midpoint estimate. In particular, a tiny residual midpoint with a
zero lower bound does not establish a nonzero residual. `minimum_scaled_pivot`
replaces the ambiguous weighted `minimum_pivot`. Ratios with a denominator
enclosure containing zero are withheld. Unrepresentable mandatory raw fields
fail, and precision-exhausted fits withhold coefficients.

The bounds cover finite arithmetic on stored inputs, not source construction,
reference approximation, quadrature error or continuum limits. Managed identities
do not validate historical reports.


## Stable normalized Fourier projection

Retained semantics v8 and an explicit projection arithmetic identity replace
normalize-then-subtract and fixed-guard Gram fitting. Center-one differences use
exact-center determinants; unit-normalized differences use a stable rationalized
decomposition. A finite interval Gram solve escalates precision within 4096 guard
bits and withholds unresolved fits. Existing cluster point arithmetic is separate.

ProjectionData now includes arithmetic_precision_bits and arithmetic_enclosures.
Every measured scalar is an estimate with outward real-decimal bounds. Tiny
positive residuals are checked relative to their own magnitude when positivity
is resolved. Ratios with a denominator enclosure containing zero are withheld;
fixed b parameters denote exact decimals. Basis columns retain raw Fourier units
and the full-support L2(dx) metric. Strictly revalidated v1..v7 source definitions
remain readable, but prior calculation reports do not acquire the new assurance.

The bounds exclude source construction, reference approximation, ground-state
selection and convergence. Live producer/schema consistency is covered by the
retained report schema contracts below.


## Retained report schema contracts

The retained numerical repairs advanced producer semantics while
the generated schemas still accepted only revision 1. Actual public reference
and projection reports failed shape validation; live tail-operator reports also
exposed an omitted normalization-method field. The generator now enumerates
the intended archival revisions 1 through 8, rejects unknown revisions, and
requires each new arithmetic identity and projection enclosure field from its
applicable revision. Revision 1 tail reports may lack the normalization field;
revision 2 and later require its known value. Archive compatibility is a shape
contract, not clearance of earlier arithmetic.

The geometry schema explicitly accepts the repaired geometry revision 2
alongside revision 1. Schema acceptance does not certify a measurement.

## Exact cutoff geometry and source-owned reference preparation

State geometry v3 constructs log(C) from the exact decimal cutoff using directed
endpoints. It returns a point only when both endpoints round identically, or fails
explicitly after4096 guard bits. Its symmetric-grid first energy moment is exactly
zero for every admitted real Fourier coefficient vector. Other geometry outputs
remain point and quadrature diagnostics, without source-error or sign certificates.
Historical geometry v1/v2 remains readable; new calculations use a distinct key.

Reference preparation v2 decodes supplied numerical points at their owning
precision before promoting and serializing them at the output precision. This
includes nested samples, atoms, forms, recipes, actions, responses and model points.
Independently precision-tagged comparisons/certificates keep their own contracts.
Finite reference/basis precision contributes to working precision; retained roots
must belong to the named state and retain their payload precision. The complete
prepared bundle is hashed into the source key, preventing reuse under old values.

Real sampled bases must be even. Center-one target samples retain the exact center
identity, and raw basis centers use exact rounded summation. Transform rescaling
uses hypot instead of a raw squared norm. These changes repair false non-even
rejections, invented extra input bits, lost support precision near C=1, and a
cancellation case that previously reported zero at a center defined to be one.
They do not certify all interior sampled values or historical research conclusions.

### Tail-model stored points and exact finite forms (v18)

Retained research semantics v18 separate the owning precision of supplied tail
forms from the arithmetic precision of their solve. Recipes decode their basis,
atoms and corrections at the external source precision; synthesized entries are
rounded at working precision. Retained energy remains a scoring input with its
own precision. The finite-tail checkpoint advances to v2 and records that precision.

Finite polynomial forms now use exact dyadic integer arithmetic followed by one
rounding per entry. This repairs false zero products from intermediate underflow
and removes summation-order loss for the admitted exact calculation. The exact
span is capped at 8,000,000 bits with a workspace preflight and explicit failure
outside resource or output exponent limits. Signed atom weights remain permitted.
An absent tail correction still means the omitted tail is unknown.

The generalized model, Cholesky factorization, eigenvectors and counterfactual
energies remain computed point diagnostics. Error expressions are conditional on
caller-supplied model assumptions. Archived v1-v17 reports remain readable but
are not reused for v18 requests. These repairs do not revalidate historical data.

### Root adapters and outward serialization (v19)

Retained root enclosures now evaluate the original payload point. Default contour
construction uses those points, and explicit contour endpoints use the external
input's owning precision. Bounds serialize with directed decimal rounding. The
finite-contour segment checkpoint advances to v2 to exclude earlier endpoint and
point-decoding behavior. Retained-root rows record the point and its precision.

Root transport uses exact-cutoff logarithm enclosures, original velocity points,
and the directional producer's outward intervals. Component closures, support and
operator motion, and supplied response additivity carry arithmetic bounds. The
operator term uses `response*t/(2*tau)` to avoid an unnecessary support square.
Shifted-secular comparisons preserve their distinct branch and source conventions.
The root and displacement hypotheses remain conditional, and unresolved arithmetic
is explicit. Archived v1-v18 reports remain readable; current cache identities and
schema contracts require the v19 arithmetic. Historical incidence is unvalidated.


### Directed prefix exports and full cache replay (0.15.1)

The prior prefix export residual squared unscaled quantities. For a stored
matrix scaled by 2^-536870880, this could underflow to zero and accept an
eigenpair whose actual backward error exceeded its tolerance by more than
92,000 times. The repaired gate bounds
`||A v-b||_2 / (||A||_F ||v||_2 + ||b||_2)` with directed arithmetic, exact
binary scaling, and a second scaling before residual squaring. Eigenpair
checks evaluate the exact stored `lambda*v` product inside the enclosure.
Norm-one checks and serialized error bounds are outward; unsupported exponent
spans fail closed. This verifies finite stored-point export equations, not
matrix assembly, continuum errors, forward eigenvalue accuracy, or branch identity.

Prefix cache reuse previously checked only part of a report: validly addressed
synthetic cache entries with altered inverse traces and negative error values
could pass. Reuse now recomputes the complete typed report from its retained
inputs and requires exact equality, including all moments, diagnostics, and
exports. This costs O(D^3) arithmetic on a warm hit; it does not recalculate Tau
matrices or solve for new eigenstates. Fresh results carry a process-local
content seal to avoid duplicate work. Explicit verification modes still replay.

The historical v4/v5 child identities are superseded by current two-moment
and extended identities v8/v10. Historical
v1/v2/v3 observations remain readable without relabeling their assurance.
Source manifests and generic numerical prefix-moment formulas are unchanged.
The counterexamples are synthetic; historical cache incidence requires separate
revalidation.


### Retained-reduction cache replay (0.15.1)

The prior retained-reduction reader checked shape, finite values, ordering,
and self-reported residuals. A synthetic cache for diag(2,3) could return
fabricated eigenvalues -999,-998 and a zero source norm with a passing verdict.
The current reader recomputes the complete report from the retained matrix
and requires exact equality, including Q-based diagnostics and the spectrum.
Warm reuse therefore costs O(d^3). Fresh ordinary results use a process-local
content seal; explicit verification modes replay. Canonical parent bindings
remain required, and the original matrix is never reconstructed or replaced.

The new child identity is `ccm-retained-reduction-v0.15.1-v2`, with minimum
reader 0.15.1. The Householder calculation and payload's computed diagnostic
assurance are unchanged. This is numerical replay, not an interval spectrum,
positivity, branch, construction, or continuum certificate. Historical objects
need fresh replay; this synthetic counterexample establishes no historical
incidence.


### Native window arithmetic and reconciliation (0.15.1)

Finite reach planning now resolves `ceil(height*ln(C)/(2*pi))` from rational
logarithm and arctangent-series enclosures. An unrepresentable count or unresolved
integer boundary returns an error. Invalid index/height targets reject; an
asymmetric height window uses the larger absolute endpoint. Digit-to-bit ceilings
also use rational bounds and checked schema conversion. The zero-count height
predictor remains an asymptotic planning estimate, never a certified zero bound.

The native secular function and its derivative sum the exact stored binary64
rationals and round once after cancellation. Intermediate denominator overflow
can no longer turn a representable value into zero. Nonfinite final values and
nonzero values that round to zero are unresolved. Exact rational arithmetic can
cost more than the previous floating-point sum. Balanced exact addition reduces
denominator work. Guards limit the inputs to one million terms, aggregate term
storage to 67,108,864 rational bits, and each partial numerator/denominator to
4,194,304 bits. They fail explicitly and are not runtime guarantees. These helpers
add direct use of the already locked num-rational 0.4.2 and num-traits 0.2.19
packages. The existing HP scientific solvers are unchanged by this repair.

Window reconciliation validates every interval and exact midpoint containment.
A certified count alone no longer promotes discovered or cross-checked roots:
certified completeness requires every candidate to carry Certified status.
Those statuses and the assertion that the count concerns the same window are
caller premises from prior verification; this adapter checks consistency, not
the certificates themselves. Decimal ordering and duplicate comparisons retain
all supplied digits. Declared root indices are preserved, missing indices stay
unknown, and zero/overflowing indices fail. Discovery remains unverified.

Independent fixtures cover these boundaries, including exact secular evaluations,
reach calculations, digit-to-bit ceilings, and public overflow/assurance/enclosure
regressions.
Historical effects remain unvalidated.


### Default Householder reduction (0.15.1)

The general `householder_tridiag_hp`, `dense_symmetric_tridiagonal_hp` and
`dense_symmetric_eigenvalues_hp` APIs now use the scaled opposite-sign reduction.
The old unscaled path could underflow a column norm and silently drop a nonzero
coupling; its same-sign reflector also canceled on nearly tridiagonal inputs.
That historical arithmetic is removed from current computation. Exact replay
of earlier outputs requires the pinned historical source, not these APIs.

General APIs accept finite, exactly symmetric square storage at33..=1,000,000
working bits and round input entries once to that precision. This preserves
explicit precision reduction used by guarded quotient-spectrum callers.
The `_stable` APIs continue to reject down-rounding and retain their existing
64-bit minimum, arithmetic and `householder-scaled-opposite-sign-v1` identity.
Their managed retained-source artifact identities therefore do not change.

New dense-reference reports identify the corrected route as
`xc_numerics_dense_scaled_householder_qr_reference_hp_v2`; generalized whitening
reports use `dense_generalized_cholesky_whitening_scaled_householder_qr_hp_v3`.
Generic public spectra, Weil convenience spectra, quotient comparisons and
sector-certificate discovery guides now reach the corrected reduction. Point
results remain computed; certification still requires original-source bounds.
Downstream QR can explicitly reject extreme exponents even when reduction is
representable. This repair does not promise all-scale QR success or relative
accuracy/sign assurance for eigenvalues tiny compared with the matrix norm.

Regressions cover tiny-column underflow, analytic globally scaled spectra,
explicit working precision and invalid domains.
Historical incidence remains unvalidated.


### Selected HP eigenpair matrix scale (0.15.1)

Selected inverse-iteration acceptance now uses an upward absolute row-sum norm
of the stored tridiagonal matrix. Padded Gershgorin search endpoints no longer
set its residual threshold or scaled backward-error denominator. The previous
absolute padding floor could accept an inaccurate vector for a tiny matrix:
for 2^-400*diag(-1,1), the selected positive state had relative residual about
0.607 yet passed at128 bits. The corrected gate rejects that state. A sufficiently
resolved input tolerance recovers the correct eigenspace; exhausted adaptive
recovery remains inconclusive. Only an exactly zero matrix uses unit normalization.

Eigenvalue bracketing and its managed spectral artifacts are unchanged. The
selected-eigenpair and adaptive adapters have no implicit cache access; persisted
computed states need source-bound revalidation. Residual reports are computed
estimates, not interval certificates. Existing boundary rejections, unrepresentable
norm bounds and inadequate absolute tolerances remain explicit failure states.
Historical incidence remains unvalidated.


### Gaussian target exponent range and auxiliary parameters (0.15.1)

HP Gaussian products now use a log-domain monomial fallback when separate
factors lose exponent range. Previously exp(-x) could become zero even though
exp(-x)*x^63 and the normalized target were representable. Tail comparisons
also use logarithms, and estimated losses at the exponent floor must fit the
component's relative working-precision budget. A vanished partial sum cannot
establish convergence for a nonzero polynomial. Unresolved range or term-budget
cases return errors. Nonzero decimal coefficients/scales cannot silently parse
as zero in the HP backend. Requested HP precision is checked before allocation:
1 through 1000000 bits, plus 64 internal guard bits; an external provider must
also support the requested working precision under its existing contract.

Both backends reject overflowing or underflowing nonzero solved auxiliary
parameters. Final auxiliary values must be finite, and HP returned values and
parameters must remain representable at requested precision. The Gaussian
definition identity advances to gaussian-series-log-range-checked-v3, including
Gaussian auxiliary series on external targets. External-only protocol identities
are unchanged. No historical payload is relabeled or numerically cleared.

Public-API regressions cover the repaired boundaries and cache identities.
Independent fixtures cover these boundaries in HP and binary64, including
log-domain reference cases across the MPFR underflow boundary.
These are computed point values and tail estimates, not full interval
certificates or uniform accuracy guarantees for arbitrary cancellation.
Historical impact remains unvalidated.


### Directed interval inertia and portable replay

Cutoff-free CCM full-matrix inertia now uses outward MPFR interval Schur
updates at the assembly precision. This bounds significand growth that made
the exact-rational recurrence impractical even for small high-precision
matrices. Exact input assembly endpoints remain in the portable certificate;
strictly signed 1x1 or 2x2 pivots determine counts. Unresolved signs stay
inconclusive, and invalid arithmetic returns an error.

Portable inertia schema 1 retains its historical exact-rational replay.
Schema 2 selects directed MPFR replay, records exact dyadic pivot enclosures,
and binds precision to the certificate digest. Replay compares every pivot
and count; unknown schemas reject. The generic exact-rational entry point and
selected-eigenvalue shifted-inertia proofs retain their existing arithmetic.

New sector-gap certificate requests use
`ccm-cutoff-free-sector-gap-certificate-v0.15.1-v4`, bind the full-matrix
inertia algorithm, and require Toolkit 0.15.1. The outer sector certificate
schema and the matrix assembly identity are unchanged. Tau point payloads
are unchanged; their separately content-addressed assurance evidence uses
the new portable proof schema. Existing valid schema-1 proofs remain valid
historical proofs; a slow algorithm does not by itself invalidate their counts.


### Stored-vector finite arithmetic

Stored diagonal actions and sequential dot products reject nonfinite decoded
inputs and nonfinite computed results. Previously segmented output could accept
infinity while file-backed output rejected it, and stored dot products could
return success with infinity or NaN. Diagonal chunks are checked before writing;
earlier chunks may already have been written if a later chunk fails, so output
must be discarded on error. Finite successful arithmetic keeps the same order
and results. These are binary64 point operations, with ordinary rounding and
underflow, not exact dot products or certified enclosures. Four regressions
cover numerical failures, exact integer references across chunk boundaries,
and invalid retained bytes. Historical incidence remains unvalidated.


### Shifted eigenvalue-count input dimensions

Both public shifted-count APIs now check positive dimension and checked n*n
equality with the supplied matrix length before cloning or computing diagonal
offsets. A malformed overflowing dimension previously caused a panic with
overflow checking, or a practically unbounded wrapped-index loop in release.
Invalid shapes now produce the existing inconclusive result. Valid shifts,
exact-rational inertia, selected-index decisions and portable proof bytes are
unchanged; existing valid cache and certificate identities remain applicable.

Independent rational orthogonal spectral fixtures cover threshold contacts,
interval perturbations, selected indices and clusters, including conclusive
counts, unresolved boundaries and portable selected proofs. Valid results match
the prior implementation, including pivot records. Invalid
domains, selection controls and proof mutations also reject. This does not
remove the exact-rational path's scaling limitation or establish historical
correctness outside the tested and derived scope.


## Sampled crossing diagnostics

Standalone native and HP target-crossing APIs reject nonfinite profile/residual
samples and non-increasing grids. Opposite nonzero signs remain connected across
exact-zero samples; the initial sign is the first nonzero sampled sign. HP grid
counts and indices no longer narrow through u32. Requested HP precision must
be 1..=999936 bits, leaving the existing 64 guard bits within the target domain.
The compatibility method integrand_appears_smooth only reports absence of a
detected sign change; it certifies neither smoothness nor a convergence rate.
These APIs do not persist artifacts. The separate retained residual-analysis
producer keeps its explicit adjacent-nonzero-profile-samples policy and identity.

Receipt adapter tests use the shared atomic-counter fixture-path helper,
avoiding concurrent timestamp collisions. This test-isolation fix changes
no numerical or persisted artifact semantics.


## Private cache staging ownership

Filesystem atomic writes and replacements reserve private sibling files with
exclusive creation and a checked process-local counter. Existing names are
skipped, with a bounded 128-collision retry; writing/sync failures clean up only
the owned staging name. Equal timestamps can no longer make two writers share
a staging handle and change an already published file. Logical payload bytes
and cache identities do not change. Destination replacement/index merging is
outside the scope of this change.

The same ownership rule now covers encoded-object adoption and corrupt-part
quarantine. Adoption keeps exclusive hard-link staging and uses exclusive
creation for a cross-filesystem copy fallback; failed reservations never delete
another writer's path. Quarantine first reserves its destination exclusively
and cleans up only its own failed reservation. These changes do not establish
concurrent index-merge or crash-atomicity guarantees.


## Native extreme-solver scale repairs

Power and block iteration now use a dimensionless shifted operator; Lanczos
breakdown uses the operator's scale without a unit floor. Block residual and
Frobenius norms preserve small nonzero values, backward errors avoid overflowing
or floored denominators, and cluster comparisons avoid merging overflowing
gaps. Native rounding and iteration counts can change. Block checkpoint schema
version 2 rejects version 1 continuation under the changed arithmetic.

These are computed point results; they do not establish certified global
eigenvalue ordering.


## Response stored-point arithmetic and residual acceptance

Earlier response arithmetic could turn finite norms into zero or infinity,
canceling dot products becoming NaN, loss of higher-precision source components,
and incorrect rounding of ordinary norms. Dot products, Euclidean norms, and
shifted Frobenius norms now preserve exact stored MPFR inputs, use directed
arithmetic with reversible binary scaling, and require both enclosure endpoints
to agree on the target rounding. Shape, precision, resource, exponent-range, and
unresolved-rounding failures return errors.

The bordered response acceptance diagnostic is an upper bound on
`||(A-lambda I)v + u*mu + f, u^T v||_2 /
(||A-lambda I||_F * ||v||_2 + ||f||_2 + |mu|)` for the exact stored points.
For a zero denominator it returns an upper bound on the numerator by explicit
convention. Separate binary exponents avoid overflowing or underflowing products
before the final quotient. A supplied cached matrix norm must match the correctly
rounded norm of the actual matrix; it cannot inflate the denominator. Directed
rounding prevents a nonzero residual from becoming a false zero. These checks
bound the displayed finite-data diagnostic, not source assembly error, continuum
error, or forward error of the response solve.

Response arithmetic identities have changed; prior response payloads require
recomputation under the new semantics. Current identities are defined in `hp.rs`.
Root-only retained repair does not
upgrade old actions, tangents, or residual evidence to the current arithmetic.

The response forcing projection `action_i - unit_i * eigenvalue_response` also
requires directed endpoint agreement for correct rounding of the exact stored
inputs. A product that rounds to `action_i` can no longer erase the remaining
nonzero forcing; a temporarily unrepresentable product may cancel before the
final, representable point is materialized. This does not remove errors already
present in the input action or state.

Response state sums now use one rounding of the complete exact stored sum.
The normalization derivative evaluates
`(target_velocity - scale * sum(response)) / sum(unit)` with directed scaled
arithmetic and requires endpoint agreement at the output precision. Its supplied
rounded state sum must replay from the actual state, while the division uses an
enclosure of the exact sum. This prevents overflowing intermediate products,
underflowed products, or cancellation in the summation from silently changing the
stored-point derivative. The input scale, target velocity, state, and tangent
remain computed source points; their construction has separate error obligations.

Response input construction now uses correctly rounded components
`xi_i / sqrt(sum_j xi_j^2)`, the scale `sqrt(L) / sum(unit)`, and target velocity
`1 / (2 sqrt(L))`, with all operands interpreted as exact stored MPFR points.
Directed sums, roots, and quotients preserve input precision, and separately
scaled exponents allow a representable unit vector even when its intermediate
norm is outside MPFR range. Results require endpoint agreement; invalid shape,
precision, cutoff, zero denominators, resource/range failures and unresolved
rounding are explicit errors. The rounded unit components need not have an
exactly unit rational norm. These formulas remove repeated-rounding errors in
the constructors, without proving the source eigenstate, assembly, action-kernel,
continuum or forward-solve accuracy. Prime-response epoch v11 and u-flow epoch
v12 supersede v10/v11; retained root-only repair keeps its narrower scope.
