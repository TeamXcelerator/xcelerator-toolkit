# Using retained prefixes for convergence research

The prefix ladder can supply observations at every accepted dimension from one
retained matrix. These observations describe finite point matrices. Source
accuracy, equality to independently constructed smaller matrices, and a model
for convergence to an infinite-dimensional limit are separate questions.

## What the two inverse moments determine

Write L=1/sqrt(T2), U=T1/T2, and r_i=lambda_1/lambda_i for i>=2, with
eigenvalues ordered increasingly. For exact SPD moments, set S1=sum(r_i)
and S2=sum(r_i^2). Then

    L = lambda_1 / sqrt(1+S2)
    U/L - 1 = (1+S1)/sqrt(1+S2) - 1
    (lambda_1-L)/lambda_1 = S2/2 + O(S2^2).

The lower endpoint's relative error depends on all excited inverse modes.
In dimension d, S2 <= (d-1)*(lambda_1/lambda_2)^2. Ground-state isolation
alone does not remove this multiplicity factor.

The v2 report retains `gap_ratio_estimate = U/L-1` and
`smallest_eigenvalue_second_order_estimate = L*(1+(U/L-1)^2/2)`.
`PrefixObservable::GapRatioEstimate` and
`PrefixObservable::SmallestEigenvalueSecondOrderEstimate` expose typed,
source-bound observations with explicit **two-mode asymptotic assumptions**:
r_2 must be small and dominate both excited-mode sums. Under those assumptions
the width approximates r_2 and the correction cancels the leading r_2^2/2
underestimate. Neither expression is a certified bound.

Generally the width measures an aggregate inverse-spectrum contribution,
approximately S1. Two moments cannot identify a spectral gap ratio uniquely:
inverse spectra {3,2,1} and {19/7,17/7,6/7} both have T1=6 and T2=14,
but lambda_1/lambda_2 equals 2/3 and 17/19 respectively. The correction can
overcorrect when several excited modes matter. Use independent spectra or
additional moments to test the two-mode model.

More precisely, as S1 tends to zero (with positive ratios), the corrected
estimate divided by lambda_1 is

    1 + (S1^2-S2)/2 - S1*S2/2 + O(S1^4).

The quadratic term is sum_{i<j, i,j>=2} r_i*r_j. In an exactly two-mode
spectrum it vanishes, leaving a negative cubic residual -r_2^3/2. When
r_3 is of order r_2^2, the terms r_2*r_3 and -r_2^3/2 have the same order.
Inferring a third eigenvalue from the cross term alone then introduces a
systematic error. The regression suite checks both the two-mode and
hierarchically separated three-mode expansions.

Dimension one has no second mode: its gap field is absent and the gap adapter
omits that row; its second-order field equals L (an empty excited-mode sum).
Unresolved negative width in higher dimensions is not repaired; unavailable finite
model values cause an explicit adapter error. Historical v1 reports remain
readable for their original scalar observations, but missing new measurements
are not interpreted as zero.

`newest_inverse_trace_fraction` is the newest contribution to the current
inverse trace. Its decay is a finite-ladder convergence indicator. Interpreting
it as a convergence rate toward an infinite-N limit requires a declared model
and independent resolution evidence.

## Third inverse moment and a testable two-mode fit

With columns X_i=v_i/sqrt(sigma_i), the accepted prefix inverse is X*X^T.
The normalized Gram matrix G=X^T*X has the same nonzero spectrum. Thus
T_j=tr(G^j). For a new border g and diagonal d,

    T3_new = T3_old + 3*g^T*G_old*g + 3*d*(g^T*g) + d^3.

The additional quadratic form costs O(k^2) per prefix, O(D^3) overall.
A retained Gram triangle costs O(D^2) storage. These are the same complexity
classes as the ladder, with additional arithmetic and memory costs.
The implementation reuses the signed dot products computed for T2, normalizes
them at working precision, and accumulates the symmetric quadratic form with
a fixed indexed reduction tree. The report retains `inverse_cube_trace`, its
increment and border quadratic form, the normalized Gram's independently
accumulated square trace, and its discrepancy from the established T2 route.
That discrepancy exposes the different finite-precision normalization paths.

For exact SPD moments, stronger endpoints are

    L3 = T3^(-1/3) <= lambda_1 <= U3 = T2/T3,
    L3 >= L2, U3 <= U2.

Their computed fields remain estimates, with no enclosure of arithmetic or
source error. They are exposed as `InverseCubeTrace`,
`SmallestEigenvalueCubeLowerEstimate` and
`SmallestEigenvalueCubeRatioUpperEstimate` observations.

The optional report also fits exactly two positive inverse modes mu1>=mu2
to T2 and T3. Writing q=T3/T2^(3/2) and s=mu1+mu2, the selected root of

    (s/sqrt(T2))^3 - 3*(s/sqrt(T2)) + 2*q = 0

lies in [1,sqrt(2)] and is 2*cos(acos(-q)/3). The two inverse modes follow
from their sum and difference sqrt(2*T2-s^2). Reciprocal estimates of the
first and second eigenvalues, their ratio, and the independent trace closure
residual (T1-s)/T1 are retained and exported through the `TwoMode*` observables.
Unlike the second-order correction, this solves the two-mode model without
truncating its expansion. It does not remove all effects of additional modes.

T1 and T2 already determine an exactly two-mode spectrum; T3 tests that model.
Fitting T2,T3 instead suppresses smaller inverse modes and leaves T1 as a
closure diagnostic. Three moments do not identify a general spectrum:
inverse spectra {2,3,9,10} and {1,6,6,11} both have T1=24, T2=194, T3=1764,
but different smallest eigenvalues and gap ratios. This counterexample is
checked exactly in the test suite.

The report uses explicit statuses for a single dimension, resolved fit,
unresolved separation, values outside the two-mode range, and unresolved
arithmetic. No clipping repairs q, a negative computed discriminant or a
missing second mode. A near-boundary fit can be ill-conditioned; an unresolved
status or a nonzero closure residual needs interpretation with source and
working-precision evidence. The observation adapter returns an error for an
unavailable finite fit, and omits dimension one for second-mode observables.
A small closure residual is not a proof of a two-mode spectrum.

### Closure as a conditional third-eigenvalue scale

Let mu_i=1/lambda_i denote the actual inverse modes and s the sum fitted to
T2,T3. For a well-separated hierarchy r_3 << r_2 << 1, with the inverse tail
below mode three negligible compared with mu_3,

    Delta = T1-s approximately equals mu_3,
    rho = (T1-s)/T1 approximately equals r_3 = lambda_1/lambda_3,
    lambda_3_scale = 1/(T1*rho) approximately equals lambda_3.

Here `rho` is the recorded `relative_trace_closure_residual`. No additional
matrix computation is required. The reciprocal deficit 1/(T1*rho) avoids
substituting an estimated lambda_1 for 1/T1. Using lambda_1_estimate/rho is
also a leading-order scale only when T1 is dominated by mu_1.

To see the conditions, write a=r_2 and R_j=sum_{i>=3} r_i^j. Linearizing the
T2,T3 fit about the first two inverse modes gives

    (s-(mu_1+mu_2))/mu_1 approximately equals
        (1+a)*R_2/(2*a) - R_3/(3*a).

Thus the unnormalized closure primarily measures the omitted inverse trace,
mu_1*R_1. Under third-mode dominance the relative corrections to rho/r_3
have scale r_2, r_3/r_2 and sum_{i>=4} r_i/r_3. These must all be small
before interpreting rho as an individual third-mode ratio. When several tail
modes contribute, the deficit estimates their aggregate inverse trace instead.
For example, two equal small tail modes give approximately 2*mu_3 and a
reciprocal deficit near lambda_3/2. Both the isolated and doubled-tail cases
are checked in the regression suite.

Use the scale only for a resolved fit and a positive, precision-stable
closure. Subtracting s from T1 can lose digits; zero or unresolved closure
must not be inverted. These conditional estimates do not identify a general
third eigenvalue and are not spectral bounds or certificates.

## Precision planning and observed cancellation

A useful conditioning-based planning heuristic is

    source bits >= ceil(log2(10) *
                        (log10(||A||) + expected eigenvalue depth
                         + wanted relative decimal digits + guard digits)).

Here depth=-log10(lambda_min); 3.322 is a conservative decimal approximation
to log2(10). For ||A|| near one, the decimal budget reduces to depth + wanted
digits + guard. This is not an accuracy guarantee: assembly error, quadrature,
dimension-dependent constants and factorization growth need separate evidence.
Wider arithmetic on the same source cannot recover construction digits.

`PrefixObservable::PivotCancellationDigits` reports log10(scale/sigma).
The v2 `innovation_cancellation` object records the worst back-substitution
component's sum of absolute rounded terms, absolute result, ratio and log10
ratio. A zero result from nonzero terms has null ratio/digits and an explicit
status. An identically zero inner sum uses ratio one. These computed local
diagnostics do not enclose propagated errors or measure the full condition
number. `PrefixObservable::InnovationCancellationDigits` requires a finite
recorded value and otherwise returns an error identifying the prefix.

A large effective inverse rank may indicate several important inverse modes
or insufficient precision. Stops, rising cancellation, precision disagreement
or direct eigenvalues outside computed moment endpoints warrant investigation.
Small diagnostics do not certify accuracy. The dense stable solver returns
point estimates: even a positive value may be unresolved relative to
matrix-scale rounding or source error. Compare arithmetic on an unchanged
retained source separately from independently assembled higher-precision data.

## Read a canonical local shard without rebuilding its matrix

The `ccm_prefix_retained` example accepts the original `{manifest,payload}`
runtime source form and a canonical source form:

```json
{"shard_manifest":"/local/shard/manifests/ab/MANIFEST_DIGEST.json"}
```

When using canonical sources, the request must also supply explicit limits:

```json
{
"local_shard_read": {
  "scratch_directory": "/local/scratch",
  "maximum_payload_bytes": 536870912,
  "maximum_package_bytes": 536870912
}
}
```

Keep the existing `matrix`, `eigenpairs`, `approved_payload_digests` and
`options` request fields described in [prefix analysis](CCM_PREFIX_ANALYSIS.md).
Paths are relative to the request directory unless absolute. The scratch
directory must be outside the source shard, with no `..` components.

`xc_cache::read_local_shard_json` checks the canonical manifest path/digest,
local descriptor and active index entry, selected encoding, every ordered
part, ZIP metadata and decoded logical payload digest. It reuses the existing
package machinery and preserves source visibility, declared assurance and
canonical provenance. Missing parts, incompatible versions, noncomputed or
inactive sources, and exceeded limits fail without acquiring a replacement.
The approved list must name the logical `payload.json` item digest, not the
canonical envelope or ZIP digest.

This reads the explicitly selected local snapshot. It does not contact remote
registries, establish current rollover/revocation status in other repositories,
resolve dependency closure or replay a certificate.

Add `nesting_matrices: [SOURCE, ...]` to the request to compare approved smaller
matrices before running the ladder. `check_prefix_nesting` compares decoded
binary points exactly at their native precisions, under the same exact cutoff
and ordered dimensions. The report's `nesting_checks` retain both source
identities, precisions, mismatch counts and first mismatching coordinates.
A mismatch is recorded, never repaired. A match establishes equality of these
stored blocks, not assembly accuracy or a theorem for all N. Observations
retain the largest-parent provenance after a successful comparison.

## Capture policy, scheduling and compatibility

`PrefixAnalysisOptions.diagnostics` and
`CcmCapturePlan::with_prefix_diagnostics` accept `PrefixDiagnosticPolicy`:

```json
{"third_inverse_moment":true,"innovation_cancellation":true}
```

Place this object under `options.diagnostics` in the retained-prefix request.
`PrefixDiagnosticPolicy::full()` selects both. New Ultra plans select it by
default. Set `innovation_cancellation` to false to omit the absolute-value
back-substitution trees; no missing cancellation measurement is interpreted
as zero. Pivot cancellation remains part of the ladder. Omission changes the
child identity, so it cannot silently satisfy a full-capture request.

The numerical entry point `analyze_prefixes_with_policy` accepts the same
policy. The established `analyze_prefixes` entry point retains its two-moment,
cancellation-enabled behavior. Factorization runs first. With multiple workers on sufficiently large
matrices, its columns assign independent lower-factor entries to workers,
while pivots and their stop screens remain in row order. Independent
innovation solves then run in parallel and moments accumulate in prefix order. The factor is released after the solves. Each inner arithmetic
sequence remains unchanged; the default report matches the frozen interleaved
implementation byte for byte, including accepted prefixes before a pivot stop.
Failures in speculative future-row entries are inspected only when that row
is reached, so they cannot pre-empt an earlier pivot stop.

Two-moment/cancellation-enabled requests retain semantics
`ccm-retained-even-prefix-moments-checked-exports-v2`, with the new policy omitted
from their serialization. Extended policies use v3 and include both flags in
their identities. Old serialized plans do not silently acquire third-moment
work: resolve a fresh Ultra plan or set the policy explicitly. Historical v1
and v2 scalar observations remain readable; missing new data is never invented.

Fresh Ultra requests reuse compatible matrix/eigenstate parents and compute
new v3 children when computation is allowed. `RequireReuse` remains strict.
Source, root and retained-reduction identities are unchanged by this extension.
Both public and private shards use their existing generic artifact envelopes;
no new artifact kind or shard schema is required. Large ladders still need
explicit source, precision, memory and runtime planning.
