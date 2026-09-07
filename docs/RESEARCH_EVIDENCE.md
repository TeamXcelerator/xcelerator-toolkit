# Frozen research evidence and capture completeness

These contracts ship in v0.15.0 and analyze retained observations. See
[numerical compatibility](NUMERICAL_COMPATIBILITY.md) for solver identities,
managed reduction reports and visibility rules, and [capture integration](CAPTURE_LEVELS.md)
for application responsibilities.

## From an idea to a reproducible finite test

1. Define the observable with `xc_core::ObservableContract`: definition and
   version, finite/plateau/minimum/extrapolated target, normalization, metric,
   transform/log base, optional root/reference/error convention, prolate index
   and deficiency convention, and derivative coordinate.
2. List the exact planned designs and cutoff families in `HypothesisSpec`.
   Independent-method comparisons use explicitly different `method` and
   `quadrature_identity` fields. Changing an observable's meaning requires an
   explicit derived conversion, rather than silently pooling a column named D.
3. Supply predictions, an absolute tolerance or finite-domain remainder for
   each point, parameter origins, evaluator digest, regime, exclusions,
   partition and known exposure history. Freeze before loading validation
   payloads. Model evaluation/fitting remains in the research application;
   the first scorer consumes a frozen prediction table, not expression strings.
4. Select metadata, then call `score_frozen_hypothesis` with a retained-byte
   loader. It checks the frozen digest, cohort, definition, design, source
   completion and duplicates before requesting bytes. It authenticates each
   loaded observation against its approved metadata digest and checks the
   decoded definition/design again. Unlisted observations are not loaded.
5. Inspect both acquisition completeness and the scientific verdict. A
   successful computation of a score can report a failed hypothesis. Missing
   required points cannot turn a smaller convenient subset into a pass.

The source/target identities in a scalar record are reported provenance, not
an independent replay of the matrix chain. Use the existing artifact resolver
and numerical verifiers to establish those additional claims. The callback
API permits those adapters without a duplicate cache backend.

## Resolution, signs and limits

`ObservableResolution` requires seven explicit components: construction,
working arithmetic, solve, finite N, measurement quadrature, reference and
export round-trip. Each is a declared rigorous/conditional bound, empirical
discrepancy, estimate, unknown or not-applicable, with explanation/provenance.
A higher-precision repeat on an unchanged matrix is not a construction check.
Finite-N uncertainty can be not-applicable for a precisely defined finite
matrix observable; a continuum claim instead needs its own finite-N analysis.

The scorer adds declared absolute components conservatively, using directed
MPFR arithmetic in the observable's units. Shared parents do not create
independent errors. Unknown components never become zero. Empirical envelopes
remain empirical even though their subsequent arithmetic is outward rounded.
Source, original analysis, export and scoring precision remain distinct.

The acceptance gate is frozen and deterministic:

- **pass_on_tested_domain:** the entire declared observation range satisfies
  the finite-domain tolerance;
- **fail_on_tested_domain:** a range is disjoint from that tolerance;
- **unresolved_at_current_resolution:** an unknown component, unresolved sign,
  unavailable required result or boundary overlap prevents a decision;
- **ineligible_data:** definitions, designs, duplicate selection or payload
  authentication do not meet the frozen contract.

A resolved failure can remain a failure when other cases are missing; the
report also marks incomplete acquisition. These verdicts are not statistical
confidence or asymptotic theorems. No RMSE-based winner is inferred from two
indistinguishable corrections. A rigorous-bound label records a supplied
claim/evidence digest; this scorer does not verify its proof or raise source
assurance.

Finite, exact-zero, rounded-zero, unresolved-sign and signed-log values have
separate variants. Signed logarithms retain the original nonzero sign as well
as the log magnitude; a negative log value is not used as the original sign.
An exact original zero has no finite signed logarithm.

`convert_log_units` explicitly converts natural/decimal logarithms and depth
signs. `scaled_correction` encloses `(value-leading)/scale`, preserving loss
of resolution under cancellation. `stabilization_ladder` reports signed
adjacent-N changes, the preceding-change denominator and the signed ratio.
It refuses mixed precision/method/quadrature ladders and leaves zero-containing
ratios unresolved. Two small increments do not become a tail bound.

`quadrature_identity` records the realized quadrature rule identity, including
actual orders. For an N ladder whose order grows with N, use
`stabilization_ladder_with_quadrature_schedule` and supply a
`StabilizationQuadratureSchedule`: a policy identity plus the exact expected
quadrature identity at each N. Every observation is checked against that
schedule; other controls must still match. The serializable result retains the
schedule and observation digests. Its increments measure the coupled N/order
change, so they do not isolate truncation error. The fixed-rule API remains
strict. Neither API silently erases realized quadrature provenance.

## Protected selection and immutable outcomes

A cutoff family cannot straddle fitting and validation partitions. An exact
design cannot be relabelled into an independent family. Duplicate selected
payloads are rejected. Protected configurations are excluded before loading;
known payload aliases of protected metadata are excluded too. Loading the
protected partition requires an explicit flag on the scoring call.

This is an operational safeguard, not proof that a person was previously
blind. Exposure history is supplied explicitly. Saving a digest cannot create
prospective status. A modified model, prediction, gate, selection policy or
regime has a different frozen identity. The scoring semantics are also bound
into that identity; unsupported semantics are rejected.

## Toolkit adapters and capture receipts

`ccm::prefix::prefix_observations` converts an existing report into typed
observations for Schur pivots, innovation mass, inverse-trace increments,
first/second inverse moments, effective inverse rank, newest trace fraction,
and moment-based eigenvalue/depth estimates. It does no factorization or
source acquisition. Each observation retains the report identity and the
largest-parent N. A parent prefix is not relabelled as a separately assembled
canonical matrix. Resolution evidence is supplied explicitly, never inferred
from the number of printed digits.

`CcmCapturePlan::receipt()` enumerates the diagnostic groups requested by the
resolved plan, including both capture phases and individual prefix checkpoint
dimensions. Applications record completed, missing, blocked or failed outcomes.
Completed groups require evidence; a recorded outcome cannot be overwritten
by another attempt. Pending or missing target-dependent work keeps the receipt
incomplete. Importers validate against the expected plan to detect omissions.
The receipt does not launch work, authorize a budget or request certification.
Paper applications still need to adopt this API explicitly.

## Local command workflow

The example is a complete local-file adapter and refuses to overwrite outputs:

```sh
cargo run -p xc-numerics --features hp --example research_score -- \
  freeze private-spec.json new-frozen.json
cargo run -p xc-numerics --features hp --example research_score -- \
  score new-frozen.json inventory.json development 256 new-packet.json
```

The specification schema is illustrated by the wholly synthetic
`crates/xc-core/tests/fixtures/hypothesis-spec.json`. An inventory is a JSON
array of `{ "metadata": ObservationMetadata, "payload": "relative/path.json" }`.
Payload paths resolve relative to that inventory. Each payload is an
`ObservationPayload`. Protected validation uses the `protected_validation`
partition and the explicit `--allow-protected-validation` flag. Outputs include
the frozen spec, inventory, signed tables, resolution provenance and score.
The `freeze`, `score` and `replay` commands perform no remote acquisition or
publication. The explicit `store` command uses managed-cache configuration;
publication still requires the normal author configuration and permissions.

A packet embeds the exact selected observation bytes, including explicit
unavailable-read outcomes, and replays the score from them. Unselected payloads
are never loaded. It does not contain or numerically recompute the referenced
matrix/root sources. Managed persistence requires the exact source manifests
for every successfully loaded observation; their dependency closure is retained.

## Corrected linear-cost tridiagonal route

`tridiag_lu_solve_pivoted_hp` fixes the later-pivot RHS application while using
the existing factorization format. Each row swap is interleaved with its
elimination. Both working storage and per-RHS work are O(n). The inverse
iteration API exposes this as `TridiagSolver::BandedInterleaved` with distinct
`semantics_id()` value `tridiag-lu-interleaved-pivots-v2`.

The historical `Banded` API remains available for deliberate replay. v0.15.0
sector diagnostics and managed prolate selected vectors explicitly select the
corrected route under distinct identities. Other legacy consumers retain their
established route. Correcting the inner solve does not certify branch selection
or the outer iteration's stopping criterion; see [numerical compatibility](NUMERICAL_COMPATIBILITY.md).

Validation covers the exact failing 3-by-3 example, nonsingular integer
tridiagonals with varied pivot patterns, dense-LU comparison, an explicit
inverse-iteration comparison, malformed factors and a 1,000-dimensional solve.
No whole-CCM speedup factor or production-scale qualification is asserted.

## Stable symmetric reduction and retained checks

`householder_tridiag_hp_stable` scales each active column by its maximum
absolute entry, then adds the norm with the first entry's sign when forming
the reflector. The resulting tridiagonal off-diagonal has the opposite sign.
This avoids cancellation in the historical reflector on nearly tridiagonal
matrices. The route identity is `householder-scaled-opposite-sign-v1`.

`dense_symmetric_tridiagonal_hp_stable` performs the same arithmetic on the
matrix without allocating or updating Q. Its tridiagonal is exactly identical
to the with-Q route for fixed inputs. Use it when only spectral estimates are
needed; use the with-Q route to check the transformation or recover vectors.
`dense_symmetric_eigenvalues_hp_stable` composes it with existing tridiagonal
QR. These routines require finite, exactly symmetric stored point matrices
and reject silent down-rounding. They are explicit analysis routes and the v0.15.0 sector diagnostic routes use
them under new identities. Legacy APIs remain for historical replay.

`assess_symmetric_reduction_hp(A, d, e, Q, bits)` checks any supplied reduction,
including legacy data, directly against A. It computes

- absolute similarity residual: `||A Q - Q T||_F`;
- relative similarity residual: `||A Q - Q T||_F / ((||A||_F + ||T||_F) ||Q||_F)`;
- absolute orthogonality residual: `||Q^T Q - I||_F`;
- relative orthogonality residual: `||Q^T Q - I||_F / sqrt(dimension)`;
- the three norms used for normalization.

When the similarity denominator is zero, the absolute residual is retained;
Q=0 still fails the orthogonality check. Scaled streaming sums avoid squaring
raw large/small magnitudes. The checks take O(dimension^3) arithmetic but do
not allocate either dense residual matrix. A small computed residual is a
backward-error diagnostic at the working precision, not a forward eigenvalue,
branch, root, positivity, input-construction, or continuum certificate.

`ccm::prefix::check_retained_reduction` authenticates sources through the
existing `RetainedEvenMatrix` interface, runs the stable reduction and both
checks, and exports a deterministic `RetainedReductionCheck`. Source and
working precision are separate. The report retains the source digest, route,
T, computed eigenvalues, actual residuals, tolerance, and dimension budget.
Q is computed but not saved in this compact report. A higher-precision check
of unchanged entries does not recover missing construction accuracy.

This diagnostic is explicitly requested and O(dimension^3); Ultra does not
launch it automatically. Its dimension budget and an estimated 16-GiB matrix
working-storage screen are checked before decomposition. The estimate covers
the two dense working matrices and per-entry overhead, not total process RSS
or the caller's already-loaded source. It is an operational screen, not an
allocation guarantee. The local function performs no source lookup or
regeneration. The separately named managed wrapper below caches its diagnostic
child without altering an existing source.

### Local retained-matrix command

```sh
cargo run -p xc-spectral --features hp --example ccm_reduction_retained -- \
  reduction-request.json new-reduction-report.json
```

The request is:

```json
{
  "matrix": { "manifest": "matrix-manifest.json", "payload": "matrix-payload.json" },
  "approved_payload_digests": ["<logical-payload-sha256>"],
  "working_precision_bits": 256,
  "maximum_dimension": 64,
  "relative_tolerance": "1e-50"
}
```

Source paths resolve relative to the request. Supply a runtime
`ArtifactManifest` and logical payload, as for `ccm_prefix_retained`; a warehouse
package digest is not a logical-payload digest. Existing outputs are refused.
When either residual exceeds the requested tolerance, the command retains the
failed report and exits unsuccessfully. Missing/invalid sources or an exceeded
budget return an error without starting the decomposition.

Qualification includes opposite leading signs near tridiagonal cancellation,
independent cyclic-Jacobi spectra, original-matrix eigenvector residuals using
the corrected interleaved-pivot solve, power-of-two scaling through exponents
+/-4000, exact output repetition with one/two/four Rayon workers, no-Q identity,
and retained-source/serialization/precision/budget checks. See
[release validation](VALIDATION.md) for actual run counts and scope. No whole-CCM performance factor is inferred.

## Managed retained reduction

`check_retained_reduction_via_cache` accepts the same source, precision, dimension
budget and tolerance as the local check, plus an artifact cache context. Its
child binds the exact source dependency and work policy. It supports required
reuse, refresh and verification without source acquisition. See
[numerical compatibility](NUMERICAL_COMPATIBILITY.md) for warm validation scope and publication eligibility.


## Managed evaluation and replay

`evaluate_hypothesis_packet` uses the same frozen-cohort, duplicate and protected
partition gates as `score_frozen_hypothesis`. The result is a
`HypothesisEvaluationPacket` containing the frozen specification, inventory,
selected observation bytes and complete score table. Its `validate` method
replays the selected reads and numerical verdict, rejecting missing or extra
reads, altered bytes, and altered score fields. Protected-partition evaluation
and replay require the explicit access flag on each call.

`persist_hypothesis_evaluation` accepts the packet, authenticated source
manifests, access flag and cache context. It checks that the manifests match
exactly the sources of successfully loaded observations. An incomplete or
scientifically failed evaluation can still be retained as evidence. The managed
kind is `research_hypothesis_evaluation`; full packets are private-only.

```sh
cargo run -p xc-numerics --features hp --example research_score -- \
  replay packet.json
cargo run -p xc-numerics --features hp --example research_score -- \
  store packet.json source-manifests.json
```

`source-manifests.json` is an array of authenticated runtime `ArtifactManifest`
records matching the observation source references. `store` uses the existing
managed cache and explicit publication controls. It does not acquire replacement
source computations or infer permission to publish from available credentials.

The [capture receipt schema](schemas/research-capture-receipt-v1.schema.json)
and [evaluation schema](schemas/research-hypothesis-evaluation-v1.schema.json)
check portable shape. Toolkit validation additionally checks coverage, digests,
exact dependencies and score replay; JSON shape alone is not scientific evidence.
