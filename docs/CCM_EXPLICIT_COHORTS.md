# Optional explicit retained cohorts

`ccm::explicit_cohort` is an additive native Rust API behind `hp`. A caller
must explicitly invoke it after acquisition. Capture levels, including Ultra,
never invoke it, and ordinary primary receipts and historical report identities
are unchanged. There is no cache-directory discovery, primary solve, root
discovery, transform evaluation, or source-generation fallback.

The consumer freezes an ordered `FrozenCohortPlan` with named members and
explicit pairs. Each member binds its eigenpair and optional root window by
artifact kind, semantic digest, payload digest, and canonical manifest digest
when present. `SourceIdentity::from_manifest` creates that binding;
`freeze_member(id, selected_manifests)` accepts an explicitly selected journal
inventory containing one distinct state and at most one distinct root window.
Exact manifest aliases are deduplicated. Multiple states or root windows require
an explicit narrower selection. This helper does not select by nominal C/N/P.

`analyze_frozen_cohort(plan, manifests, cache, resources, scratch_directory)`
uses the existing `ArtifactCacheContext`, exact resolver authentication, and
shared `RunResources`. Pass manifests from the selected inventories, including
available factorization/sector ancestry. Missing direct parents can be resolved
by their exact identities through the supplied resolver. The scratch directory
is created if needed and holds resolver-owned verified streams. Member IDs must
be unique; pairs must be unique even if reversed; every member must participate
in a pair; requested root ordinals are strictly increasing and positive.

The adapter authenticates the matrix ancestry through actual factorization and
sector manifests without opening a matrix payload. It checks persisted semantic
envelopes against their digests, verifies matrix/state C/N/binary precision and
declared Tau identity, requires the supported normalization, and records actual
operator quadrature orders, operator recipe, state solver route, subspace, and
all policy differences. It authenticates state/root/secular payloads and checks
their semantic and parent bindings. A source stamp authenticates retained
evidence; it does not replay the original ground-state proof or independently
certify matrix construction error.

Each pair reports exact rational eigenvalue and retained-root differences,
absolute changes, relative changes against the left absolute value when
nonzero, zero denominators, and sign crossings. State comparisons use directed
MPFR arithmetic for normalized absolute overlap and sign-aligned distance after
signed Fourier zero padding at the same cutoff. The overlap range is explicitly
intersected with the mathematical Cauchy-Schwarz range. Different cutoffs have
no asserted common physical basis. Route/subspace differences remain visible;
single-vector rotation in an unresolved cluster remains qualified.

Root rows preserve every requested ordinal, missing/failed entries, acquisition
route, reference-seed use, source status, and retained completeness. These are
comparisons of retained points. A converged point is not promoted to a certified
root interval, a global kth-root assertion, or a true-zeta enclosure. Certificate
payload consumption and interval-root comparison are not implemented here.
Precision changes that also change the resolved quadrature are explicitly
classified as coupled changes. All other changed policy fields remain listed;
no generic single-axis convergence interpretation is inferred.

Set `embedded_residual: true` only on selected pairs needing the heavier tier.
The left state must have fewer modes at the same cutoff. Only that pair's larger
Tau payload is opened and decoded under the shared resource budget. Directed
arithmetic measures the shifted residual of the normalized embedded smaller
state and splits its squared norm into original-mode and added-mode terms.
The matrix is dropped after the pair. This is a stored-point matrix residual;
assembly/source uncertainty, spectral separation, target-state identification,
analytic state-error bounds, and infinite tails remain unresolved. The low-mode
term includes changed assembly/quadrature and the smaller source's own residual.

Managed reports use the existing private `ccm_finite_diagnostic_analysis`
artifact kind in `ccm-evidence` with the distinct request semantics
`ccm-explicit-retained-cohort-v1`. Existing report keys and payload bytes are
untouched. Ordered source identities, edges, metrics, and residual requests bind
the new request. Member labels, paths, lookup order, elapsed time, resource
availability, and counters do not bind its mathematical identity or numerical
payload. The standard private-only staging boundary also applies under a Both
publication policy. This API does not execute publication. The zero-matrix-read
property describes numerical comparison: author staging can separately decode
dependency payloads through the generic cache fallback when encoded transport
is unavailable. Those staging reads are not included in the comparison's
payload counters.

`CohortExecution` separates the managed `measurements` from `operations` and its
optional artifact manifest. Incomplete source reads are reported without
persisting an incomplete artifact. A replay of a successfully managed request
reuses the report before opening numerical payloads. Operational decoded byte
counts include repeated pair reads; they are not network-transfer measurements.
Peak reservations are cooperative estimates, not measured process RSS. States
are loaded sequentially per pair with at most two lightweight members resident;
there is no automatic retention of an entire campaign's matrices.

`execution_complete`, `requested_coverage_complete`, and numerical acceptance
are separate. Missing root values preserve incomplete coverage even when the
comparison executed successfully. No acceptance tolerance, extrapolated limit,
monotonic-chain test, or convergence predicate is supplied. A future acceptance
policy can consume these exact measurements without changing this request.

`tests/explicit_cohort.rs` qualifies synthetic source authentication, managed
replay and alias independence, zero lightweight matrix opens, requested root
coverage/acquisition, and a directed residual oracle. A producer-backed native
regression also acquires C13/N4/N6/P192 seeded and independent sources under both
fixed-guard and adaptive root policies. It checks the schema-specific semantic
`adaptive_v1` versus payload `adaptive` mapping, rejects authenticated inconsistent
payload policies, and preserves the absent N4 independent root as incomplete
requested coverage. Analysis and replay prohibit dense matrix payload reads.
These tests do not establish large-campaign performance or qualify the historical
retained C13 archive. Consumers must preserve source evidence and scientific
qualifications when applying the API to a real campaign.
