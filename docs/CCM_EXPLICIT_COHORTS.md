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

The adapter first authenticates an explicitly declared numerical Tau identity
against the selected matrix manifest, without opening a matrix payload. Equal
consumed bytes may have different publication provenance or store aliases. Both
current content identities and legacy references are supported; legacy references
also require the declared semantic digest. If no numerical declaration exists,
the adapter requires exact ancestry through factorization and sector manifests.
A malformed or mismatched declaration is rejected rather than falling back to
ancestry. Ambiguous operator policies still require an explicit selection.
Reader-version, quality, immutability and caller visibility gates apply before
payload reads. Exact publication ancestry remains a separate provenance check.
Root windows likewise admit an equivalent secular payload by their declared
content digest, with acquisition, precision, parity and normalization checks.
If that declaration is absent, exact historical ancestry is required. An
invalid declaration never falls back to a publication edge. Full reader policy
also applies to supplied parent aliases, before choosing one for a read.
Combined inventories may contain other publications of the selected state's
values; the frozen state/root manifest selection must still match exactly.
Unselected publications and reader-ineligible aliases do not hide an eligible
selected source. If none remains, the error retains the rejection reason.
The adapter checks persisted semantic
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
The left state must have fewer modes at the same numerical cutoff. Metadata
preflight checks both requirements before opening payloads. Only that pair's
larger Tau payload is decoded under the shared resource budget. Directed
arithmetic measures the shifted residual of the normalized embedded smaller
state and splits its squared norm into original-mode and added-mode terms.
The matrix is dropped after the pair. This is a stored-point matrix residual;
assembly/source uncertainty, spectral separation, target-state identification,
analytic state-error bounds, and infinite tails remain unresolved. The low-mode
term includes changed assembly/quadrature and the smaller source's own residual.

Managed reports use the existing private `ccm_finite_diagnostic_analysis`
artifact kind in `ccm-evidence` with the distinct request semantics
`ccm-explicit-retained-cohort-v4`. The version change gives corrected reports a
distinct identity from historical v1 and v2 reports. The request binds ordered
state, root, and selected Tau payload digests, compared policy digests, pair
indices, requested root ordinals, and residual requests.
When an old native source has only exact ancestry, conflicting parent values
require a narrower inventory to make the frozen selection unambiguous.
Exact selected producer ancestry and the frozen acquisition plan are retained in
private manifest annotations, including the selected matrices. Embedded residual
rows bind the matrix's numerical payload; their bytes are stable across producer
aliases. The reader authenticates both legacy Tau
source references and content-based Tau references, and explicitly admits seeded
root payload schemas 7/8 alongside legacy schemas 3/5/6. Member labels, paths, lookup order, elapsed time, resource
availability, and counters do not bind its mathematical identity or numerical
payload. The standard private-only staging boundary also applies under a Both
publication policy. This API does not execute publication. The zero-matrix-read
property describes numerical comparison: author staging can separately decode
dependency payloads through the generic cache fallback when encoded transport
is unavailable. Those staging reads are not included in the comparison's
payload counters.

`CohortExecution` separates the managed `measurements` from `operations` and its
optional artifact manifest. Incomplete source reads are reported without
persisting an incomplete artifact. The first cache hit in a process checks the
report against its selected numerical sources. A verified repeat can reuse it
without opening those sources again. Operational decoded byte
counts include repeated pair reads; they are not network-transfer measurements.
Peak reservations estimate cooperative working memory. The decoder charges
state coefficients using their count and binary precision before opening the
payload. Each pair retains at most two decoded states.

`execution_complete`, `requested_coverage_complete`, and numerical acceptance
are separate. Missing root values preserve incomplete coverage even when the
comparison executed successfully. A replay flag requires matching state, Tau,
root payload selections, and comparison policies. Cached reports must agree
with the selected policies and their stated root coverage.
No acceptance tolerance, extrapolated limit,
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


A frozen member may name a matrix by its semantic, payload and optional canonical
manifest digests. The selected matrix must satisfy the state's authenticated
operator ancestry. A comparison across distinct root domains is unresolved.
Only converged values on both sides qualify an ordinal for complete coverage.
An incomplete report has no numerical identity. On the first encounter with an
unverified report, the consumer checks its values against the retained inputs.
The process remembers up to 64 verified request and report digest pairs. A repeat
hit can skip that replay while its exact digest pair remains in memory.
Fresh-process verification compares each root row as it is produced and releases
its decimal workspace before checking the next row.
