# Numerical compatibility and existing artifacts

v0.15.0 separates corrected numerical routes from historical results through
versioned semantic identities. An upgrade does not rewrite stored artifacts or
establish the accuracy of an earlier experiment. The changes below apply to
the library; applications must update their dependencies and rebuild to use them.

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
target-definition digests bind `gaussian-series-relative-geometric-tail-v2`.
External-only targets retain their existing protocol identity; external targets
with Gaussian auxiliary series also bind the new summation semantics. Old
target-dependent cache results remain historical and cannot satisfy requests
using the new definition digest. An unchanged descriptor file therefore has a
new evaluation identity. No original payload is silently relabeled.

The tail comparison is computed floating-point arithmetic, not a directed
interval certificate for the complete evaluator. Cancellation and source
accuracy remain separate concerns. Independent Arb checks verify selected
values, parameters and unequal scales using incomplete-gamma integral tail
bounds; full historical distance and conclusion revalidation remains open.

### Stable cutoff-flow derivative identity

Current u-flow production uses `ccm-u-flow-response-v0.15.1-v4`, with reader
floor 0.15.1. The stable gamma difference used in Archimedean derivatives
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

The ordinary legacy `Banded` and Householder numerical APIs remain available
for deliberate historical replay. Current sector diagnostics and managed
prolate selected-vector computations explicitly select `BandedInterleaved`.
Preserving an old API is not an accuracy endorsement of that route.

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

The read-only metadata inventory can identify the known cutoff-free
zero-mode certificate semantics and their recorded descendants, and count
artifacts requiring new identities under the v0.15.0 sector/distance routes:

```sh
python tools/ccm_artifact_impact.py /local/shard-a /local/shard-b \
  --registry /local/registry --verify-manifest-bytes \
  --output /local/audits/upgrade-impact.json
```

Supply every relevant dependency shard, including rollover shards and public
parents of private children. Exit status 1 means changed active identities,
affected or unrecognized active certificate semantics, or incomplete metadata coverage;
status 2 means the inventory could not run. The optional byte check hashes
manifest files only. This tool does not check package bytes, recompute numerical
results, or establish which old numerics were inaccurate. An unflagged artifact is not
a validated numerical result.

`active_recompute_count` and `active_recompute_by_kind` describe upgrade work;
`active_affected_count` describes the known certificate defect. These are
different questions. Missing dependency manifests can undercount descendants.
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

### Root-response normalization

The root-normalization correction introduced
`ccm-prime-power-response-v0.15.0-v3` and
`ccm-u-flow-response-v0.15.0-v3`. Current cutoff-flow production additionally
uses the stable-derivative v4 identity described above. Payload schema 2 and the artifact kinds are
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
or artifact kind. The impact inventory counts older response identities and
their descendants under `active_recompute_count`; its separate
`active_affected_count` remains specific to the certificate defect rule.

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
for its residual definitions and [release validation](VALIDATION.md) for the
tested scope.

### Extended prefix policies

New Ultra plans capture the third inverse moment under prefix semantics v3.
The optional cancellation flag is part of the same explicit diagnostic policy.
Legacy two-moment/cancellation-enabled requests retain their v2 identities and
payload bytes through the phase scheduling change. Neither policy changes
source, root, eigenstate or retained-reduction identities. Old serialized plans
keep their old requests; fresh Ultra plans or explicit policy selection request
new children. The impact inventory reports older prefix children as candidates
for the new-data backfill, not as evidence of corrupted numbers.

### Capability-dependent retained diagnostics

Transform-enclosure and band-reconstruction requests bind the availability of
the Arb backend into their semantic cache identity. A build without Arb can
produce a qualified feature-required absence; that record cannot satisfy a
later Arb-enabled request. Legacy requests without a capability declaration
also require a fresh calculation under the new identity. Historical bytes are
preserved. This identity repair does not invalidate an independently verified
finite enclosure or claim that a missing result contained a false certificate.
