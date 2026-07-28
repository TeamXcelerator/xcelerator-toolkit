# v0.13.3 Cache Schema Principles

This document is the concise schema contract used by manifests and validators. TD-04 defines the complete resolver, transport, transaction, trust, durability, and recovery design.

## Artifact granularity

An artifact is the smallest expensive result with an independently meaningful semantic key, dependency set, validation rule, durability policy, and reuse boundary. Examples include a quadrature rule, compact archimedean integral primitives, an expensive prime-component matrix, an assembled matrix, a parity-reduced operator, a factorization, a selected eigenpair, a CCM spectral source, a bounded root range, a completed spectral window, a solver checkpoint, or a certificate bundle. Cheap prime enumeration and analytic pole data remain properties of their consuming artifacts; they are not separate remotely fetched objects.

A request is expanded into an artifact-plan DAG before execution. Every node declares its exact dependencies and admissible load, derive, recompute, or certify candidates. Planner policy supplies a complete deterministic preference order over trust, precision, locality, transfer cost, recomputation cost, and action preference, plus cumulative resource ceilings. Planning proceeds dependency-first, rejects unavailable, revoked, incompatible, under-precision, under-assurance, dependency-incomplete, disabled, or over-budget candidates, and records every reason. The plan identifies the selected route for each satisfiable node and remains an explainable nonmutating forecast; execution revalidates the selected identities and policy at use time.

## Four identity layers

The schema never overloads one hash with multiple meanings:

1. **Semantic identity** hashes the canonical mathematical request, algorithm semantics version, and material configuration.
2. **Canonical payload identity** hashes the logical unencoded payload manifest, including ordered file or stream digests and dependency identities.
3. **Transport identity** hashes a specific encoding and its ordered byte objects, such as deterministic ZIP/ZIP64 metadata and split parts.
4. **Attestation identity** hashes a validation, certification, publication, revocation, or receipt statement about another identity.

One semantic artifact may have multiple payload revisions only when their semantic-equivalence policy permits it, and one canonical payload may have multiple transport encodings. Byte objects are immutable and addressed by SHA-256. Derived indexes and ledgers may be rebuilt from canonical manifests, Git reachability, and attestations.

## Canonical manifest

An artifact manifest includes at least:

- schema, artifact family, and semantic versions;
- the complete canonical semantic-key envelope and its verified digest;
- resolved mathematical-configuration digest;
- the complete canonical logical-payload envelope, its verified digest, ordered logical items, logical byte count, and exact dependency identities;
- ordered transport encodings and object digests;
- exact dependency semantic, manifest, and payload identities;
- mathematical claim scope and assumptions;
- producer toolkit version, requested assurance, and reader compatibility;

The canonical payload envelope records scalar backend, optional exact precision in bits, scalar representation, dimensions, endianness, special-value encoding, deterministic logical items, and exact dependency identities. The producer toolkit version is identity-bearing manifest metadata and is repeated in its digest-bound attestation. Dependency versions, execution fingerprint, source revision, actor or creator, event time, location, policy, achieved assurance, validation evidence, publication facts, and public-sanitization evidence live in linked attestations or state records. Two runs from the same producer version can therefore produce different attestations for the same semantic, payload, and manifest identities.

One central compatibility policy supplies the minimum producer version, reader range, and accepted manifest schemas for every artifact family/kind pair. Floors are independently adjustable by kind: a defect in tau construction can invalidate `ccm_tau_matrix` and its dependent closure without invalidating unaffected quadrature or prolate artifacts. Raising a producer floor makes older entries inadmissible cache hits, so normal execution treats them as misses and recomputes them. The same policy is shared by managed remote resolution and every direct numerical cache reader.

Publication is producer-version monotonic for each semantic identity. Before payload upload, the publisher reads the current shard index and rejects an incoming producer older than any active discoverable producer. The atomic discoverability phase repeats the check against the latest repository head, closing the concurrent-publication race. Newer producers may publish alongside immutable older history; consumers order compatible candidates by producer version first and assurance second. Revoked or quarantined entries do not block a replacement.

Certificates, validation reports, and publication-ready exports are separate ordinary artifacts. A canonical `ArtifactLinkSet` attaches any number of their exact semantic, manifest, and payload identities to a numerical subject without embedding report bytes in its payload or changing the subject identity. Link roles and identities are uniquely ordered and digest-bound.

Unknown required fields, ambiguous semantics, missing dependency closure, revoked evidence, or untrusted attestations fail closed.

The manifest is deliberately self-contained for identity rebuild and decoded validation: a reader recomputes both the semantic digest and canonical payload digest from their embedded canonical envelopes. A digest-only payload pointer is insufficient because it would allow transport verification but not verification of decoded paths, sizes, content hashes, or dependencies after a selective remote fetch.

## Independent state axes

Artifact state is a product of independent axes, not a single promotion ladder:

| Axis | Representative values |
|---|---|
| Completion | planned, in-progress, partial, complete, failed, abandoned |
| Assurance | unchecked, structurally validated, computed, cross-checked, certified |
| Disposition | active, deprecated, quarantined, revoked |
| Storage location | process, workstation, project-private, team-private, public, export bundle |
| Publication target | planned, uploading, batch-verified, remote-verified, receipt-complete, failed, abandoned |
| Durability | ephemeral, local-durable, remote-single, mirrored, archive-pinned |
| Compatibility | current, quarantined, rejected |
| Trust | unverified, trusted, revoked, expired |

Structural and numerical validation details, requested assurance, achieved assurance, and rejection reasons are typed evidence linked to the assurance axis. Changing location does not increase assurance. Publication does not imply certification. A certified artifact may remain private, and a correctly labeled computed or cross-checked artifact may be public when policy permits. Policy evaluates the necessary axes for the requested operation.

## Dependencies and resolution

A manifest names exact dependency identities and the minimum evidence required of each. Resolution searches the configured overlays, normally local, private, then public, but accepts the first artifact satisfying all requested semantics, schema, dependency, assurance, trust, revocation, durability, and execution-compatibility policy. A higher-precedence artifact may be skipped for an admissible lower-precedence artifact.

Every declared artifact family/kind pair must have exactly one `ArtifactValidatorRegistration`. Registry coverage validation fails when a declared type is unregistered. `validate_for_reuse` emits ordered evidence for all seven mandatory facets: schema versions, canonical structure, artifact-specific mathematical invariants, decoder-observed backend/precision/dimensions, exact dependency identities, decoded logical-item sizes/hashes, and toolkit-reader compatibility. Reuse is permitted only when every facet passes; invariant callbacks must return content-addressed evidence, and each failed facet retains a specific reason.

Contributor-reviewed publication additionally requires typed `ContributorPublicationEvidence`. It names the contributor and owner authorizer, records whether the grant is written authorization or an approved contributor agreement, scopes that grant to exact destinations and repositories, identifies the fork or non-main contribution branch and immutable 40-hex head, and links a nonzero pull-request number plus authorization and pull-request evidence digests. Reviewer approvals count only when made by a different principal and bound to that exact pull-request number and head. Owner-direct publication retains its no-pull-request path under the existing owner policy and live target write-permission proof.

Each target journal receives immutable `TargetPublicationAuditEvidence` before execution begins. The per-target receipt records the authenticated actor and permission evidence, authority mode, policy ID and digest, target visibility/repository/shard/branch, semantic/manifest/payload/transport identities, validator evidence, optional contributor authorization, full reviewer approval records, payload and batch-record commit IDs, metadata commit, exact remote-verification results, and verification/publication time. Reviewer approvals may be empty only for owner-direct authority. Because a receipt cannot name the Git commit that contains itself, `build_publication_inventory` joins remotely completed receipts with the final journal and adds the discoverability commit, receipt digest, all commit IDs, and explicit success/failure state for every requested target.

Resolution validates the entire dependency closure before returning publication-grade or Certified data. A stale or corrupt derived index can cause a miss but cannot make an invalid artifact admissible.

Fast validation proves trusted metadata and exact dependency identities without payload staging and is never sufficient for Certified consumption. Full validation walks the resolved graph dependency-first, de-duplicates exact semantic/manifest identities, rejects cycles, selectively retrieves every artifact transport, reconstructs each package, and verifies deterministic encoding plus every decoded logical item and canonical payload identity. The result explicitly records every validated dependency and closure completion.

Every lookup also produces a `CacheAccessProvenance` record suitable for attachment to `SolverProvenance`. It retains the operation and semantic identity, configured overlay order, hit or miss, reuse or recomputation decision, exact selected repository/revision/document paths, all rejected candidates and reasons, validation mode/outcome/detail, and the identities validated by full closure materialization. These records are operational provenance and never alter canonical mathematical identity. `cache_duplication_audit` deterministically reports every semantic identity recomputed despite an admissible hit.

Deduplication planning is backend-local and privacy scoped. `plan_deduplication` accepts only inventories the caller has already been authorized to observe and never performs discovery. Reports keep logical payload reuse, exact-target-repository physical reuse, cross-repository copies, and new uploads as separate byte/object totals. Matching bytes in another GitHub repository remain a copy, never a physical-dedup claim. Public planning structurally rejects private, team, or local target inventories and observations, and reports zero private-store probes.

## Registry and shard ownership

The public and private top-level registries contain topology only: artifact-family routes, ordered shard candidates, shard status, successor relations, and policy/trust anchors. They do not enumerate every semantic key, object hash, or routine data-repository change.

Each shard owns its artifact inventory, partitioned semantic indexes, manifests, encodings, objects, validation and publication attestations, revocations, capacity ledger, transaction journals, and receipts. Normal artifact publication updates only the selected shard. The registry changes when topology changes, such as shard creation, retirement, or family rerouting.

## Repository layout

```text
shard.json
schemas/
indexes/<family>/<partition>.json
manifests/<semantic-prefix>/<manifest-digest>.json
encodings/<payload-prefix>/<encoding-digest>.json
objects/sha256/<prefix>/<object-digest>.part
attestations/validation/<digest>.json
attestations/publication/<digest>.json
revocations/<digest>.json
revocations/indexes/<identity-prefix>.json
ledger/capacity.json
transactions/<transaction-id>/plan.json
transactions/<transaction-id>/batches/<sequence>.json
transactions/<transaction-id>/receipt.json
```

Partition indexes are replaceable projections. Canonical manifests, immutable objects, signed attestations, revocations, and committed receipts are the reconstructable source material.

Every shard-index entry names the stable publication transaction that introduced it. A resolver uses that identity to fetch `transactions/<transaction-id>/receipt.json` and verifies that the same atomic discoverability commit binds the index partition, manifest, and transport record. An index entry without a matching canonical receipt cannot satisfy a cache lookup.

Dependency identities contain the dependency artifact family in addition to exact semantic, manifest, and payload digests, allowing bounded recursive resolution through family-only topology. Revocation lookup uses two-hex-digit identity partitions. Records are canonically ordered by scope and identity, retain reason, effective time, replacement, incident reference, and authorizing-evidence digest, and are evaluated before a candidate is returned.

Ordinary corrections use separate two-hex-digit supersession partitions under `supersessions/indexes/`. Each immutable edge binds the exact prior and replacement semantic, manifest, and payload digests together with its reason, effective time, and authorizing evidence. Both manifests must already exist at the same exact shard revision before an edge can be planned. Canonical merging is idempotent but rejects changing an established edge. Consumers can follow a bounded, cycle-checked chain without changing or losing the original pinned identity.

## Deterministic encoding and hard size rules

Logical payloads are streamed through a deterministic ZIP/ZIP64 encoder and byte-split without holding the full archive in memory. Encoding metadata fixes entry order, normalized paths, timestamps, permissions, compression method and level, ZIP implementation version, and split size.

Project hard rules are:

- every Git-managed file is strictly below 100 MB;
- the default byte-split part size is 90 MiB;
- no publication batch, commit, or push introduces more than 1,000,000,000 new payload bytes;
- a shard cannot accept a transaction when projected reachable repository payload would exceed 100,000,000,000 bytes.

Large placement and materialization preflight uses a separate canonical cost-governance document. Each candidate declares logical size, backend-local unique addition, transfer size, destination, retention class, operational suitability, and a digest-bound currency/storage/transfer estimate. Policy enforces explicit quotas and allowed retention, with GitHub public/private storage first while free and suitable. Paid external storage requires approval bound to the exact backend, cost estimate, governance-policy digest, approver, cost ceilings, justification, and evidence; overriding an available suitable GitHub destination must be authorized explicitly.

The publisher verifies every part digest and reconstructs and verifies the canonical payload before making an index entry visible.

## No-full-clone remote access

Resolvers and publishers use GitHub repository/ref, tree, blob, and commit operations or an equivalent bounded transport. They fetch only topology, relevant index partitions, manifests, encodings, receipts, revocation projections, and selected objects. Neither reading nor publishing requires a persistent full shard clone. A selective materialization retains independently verified parts in a local content store, reconstructs the package atomically, and verifies every decoded logical item against the canonical payload envelope embedded in the manifest. A valid existing package can be reused without a remote object read.

All reads and mutations are bounded, cancellable, retryable, and hash-verified. Local temporary state is limited by the resolved resource profile.

`package_and_stream_split_zip64` is the bounded production boundary between canonical logical files and an upload/checkpoint sink. The deterministic ZIP writer uses one seekable temporary archive under the temporary-disk budget; the splitter then holds at most one configured-size part in memory and synchronously transfers ownership to the sink. The packager never retains split parts, so an upload implementation need not hold the archive and a second complete part set simultaneously. The archive is removed on success, cancellation, or sink failure. A sink acknowledges a part only after durable immutable upload/checkpoint; already acknowledged parts remain resumable but cannot become discoverable until the later metadata/index/receipt phases complete.

Every remote overlay and publication route carries a `RemoteFabricTrustPolicy` in addition to topology generation/digest pinning. It allowlists trust anchors and policy digests and names exact registry/shard repository, owner, branch, and revision rules. A rule either pins one 40-hex revision or carries a digest-bound ancestry statement naming the trusted ancestor and observed descendant. A separately digest-bound protected-branch statement must match that same repository, branch, and head; prohibit force pushes; restrict writers; use an approved trust anchor; and remain current at the request evaluation time. Resolution verifies the registry root before reading topology, then its policy, then every shard head. Publication planning and every resume revalidation apply the same checks before capacity reads or mutation.

## Offline and archival bundles

An export bundle is a portable directory-local cache overlay. Canonical `bundle.json` declares a uniquely ordered root set, every exact dependency-complete artifact record, achieved assurance, disposition, self-contained manifest, selected transport encoding, and optional origin receipt. Original manifest-directed part paths are retained beneath `parts/`, preserving transport identity; duplicate content may use local hard links without changing logical bytes. Bundle identity is the canonical manifest digest and excludes its filesystem location.

`resolve_semantic_artifact` is the backend-neutral read boundary. It takes one `RemoteSemanticQuery` and an explicitly ordered set of local-filesystem, export-bundle, private-GitHub, and public-GitHub sources. Every source also carries exactly one ordered policy class: process-local, workstation-local, project-private, team-private, public-published, or optional-remote. The resolver rejects a class-order inversion, retains both the complete source-name order and class order in its report, and reads through independent roots/repositories without copying them into a common cache. Local canonical directories use the verified `bundle.json`/`parts` layout, so they retain the same four identity layers as disconnected bundles and remote shards. The normalized result always returns overlay class, source kind/name/location, immutable backend revision, semantic and manifest digests, assurance, disposition, and the canonical manifest; changing storage backend cannot change the semantic key. Backend validation failures become structured source rejections, while cancellation remains terminal.

The semantic query carries typed consumption minima in addition to identity: achieved assurance, allowed scalar backends, minimum precision, optional exact resolved-configuration digest, required receipt-linked provenance evidence digests, accepted publication policies, disposition permission, and toolkit reader compatibility. The resolver checks representation/configuration immediately after the canonical manifest and provenance requirements immediately after the canonical receipt, before transport materialization. An absent precision is below every positive minimum. Certified requests still require full dependency-closure payload validation.

Export is streaming, resource-bounded, cancellable, and atomically visible. It rejects missing dependencies, artifacts outside the declared root closure, dependency cycles, conflicting path identities, symlinks, corrupt part or package bytes, and manifest/encoding/receipt drift. Transport verification hashes every part and concatenated package. Full verification also reconstructs each package in caller-provided scratch space outside the immutable bundle and validates deterministic ZIP metadata and every decoded logical item against the canonical payload envelope. Offline consumption rechecks the bundle, reader compatibility, assurance, disposition, and dependency closure, then fully validates the selected artifact without any remote lookup. Quarantined and revoked entries remain archivable but cannot be consumed.

Local pruning is plan-first and fail closed. A candidate must use a local location class and name one exact regular non-symlink file by path, length, and SHA-256. Policy evaluates age, reachability, pins, active transactions, assurance, disposition, recomputation cost, receipt-complete required targets, verified copies, failure-domain independence, and archive requirements after excluding that copy. Confirmed execution recomputes the plan, streams and verifies the unchanged file with cancellation polling, and only then removes it. Remote locations are structurally invalid prune candidates.

## Transactional publication

Publication requires an explicit resolved request naming `private`, `public`, or `both`, plus authenticated principal, authority, validation policy, durability, and cleanup policy. No flag means no remote mutation.

Repository deployment names follow `xcelerator-cache-<visibility>-registry` for registries and `xcelerator-cache-<visibility>-<family-id>-<sequence>` for shards, where visibility is `private` or `public` and sequence is a four-digit, one-based number beginning at `0001`. This grammar is validated during bootstrap and rollover. It is not a routing shortcut: readers and publishers still obtain the exact repository from the trusted visibility-specific registry, and private and public shard histories and writable pointers remain independent.

Owner-direct publication may automatically validate and publish when authority and policy permit. Contributors stage privately and use a review branch or fork plus approval attestations for public publication. Public manifests pass an allowlist sanitizer; credentials, private locations, unpublished inputs, and disallowed metadata are rejected.

The transaction protocol is:

1. resolve policy and authority;
2. freeze semantic manifests and dependency closure;
3. validate, sanitize, encode, split, and hash locally;
4. for a private destination, atomically acquire that physical shard's renewable publication lease;
5. re-read the exact shard head, live indexes, and capacity ledger after lease acquisition, then reserve projected capacity against that state;
6. upload immutable payload objects in batches containing no more than 1,000,000,000 new payload bytes, updating the ledger to the exact committed batch boundary;
7. publish the immutable manifest, transport record, validation attestations, repository-level transaction record, and live index only after every referenced object is reachable;
8. for a private destination, atomically update the cache branch and renew the lease in the same Git push; for a public destination, retain expected-head compare-and-swap;
9. verify the committed files and end-to-end resolution from the accepted ref;
10. release the private lease and temporary state according to durability policy.

Transaction IDs and batch digests make retry idempotent. Existing identical objects are reused. Concurrent ref changes cause re-read, revalidation, and retry; conflicting semantic publication fails visibly. For dual publication, private and public have separate journals and receipts. One target may succeed while the other remains resumable or failed, and that partial outcome is never collapsed into success.

### Private-shard publication coordination

Every writable private shard owns an orphan `xcelerator-coordination` branch. It contains only `coordination/state.json` and, while a publisher holds the shard, `coordination/publication-lock.json`; it does not share history with or add lock commits to `main`. The first private publisher initializes this branch automatically with an atomic create-if-absent operation.

One lease covers one physical private repository. Different private shards therefore publish concurrently, while configurations targeting the same shard wait with bounded exponential backoff and visible owner, generation, and remaining-lease status. The lock records a non-secret run identity, authenticated principal, toolkit version, hashed instance identity, process ID, observed cache head, acquisition/heartbeat/expiry times, and a monotonically increasing fencing generation. It never records credentials, local paths, or raw machine account names.

Acquisition, renewal, takeover, and release are compare-and-swap state transitions on the coordination branch. A crashed publisher's lease becomes eligible for takeover only after its expiry and clock-skew grace period. Every private repository batch uses one atomic two-ref push: `main` advances to the batch commit and `xcelerator-coordination` advances to the renewed lease commit. If either expected head changed, Git accepts neither update. A stale publisher therefore cannot advance the cache or ledger after another server takes over.

The capacity ledger is committed with every cache batch and represents exactly the new immutable and metadata bytes reachable at that boundary. Identical existing paths are omitted from retries, a different payload at an immutable object path fails closed, and live metadata is ordered after payload objects so an interrupted transfer cannot expose an index entry whose objects are absent. Release removes the active lock document but preserves the generation and last completed transaction in the coordination state. Private coordination needs no additional repository, PAT, workflow, or consumer configuration.

## Capacity, rollover, and rebuild

The shard ledger reports logical artifact bytes, unique reachable object bytes, projected new bytes, reachable Git payload, estimated history/overhead, reserve, and remaining safe capacity. Admission is serialized against the current ref so concurrent publishers cannot both consume the same reserve.

When admission would exceed the hard ceiling or governance requires separation, an authorized topology transaction creates and validates a successor shard, updates the registry route atomically, and marks the old shard read-only. A failed rollover leaves the old route valid.

Successor activation is fail closed. `SuccessorShardReadinessEvidence` binds repository ownership and visibility, branch protection, trust metadata, a read/write health check, reviewer approval, and a clean exact-revision audit. The successor must be empty, have no audit issues or unreferenced state, and expose a canonical initial ledger with the approved 100,000,000,000-byte hard capacity and complete history coverage. The registry replacement increments generation, binds the prior topology digest, marks the old writer read-only, and activates only its next-sequence declared successor. Registry mutation rechecks both head and path digest and uses one remotely verified compare-and-swap commit.

Index and ledger rebuild scans canonical manifests, encodings, objects, attestations, revocations, receipts, append-only payload-batch records, and Git reachability. The bounded ordinary audit enumerates tree paths without downloading payload blobs, validates every batch record and receipt binding, reconstructs complete discoverable index entries, and reports missing or unreferenced paths. When every reachable payload object has one consistent first-introduction record, it reconstructs exact first-seen bytes—including incomplete transactions—and compares them with the ledger's completed-plus-abandoned payload accounting. Missing, corrupt, or conflicting coverage is reported explicitly and never used to guess reclaimed capacity. The rebuilt projection must reconcile with remote hashes before writers resume.

Index repair is a distinct explicit transaction. A canonical bounded plan names the audited revision, each observed index digest, each reconstructed replacement digest and byte count, and whether entries would be removed. Entry removal and repair with unresolved integrity errors require separate policy authorization. Execution requires explicit confirmation, fresh write permission for the network-registry-bound endpoint, unchanged observed digests and branch head, one compare-and-swap commit, and remote verification. Ref conflict invalidates the plan. Capacity-ledger repair is not inferred from index repair and may proceed only when the audit proves every accounting component required by the selected ledger policy.

## Durability, trust, and revocation

Artifact plans declare one typed durability class: recomputable, expensive-reproducible, irreplaceable-source, or publication-or-certificate-record. The family policy explicitly states the minimum verified-copy count, minimum independent failure-domain count, allowed locations, and whether an archive copy is mandatory. Each copy carries payload identity, locator, failure domain, verification evidence, revocation state, and an immutable publication or placement/deposit receipt for remote locations. Multiple repositories in the same failure domain count as multiple copies but only one independent domain.

Local cleanup planning removes the candidate copy from the evidence set before assessing durability. It also evaluates age, pins, dependency reachability, assurance, disposition, recomputation cost, active transactions, and receipt completion for every requested publication target. Cleanup may remove local working copies only after all gates pass. Essential research artifacts retain immutable archive evidence and cannot rely only on a mutable live index. Remote pruning is never implicit.

Trust policy verifies principal, authority scope, signature, attested identity, repository/ref binding, toolkit and schema policy, and revocation state. Clients pin or monotonically advance signed registry and shard state to resist rollback. Revocation is an immutable signed overlay; objects are not rewritten. Emergency repair publishes corrected manifests or indexes plus an audit attestation.

Author refresh and replacement are explicit operations. `XC_CACHE_MODE=refresh` bypasses every local and remote candidate but retains normal validation, local storage, dependency recording, packaging, and publication. When `XC_PUBLISH_REPLACE=true` is combined with author mode and enabled publication, each discoverability commit removes prior entries for the affected semantic identity and installs the fresh entry as its sole current selection. After all artifacts are published, the toolkit audits each exact affected shard revision and issues a separate compare-and-swap cleanup commit removing only unreferenced current-tree manifests, transport encodings, and payload objects. Shared objects and historical receipts remain. Capacity is never decremented because the removed blobs remain in Git history.

Ordinary author publication is destination-aware. Before obtaining a private publication lease or staging repository files, the publisher reads each required destination index partition once and confirms exact active artifacts against their canonical repository-batch records. Confirmed artifacts are excluded from the new batch without reading their manifests, encodings, attestations, ZIP parts, or payload objects. Their index entries and original publication transaction identifiers remain unchanged. Only missing or no-longer-proven artifacts enter the new transaction; a destination for which every candidate is already proven produces no shard commit and no coordination-branch lock activity. `XC_PUBLISH_REPLACE=true` deliberately bypasses this no-replay behavior.

## Domain reuse

CCM preserves the full assembled tau matrix as the first and fastest ordinary cache hit. On a full-matrix miss it independently resolves compact alpha/beta/gamma archimedean integrals and the expensive prime-component matrix, reconstructs the analytic pole and archimedean matrices, assembles and validates tau, and records the exact component dependencies. The default `even-sector` policy retains every established v0.13 identity and reuses an orthonormal even-sector matrix plus its solver-specific LU factorization. The `natural` policy uses the full matrix without projection. The `adaptive-even` policy also uses the full matrix but applies the historical conditional even projection during inverse iteration. Natural and adaptive work may share the same full-matrix factorization, but their selected eigenpairs and all eigenpair-derived artifacts are semantically disjoint. Explicit sector research additionally derives the historical odd basis `(e_k-e_-k)/sqrt(2)`, computes independently addressable low even and odd spectra, and derives GapLog from those exact spectrum artifacts. Selected natural, adaptive-even, and even-sector ground states, parity spectra, secular sources, configuration evidence, and optional certificates remain independent artifacts.

The concrete CCM execution graph and shard placement are:

| Artifact | Family | Runtime role |
| --- | --- | --- |
| `ccm_archimedean_integrals` | `ccm-components` | Reusable compact alpha/beta/gamma source data |
| `ccm_prime_component` | `ccm-components` | Reusable expensive prime-component matrix, including prime-content metadata |
| `ccm_tau_matrix` | `ccm-matrices` | Preferred full-matrix hit and assembly boundary |
| `ccm_even_sector_matrix` | `ccm-matrices` | Orthonormal parity reduction used by forced-even solves |
| `ccm_odd_sector_matrix` | `ccm-matrices` | Historical orthonormal odd-parity reduction used by sector research |
| `ccm_factorization` | `ccm-matrices` | Full or even LU factors used directly by inverse iteration |
| `ccm_weil_eigenpair` | `weil-states` | Natural, adaptive-even, or even-sector selected state |
| `ccm_sector_spectrum` | `weil-states` | Ordered low eigenpairs for one explicitly named even or odd sector |
| `ccm_secular_source` | `ccm-roots` | Stable identity for the normalized eigenpair-derived secular equation |
| `ccm_root_count_window` | `ccm-roots` | Exact finite-source root count for one rational height window |
| `ccm_root_discovery_window` | `ccm-roots` | Reference-free computed or certified root window with explicit discovery provenance |
| `ccm_root_refinement` | `ccm-roots` | Bounded, one-based indexed root range with exact seeds and solver policy |
| `ccm_spectral_window` | `ccm-roots` | Reference-free discovered window with count reconciliation and optional root certificates |
| `ccm_convergence_diagnostics` | `ccm-evidence` | Per-configuration result and convergence summary |
| `ccm_validation_record` | `ccm-evidence` | Natural-versus-forced evenness evidence |
| `ccm_sector_gap` | `ccm-evidence` | Even/odd ground-state depths, `GapLog=D_even-D_odd`, direct difference, ordering, and simplicity margin |
| `ccm_post_discovery_comparison` | `ccm-evidence` | External-reference comparison attached only after independent discovery is complete |

Cheap prime enumeration and the analytic pole formula are embedded in the semantic identity or payload metadata of their consumers. They are deliberately not fetched as separate objects. Every matrix or state cache hit is checked against its exact dependency: parity matrices are reconstructed from tau, factorizations pass a deterministic solve residual, and eigenpairs replay the tau residual and their structured inverse-iteration stopping evidence. That evidence records the configured limit, steps used, unshifted convergence, final Rayleigh change, shifted-refinement outcome, and final relative residual. Root ranges must be finite, positive, ordered, and identity-bound, and evidence values must equal the results produced by their dependencies.

Roots are stored as bounded ordered windows, not one remote object per scalar root. Independent discovery and reference-seeded refinement are different artifact kinds and cannot satisfy one another's semantic requests. An independent range records its one-based bounds assigned by cumulative finite-source counts, discovery/count methods, `reference_seeds_used=false`, each value's explicit convergence status, per-root iterations, final MPFR correction, residual, achieved-digit estimate, the 64-bit requested-accuracy guard policy, and exact secular-source dependency. A refinement range records its exact supplied starting points and `reference_seeds_used=true`. Cache replay recomputes the achieved-digit estimate and secular residual and rejects inconsistent status claims. Computed assurance retains finite, ordered `stagnated` and iteration-limited `approximate` values without relabeling them as converged; a failed outcome with no value invalidates the window. Cross-checked or certified assurance requires every root to be fully converged. The run-evidence artifact carries the same mode, index bounds, exact root dependency, and separate outcome counts. This keeps research data directly accessible without allowing child-object counts or remote lookup overhead to dominate the numerical calculation.

Ordinary positive independent windows retain the v6 semantic identity, JSON
shape, and reader floor established by earlier v0.13.x releases. Advanced
signed or finite-shortfall requests use v7 numerical artifacts with a
v0.13.3 reader floor. A v7 numerical artifact is identified by its actual
finite-source root domain and complete discovered window, never by the
caller's requested count or permission to accept a shortfall. The requested
count, returned projection, selected canonical ordinals, and shortfall policy
are stored in a separate compact convergence-evidence artifact. A larger
canonical advanced window can therefore satisfy a contained request without
duplicating numerical payloads, while an older reader continues to resolve
only the unchanged v6 artifacts it understands.

The exact FLINT numerator certificate proves the roots of the stored point-valued finite secular source. It must not be relabeled as an end-to-end interval certificate for the mathematical finite CCM operator unless Tau uncertainty, selected-eigenvalue simplicity and gap, eigenvector error, normalization, and residue intervals have all been propagated. Artifact claim scope records this distinction; an exact stored-point surrogate remains valuable rigorous software evidence but is not the stronger finite-operator claim.

The Tau/Weil spectrum and the roots of the eigenpair-derived secular equation are separate mathematical objects. The ordinary CCM reproduction route uses the lowest even Tau eigenpair as the secular source; higher Tau eigenpairs are not interpreted as successive zeta zeros. Extending a finite calculation beyond the first root prefix therefore means requesting additional indexed secular-root windows while increasing `N`, precision, and cutoff according to convergence evidence. No finite artifact is labeled as containing “all zeros.”

Sector GapLog is also separate from a secular-root gap. For the lowest algebraic eigenvalue in each parity block, the stored definition is `D_even=-log10(abs(lambda_even))`, `D_odd=-log10(abs(lambda_odd))`, and `GapLog=D_even-D_odd=log10(abs(lambda_odd)/abs(lambda_even))`. The raw difference `lambda_odd-lambda_even`, its sign, and `-log10(abs(lambda_odd-lambda_even))` are retained as distinct fields so downstream research cannot silently substitute one definition for another. At least the first two eigenpairs per sector are retained so the even ground-state simplicity margin is independently available.

Mk assigns separate artifacts to exact moments, basis and symmetry data, transformations, dense fixtures, matrix-free or structured operators, approximation bounds, adaptive-space histories, checkpoints, candidates, and quotient certificates. Corrected construction semantics invalidate only their dependent closure.
