# Cache schema principles

This document is the public schema contract implemented by manifests,
validators, resolvers, transport, transactions, trust policy, durability, and
recovery behavior.

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

Logical payloads are streamed through a deterministic ZIP/ZIP64 encoder and byte-split without holding the full archive in memory. Encoding metadata fixes entry order, normalized paths, timestamps, permissions, compression method and level, ZIP implementation version, ZIP64 behavior, and split size. The unchanged single-entry workstation/publication encoder retains profile V1 and remains byte-for-byte identical across its in-memory and file-backed routes. The corrected file-backed writer uses profile V2 only when an envelope contains multiple items and requires ZIP64 local-header metadata on every V2 entry. Each workstation manifest persists the exact byte-affecting encoder profile. Staging adopts encoded bytes only when that provenance is present and supported; unprofiled legacy objects remain readable logical cache hits but are re-encoded as V1 for publication. For a retained single-entry object, V1 and the superseded interim V2 label may be interchanged only when the resulting transport digest is already authorized by the retained manifest.

Project hard rules are:

- every Git-managed file is strictly below 100 MB;
- the default byte-split part size is 90 MiB;
- no publication batch, commit, or push introduces more than 1,000,000,000 new payload bytes;
- a shard cannot accept a transaction when projected reachable repository payload would exceed 100,000,000,000 bytes.

Large placement and materialization preflight uses a separate canonical cost-governance document. Each candidate declares logical size, backend-local unique addition, transfer size, destination, retention class, operational suitability, and a digest-bound currency/storage/transfer estimate. Policy enforces explicit quotas and allowed retention, with GitHub public/private storage first while free and suitable. Paid external storage requires approval bound to the exact backend, cost estimate, governance-policy digest, approver, cost ceilings, justification, and evidence; overriding an available suitable GitHub destination must be authorized explicitly.

The publisher verifies every part digest and reconstructs and verifies the canonical payload before making an index entry visible.

## No-full-clone remote access

Resolvers and publishers use GitHub repository/ref, tree, blob, and commit operations or an equivalent bounded transport. They fetch only topology, relevant index partitions, manifests, encodings, receipts, revocation projections, and selected objects. Neither reading nor publishing requires a persistent full shard clone. For a multipart cache miss, the Git transport resolves the immutable part paths up front and hydrates the missing blob set in bounded batches; the already-existing download workers then stream the local immutable blobs concurrently. Dependency prefetch excludes complete local packages and individually retained local parts before issuing any Git preparation. When a filtered tree omits a blob size, batched prefetch reserves the exact maximum supplied by the retained part record, capped at GitHub's 100 MB file boundary; an ordinary bounded read likewise uses its effective caller limit. `XC_CACHE_DOWNLOAD_CONCURRENCY` controls those workers and `XC_CACHE_PREFETCH_CONCURRENCY` controls independent repository preparation (both default 4, range 1--8). This batching changes neither transport identity nor verification: every downloaded part is still size- and SHA-256-checked, reused parts are checked during the one-pass reconstruction copy, the concatenated package is checked before atomic visibility, and every decoded logical item is checked against the canonical payload envelope embedded in the manifest. A reused local part that is a regular file of the wrong size, or that fails its reconstruction digest check, is moved aside within the part store as `<part>.corrupt-<pid>-<nanos>` and fetched again exactly once; quarantine scratch cleanup is best-effort and warns rather than invalidating a verified result when the operating system temporarily holds a file, a second fetch failure is reported unchanged and leaves no visible package, and a symlink or non-regular file fails closed without repair. A corrupt retained complete package is removed only after a verification error proves corruption—not cancellation—so a later attempt can reconstruct it, and concurrent reconstructors verify and reuse the same immutable winner. Bulk blob-presence checks use one `git cat-file --batch-check` process per batch, with stdin writing and stdout/stderr draining performed concurrently; individual blob reads retain exact-type checks. Git tree reads accept only regular-file modes, never symlink blobs. Every in-flight repository operation of one transport holds a reservation in a shared temporary-disk ledger. Repository/session locks are acquired before reservation, landed bytes are not counted twice, and success, error, and unwind all release the reservation, so concurrent preparations cannot collectively exceed `maximum_temporary_disk_bytes` or the filesystem reserve. Metadata fetches reserve a bounded allowance. Immutable-path digest reads reserve an exact available blob size or the 100 MB GitHub hard-file boundary when a filtered clone cannot know the size before hydration; they never reserve the much larger run-wide transfer ceiling, and then verify the hydrated and streamed sizes exactly. Resolved remote state is keyed by the exact canonical manifest digest, never by the logical payload digest alone, because identical payload bytes under different dependency closures are different artifacts with different transports. A valid existing package can be reused without a remote object read. Immutable registry/index/manifest/encoding/receipt documents are cached only for the pinned repository revision and never across revision identities.

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

For each family, the registry names one ordered shard inventory and exactly one
active writable shard. Managed publication targets only that writable shard.
Resolution checks the active shard first and then historical shards from newest
to oldest, applying the same manifest, index, receipt, reader-version, and
assurance validation in every repository. Exact dependency preflight follows the
same inventory, so a new child can safely name a parent that remains in a
read-only predecessor after rollover. The optional family-level
`active_writable_shard` permits a backward-compatible transition: the root and
family `current_writable_shard` fields can stay on the predecessor so released
single-shard readers continue to find existing objects, while rollover-aware
clients publish to and read from the successor.

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

When admission would exceed the hard ceiling or governance requires separation, an authorized topology transaction creates and validates a successor shard, updates the family route atomically, and marks the old shard read-only. The family inventory retains both repositories: new writes go only to the successor while existing artifacts and dependency proofs remain resolvable from the predecessor. A failed rollover leaves the old route valid.

Successor activation is fail closed. `SuccessorShardReadinessEvidence` binds repository ownership and visibility, branch protection, trust metadata, a read/write health check, reviewer approval, and a clean exact-revision audit. The successor must be empty, have no audit issues or unreferenced state, and expose a canonical initial ledger with the approved 100,000,000,000-byte hard capacity and complete history coverage. The registry replacement increments generation, binds the prior topology digest, marks the old writer read-only, and activates only its next-sequence declared successor. Registry mutation rechecks both head and path digest and uses one remotely verified compare-and-swap commit.

Index and ledger rebuild scans canonical manifests, encodings, objects, attestations, revocations, receipts, append-only payload-batch records, and Git reachability. The bounded ordinary audit enumerates tree paths without downloading payload blobs, validates every batch record and receipt binding, reconstructs complete discoverable index entries, and reports missing or unreferenced paths. When every reachable payload object has one consistent first-introduction record, it reconstructs exact first-seen bytes—including incomplete transactions—and compares them with the ledger's completed-plus-abandoned payload accounting. Missing, corrupt, or conflicting coverage is reported explicitly and never used to guess reclaimed capacity. The rebuilt projection must reconcile with remote hashes before writers resume.

Index repair is a distinct explicit transaction. A canonical bounded plan names the audited revision, each observed index digest, each reconstructed replacement digest and byte count, and whether entries would be removed. Entry removal and repair with unresolved integrity errors require separate policy authorization. Execution requires explicit confirmation, fresh write permission for the network-registry-bound endpoint, unchanged observed digests and branch head, one compare-and-swap commit, and remote verification. Ref conflict invalidates the plan. Capacity-ledger repair is not inferred from index repair and may proceed only when the audit proves every accounting component required by the selected ledger policy.

## Durability, trust, and revocation

Artifact plans declare one typed durability class: recomputable, expensive-reproducible, irreplaceable-source, or publication-or-certificate-record. The family policy explicitly states the minimum verified-copy count, minimum independent failure-domain count, allowed locations, and whether an archive copy is mandatory. Each copy carries payload identity, locator, failure domain, verification evidence, revocation state, and an immutable publication or placement/deposit receipt for remote locations. Multiple repositories in the same failure domain count as multiple copies but only one independent domain.

Local cleanup planning removes the candidate copy from the evidence set before assessing durability. It also evaluates age, pins, dependency reachability, assurance, disposition, recomputation cost, active transactions, and receipt completion for every requested publication target. Cleanup may remove local working copies only after all gates pass. Essential research artifacts retain immutable archive evidence and cannot rely only on a mutable live index. Remote pruning is never implicit.

Trust policy verifies principal, authority scope, signature, attested identity, repository/ref binding, toolkit and schema policy, and revocation state. Clients pin or monotonically advance signed registry and shard state to resist rollback. Revocation is an immutable signed overlay; objects are not rewritten. Emergency repair publishes corrected manifests or indexes plus an audit attestation.

Author refresh and replacement are explicit operations. `XC_CACHE_MODE=refresh` bypasses every local and remote candidate but retains normal validation, local storage, dependency recording, packaging, and publication. When `XC_PUBLISH_REPLACE=true` is combined with author mode and enabled publication, each discoverability commit removes prior entries for the affected semantic identity and installs the fresh entry as its sole current selection. After all artifacts are published, the toolkit audits each exact affected shard revision and issues a separate compare-and-swap cleanup commit removing only unreferenced current-tree manifests, transport encodings, and payload objects. Shared objects and historical receipts remain. Capacity is never decremented because the removed blobs remain in Git history.

Ordinary author publication is destination-aware. Before obtaining a private publication lease or staging repository files, the publisher reads each required destination index partition once and confirms exact active artifacts against their canonical repository-batch records. Confirmed artifacts are excluded from the new batch without reading their manifests, encodings, attestations, ZIP parts, or payload objects. Their index entries and original publication transaction identifiers remain unchanged. Only missing or no-longer-proven artifacts enter the new transaction; a destination for which every candidate is already proven produces no shard commit and no coordination-branch lock activity. `XC_PUBLISH_REPLACE=true` deliberately bypasses this no-replay behavior.

Repository-permission evidence is time-bounded. A family batch refreshes its live GitHub write session after destination inspection and uses that exact refreshed session when authorizing prepared candidates; it refreshes again immediately before sidecar creation and every remote commit. A long existing-family scan therefore cannot feed stale family-wide evidence into a later authorization bundle, while every mutation retains an independent freshness check.

Live semantic indexes govern ordinary reuse and name only the current admissible selection. Published dependency closure is different: it names an exact semantic, manifest, and payload identity that remains immutable when a later author replacement advances the live index. Identity resolution first uses an exact live-index entry; if that entry has been superseded, it reads the exact retained canonical manifest and requires a matching canonical repository-batch proof from the same authorized shard revision before materialization. Historical entries never participate in key-based selection, and a missing or mismatched manifest, payload, transport, destination, repository, or batch proof fails closed.

Destination publication preflight applies the same historical-identity rule. An exact dependency need not remain the active semantic-index selection, but its retained manifest must reproduce the requested family, semantic, manifest, and payload digests and a bounded scan of canonical batches must prove publication to the exact destination, family, repository, branch, and manifest path. The validated historical inventory is cached per immutable shard revision for the duration of the publication run. This prevents a legitimate superseded parent from being reported missing while keeping unproven orphan manifests ineligible.

Local publication staging namespaces drafts by semantic digest, canonical-payload digest, and source-manifest digest. The raw logical-item digest is insufficient: two canonical manifests can intentionally carry identical `payload.json` bytes while retaining different exact dependency envelopes, and otherwise identical canonical manifests can differ by producer identity. Distinct identities coexist as distinct drafts; retained canonical identity is checked before any local source-key shortcut, while exact manifest/payload identity still deduplicates the identity-based and key-based walks of the same published artifact. Source-key lookup retains all matching drafts and deterministically selects the strongest source quality and then the lexicographically smallest source-manifest digest. A reopened or same-process staging sink uses an already validated draft's canonical manifest to continue walking its parents without resolving and decoding that payload again, but it re-resolves a key-based edge when the retained draft lacks a recorded source quality or does not meet the edge's required quality. Staging presence never suppresses traversal, so an interrupted directory containing a child but missing an ancestor is repaired on the next walk. A successful transitive walk is memoized only for the current process; that memo deliberately does not survive a restart. The workstation cache answers exact published-identity queries from a persistent inventory, `identities/<prefix>/<semantic-digest>.json`, that every write maintains under a process lock and an advisory file lock. The inventory only grows, so its byte length is a change signature: the in-process memo of a query is reused only while the inventory file is unchanged, which makes an artifact written by another process visible without a restart; a signature change during a query is reread before the result is memoized. Each inventory candidate revalidates the retained canonical manifest against the adapter's semantic key, logical payload bytes and provenance; a forged adapter cannot satisfy or pollute an exact identity lookup, and drive-qualified or otherwise non-normal manifest paths are rejected. An identity absent from a writable store's inventory triggers one bounded scan of that semantic digest's manifest directories per process, which also repairs caches written before the inventory existed. A read-only store cannot persist that repair and therefore rescans on every inventory miss. Before closure payloads are consumed, exact sibling dependencies are resolved as metadata and their immutable transports are prepared in bounded repository batches. A dependency whose selected layer retains a provenance-bound verified encoded object is staged as metadata only: the logical payload is never inflated, the encoded object is split and hashed, and the result is bound either by decoding it against the manifest's logical digest (an unpublished artifact) or by the retained canonical manifest's transport digest (a published one). When materialization also verified the original split parts, staging hard-links those immutable content-addressed parts directly, with a verified copy fallback for a different filesystem; an adopted hard link is hash-verified before its descriptor is exposed, parts that the process did not itself verify are hashed before they are linked, and if the part proof is unavailable, its files were pruned, or a regular part file has the wrong size or digest, staging safely splits the verified complete package (a symlink, including a broken symlink, or non-regular file fails closed). Reopening validates that the deserialized parts path is under its canonical semantic/payload/storage-identity draft root and checks that every part is a regular file of the recorded size; the publisher hashes every staged part immediately before it enters Git, so that check is not duplicated at reopen. The unchanged single-entry encoder retains profile V1 and existing transport identities; profile V2 is emitted only by the corrected multi-item writer and verification requires ZIP64 local-header metadata on every V2 entry. Workstation ZIP objects up to 90 MiB are read once into a bounded buffer, hashed from that buffer, and then inflated from memory, within a process-wide 256 MiB in-memory allowance shared by all concurrent reads; a read that does not fit streams through the two-pass path without waiting. `XC_CACHE_SINGLE_PASS_ZIP_BYTES` and `XC_CACHE_IN_MEMORY_ZIP_BYTES` accept 1 byte through 16 GiB; invalid or zero values retain their defaults. Before allocation, the reader requires a regular non-symlink file of exactly the declared size, bounds the compressed read to that size, and rejects trailing data; before inflation it requires the ZIP entry's declared size to equal the manifest and refuses output beyond that bound. Transport records accept only the named V1/V2 profiles and reject duplicate repository paths. When several variants for one semantic identity are absent from a destination, one family batch commits every immutable manifest and transport, orders closure-only and older variants first, and leaves the newest key-addressable dependency-complete variant as the live index selection. If an equal-or-newer discoverable entry is already live, an older closure-only variant is still committed with its immutable manifest, encoding, objects, and repository-batch proof but does not mutate the live semantic index. This exception applies only to exact dependency closure; an ordinary publication cannot downgrade a discoverable producer.

Assurance requirements and attestations are keyed by source artifact and
logical content rather than canonical closure, so staging applies them to
every colliding canonical draft and reports retained assurance only when all
such drafts agree. Reopen path validation also accepts the bounded v0.14.1
two-level `semantic/payload/{draft.json,parts}` layout in addition to the
current source-manifest-namespaced layout; paths outside either exact shape
remain invalid.

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
| `ccm_root_conditioning_analysis` | `ccm-evidence` | Per-root secular term scale, derivative conditioning, and retained-pole geometry |
| `ccm_prime_power_response_analysis` | `ccm-evidence` | Opt-in per-event `dQ/du` transport through an isolated lowest-even eigenvalue, full eigenvector, and retained roots; v2 binds same-sector Sturm-gap evidence |
| `ccm_u_flow_response_analysis` | `ccm-evidence` | Opt-in decomposed pole/archimedean/prime/total `u`-flow through an isolated lowest-even state and moving secular roots; v2 binds same-sector Sturm-gap evidence |
| `ccm_target_distance` | `ccm-distance` | Private-only weighted distance to a runtime-supplied target; schema v2 binds the opaque definition digest |
| `ccm_distance_resolution_evidence` | `ccm-distance` | Private-only tail and same-rule refinement evidence for runtime target distance |
| `ccm_target_residual_analysis` | `ccm-distance` | Private-only signed and crossing diagnostics for a runtime target residual |
| `ccm_deviation_decomposition` | `ccm-distance` | Private-only opt-in projection onto a runtime-supplied auxiliary profile, with residuals under both readings of the distance weight; schema v3 binds the opaque definition digest |
| `ccm_eigenfunction_profile` | `ccm-distance` | Target-independent sampled CCM eigenfunction and normalized coefficients; public-eligible |
| `ccm_discretization_distance` | `ccm-distance` | Target-independent distance between two discretizations; public-eligible |
| `ccm_validation_record` | `ccm-evidence` | Natural-versus-forced evenness evidence |
| `ccm_sector_gap` | `ccm-evidence` | Even/odd ground-state depths, `GapLog=D_even-D_odd`, direct difference, ordering, and simplicity margin |
| `ccm_sector_gap_certificate` | `ccm-evidence` | Opt-in exact cutoff-free finite-sector enclosures, parity outcome, simplicity gap, and positivity result |
| `ccm_post_discovery_comparison` | `ccm-evidence` | External-reference comparison attached only after independent discovery is complete |

Cheap prime enumeration and the analytic pole formula are embedded in the semantic identity or payload metadata of their consumers. They are deliberately not fetched as separate objects. Every matrix or state cache hit is checked against its exact dependency: parity matrices are reconstructed from tau, factorizations pass a deterministic solve residual, and eigenpairs replay the tau residual and their structured inverse-iteration stopping evidence. That evidence records the configured limit, steps used, unshifted convergence, final Rayleigh change, shifted-refinement outcome, and final relative residual. Root ranges must be finite, positive, ordered, and identity-bound, and evidence values must equal the results produced by their dependencies.

Response payload schema v2 is fail-closed. Both response kinds require the
`even-sector` state route, depend on the even-sector matrix and indexed
even-sector eigenvalues, and retain the disjoint HP Sturm enclosures for
indices zero and one, their positive gap lower bound, and the selected state's
absolute and relative residual plus residual-to-gap bound. The bordered solve
is performed in the reduced even sector and lifted back to the full coefficient
layout for retained root transport. If the same-sector gap cannot be resolved,
or the selected state cannot be associated with the isolated lowest branch,
capture stops with `unresolved_near_crossing`; a small bordered residual alone
cannot admit the artifact. The v2 semantic key and dependencies prevent reuse
of unguarded v1 response payloads.

Roots are stored as bounded ordered windows, not one remote object per scalar root. Independent discovery and reference-seeded refinement are different artifact kinds and cannot satisfy one another's semantic requests. An independent range records its one-based bounds assigned by cumulative finite-source counts, discovery/count methods, `reference_seeds_used=false`, each value's explicit convergence status, per-root iterations, final MPFR correction, residual, achieved-digit estimate, root-precision policy, and exact secular-source dependency. A refinement range records its exact supplied starting points and `reference_seeds_used=true`. Fixed-guard v6/v7 windows remain the default and preserve the historical 64-bit requested-accuracy reserve, payload, and identity. Adaptive v9 is opt-in, binds the exact secular-source content digest, and retains the target, resource ceiling, evaluation and verification precisions, escalation count, wider stored-point correction, and stopping reason under the explicit `exact_stored_point_source` scope. Cache replay recomputes the achieved-digit estimate and secular residual; v9 additionally recomputes the wider correction from the exact stored source and root. Computed assurance retains finite, ordered `stagnated` and iteration-limited `approximate` values without relabeling them as converged; a failed outcome with no value invalidates the window. Cross-checked or certified assurance requires every root to be fully converged. A v9 miss always starts from the identity-bound request seeds; it never promotes or warm-starts from v6/v7, because path-dependent iterations and escalation evidence are part of the payload. `RequireReuse` never computes a missing v9 child. The run-evidence artifact carries the same mode, index bounds, exact root dependency, and separate outcome counts. This keeps research data directly accessible without allowing child-object counts or remote lookup overhead to dominate the numerical calculation.

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

`ccm_sector_gap_certificate` is a separate schema-2 certified child and never
relabels `ccm_sector_gap`. Its semantic identity binds the even-spectrum,
odd-spectrum, and numerical gap parent digests, the Arb backend, precision,
cutoff-free assembly policy, and bracket policy. The payload retains the full
raw exact-rational Tau interval matrix, its assumption-free interval-LDLT
inertia certificate, a derived reflection-orbit canonical matrix, and the
component-evidence digest. Positive definiteness is true only when the raw
full-matrix inertia proof is conclusive and all pivots are positive. Separately,
replay intersects transpose/reflection symmetry orbits, rebuilds the
orthonormal parity blocks, replays exact shifted-inertia boundaries for the
lowest two even eigenvalues and lowest odd eigenvalue, and recomputes the
even-sector simplicity gap. Those parity, ordering, and sector-simplicity
claims explicitly record the premise that the exact closed-form CCM matrix is
centrosymmetric; positivity does not depend on it. Signed lower and upper bounds for
`lambda_odd-lambda_even` determine a finite parity outcome of `even`, `odd`, or
`unresolved`; no outcome is assumed in advance. Ordinary retained
sector values and cutoff-free midpoint values are recorded as search guides,
not proof inputs. This artifact is routed to the existing `ccm-evidence`
public/private shards and has a 0.14.1 producer and reader floor.

Mk assigns separate artifacts to exact moments, basis and symmetry data, transformations, dense fixtures, matrix-free or structured operators, approximation bounds, adaptive-space histories, checkpoints, candidates, and quotient certificates. Corrected construction semantics invalidate only their dependent closure.
