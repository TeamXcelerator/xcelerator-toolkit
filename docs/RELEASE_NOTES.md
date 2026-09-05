# Release notes

## 0.14.3

Version 0.14.3 adds backward-compatible rollover for registry-managed cache
families. A family document may name an `active_writable_shard` while retaining
its legacy `current_writable_shard` pointer. Rollover-aware readers search the
active shard first and then every declared predecessor from newest to oldest;
managed publication and exact dependency preflight use the active shard while
accepting dependency proofs retained in historical shards. A semantic identity
named by the active shard shadows older copies, preserving revocation, reader-
floor, and assurance decisions. Released single-shard readers continue to use
the unchanged legacy pointer and therefore retain access to the predecessor.

This routing release does not change artifact semantic keys or logical payload
schemas. Existing artifacts remain readable. Newly computed manifests record
producer version 0.14.3 in the normal release-bound way; artifacts reused from
an earlier release retain their verified canonical manifest identities.

## 0.14.2

Version 0.14.2 is a cache-reuse performance and correctness release. It removes
the remaining avoidable overhead of runs that reuse already cached artifacts
and hardens the new fast paths against identity ambiguity, corrupt local state,
concurrent resource overcommit, and interrupted staging. Existing 0.14.1
artifacts remain readable. No semantic or payload schema version is raised by
these cache changes. Newly produced canonical manifests record producer
version 0.14.2, so their manifest identities change in the normal release-bound
way even when their logical payload and transport bytes are unchanged. Every
fast path remains closed by a digest check before anything becomes visible
locally or enters Git.

- Cold multipart GitHub reuse now resolves all missing part blobs up front and
  hydrates them in bounded batches before the existing download workers stream
  the immutable objects concurrently. This removes the hidden one-network-
  fetch-per-part serialization that made a 14-part matrix package appear
  parallel in the in-memory test while remaining serial on a cold Git shard.
  Complete local packages and already retained local parts are removed from
  the prefetch set before Git preparation begins. Bulk blob-presence checks
  use one `git cat-file --batch-check` process per batch. Its input is now
  written concurrently with output draining, preventing the pipe deadlock
  that large (roughly 300-or-more-object) batches could trigger; individual
  object reads retain their separate exact-type check.
  Reused local parts are SHA-256 checked during the reconstruction copy instead
  of in a separate preliminary full-file pass; the package and decoded logical
  payload checks are unchanged. A reused regular file whose size or bytes no
  longer match is moved aside within the part store and downloaded again
  exactly once, instead of failing the run identically on every attempt. The
  quarantine scratch file is removed on a best-effort basis when the operation
  exits; an operating-system cleanup lock produces a warning and a later drop
  retry rather than invalidating an otherwise verified result. Symlinks and
  non-regular files remain hard failures. A corrupt
  retained complete package is removed after its failed verification so the
  next attempt can rebuild it from verified parts, and concurrent processes
  completing the same immutable package verify and reuse the winner.
  Temporary-disk accounting is shared by every in-flight repository operation
  of one transport, does not double-count bytes after they land, takes
  repository/session locks before reserving, and releases reservations on
  success, error, or unwind. Metadata-only Git fetches now reserve a bounded
  allowance. Immutable-path digest reads reserve the exact blob size when Git
  can expose it and otherwise the 100 MB GitHub hard-file boundary, never the
  20 GiB--2 TiB transfer ceiling; hydrated and read sizes are checked exactly.
  Batched part prefetch uses each caller's exact retained-part size as the
  fallback when a filtered tree cannot expose the size, so closure-scale
  preparation does not multiply a 100 MB reservation across small objects.
  Large-artifact logs split fetch, reconstruction, and verification/decode
  time.
- Local ZIP reuse returns the verified encoded object from the same pass that
  verifies and decodes its logical payload, avoiding a second complete hash of
  the compressed ZIP before workstation adoption. Compressed objects up to
  90 MiB (the split part size) are read from disk once, hashed from that
  bounded buffer, and then inflated from memory, within an aggregate 256 MiB allowance shared
  by every concurrent read in the process; a read that does not fit streams
  as before, without waiting. `XC_CACHE_SINGLE_PASS_ZIP_BYTES` and
  `XC_CACHE_IN_MEMORY_ZIP_BYTES` override the two limits. Valid overrides are
  1 byte through 16 GiB; zero, overflow, and larger values keep the default.
  The single-pass reader also checks the regular-file type
  and exact declared size before reading, bounds the read to that size, and
  rejects a trailing byte, so a corrupt or concurrently enlarged file cannot
  exceed the reservation. Immutable remote metadata is retained by repository
  revision for the process. Resolved remote state is keyed by the exact
  canonical manifest digest rather than the logical payload digest, so two
  published artifacts with identical payload bytes and different dependency
  closures can never exchange encodings or parts.
- The workstation cache maintains a persistent identity inventory
  (`identities/<prefix>/<semantic-digest>.json`) on every write. Exact
  published-identity lookups read that inventory instead of walking the
  artifact directory tree. A cache written before the inventory existed, or
  by an older toolkit, is repaired by one bounded scan per semantic digest per
  process. The in-process memo revalidates against the inventory file, so an
  artifact written by another process into the same cache is visible without
  a restart, and a signature change during a query is reread before the result
  can be memoized. Writable legacy stores scan once per semantic digest and
  repair the inventory; read-only legacy stores rescan on each inventory miss
  so one identity cannot hide another under the same digest. Inventory paths
  reject drive-qualified and non-normal components. Every inventory candidate revalidates the retained canonical
  manifest against the adapter's semantic key, logical payload bytes and
  provenance before it may satisfy an exact identity lookup.
- The unchanged single-entry workstation/publication encoder retains the V1
  profile and therefore preserves every existing transport identity. The
  file-backed packager uses V2 only for envelopes with multiple items, because
  that is the route whose bytes changed to require ZIP64 metadata on every
  entry. Verification checks
  the claimed V2 local-header property. The exact profile is persisted with
  each workstation object and must match before staging adopts its encoded
  bytes. Legacy unprofiled objects remain valid logical cache hits but are
  decoded and re-encoded as V1 for publication. A retained single-entry object
  from the superseded interim V2 build is relabeled V1 only when doing so
  exactly reproduces a transport digest authorized by the retained V1 manifest.
- Publication closure traversal can continue from the validated canonical
  manifest of an already staged draft without resolving and decoding that
  payload again. It still traverses every parent, including after reopening an
  incomplete staging directory, so the resume correctness guarantee is
  preserved. Completed closure subtrees are memoized for the current process,
  exact sibling dependencies are progressively prepared in bounded batches,
  and per-repository Git locks permit independent shards to hydrate
  concurrently. A dependency whose verified encoded object a cache layer
  already holds is staged from that object and its retained parts without
  inflating the logical payload at all; parts this process did not itself
  verify are hashed before they are linked, and a regular file of the wrong
  size or with the wrong digest falls back to splitting the verified package
  (a symlink or non-regular file fails closed). Staging adopts an already
  verified deterministic ZIP when one is available and hard-links verified
  content-addressed parts directly (or performs a verified copy across
  filesystems). Reopening a staging directory and resuming an existing draft
  check part structure and size only; every staged part is still hashed by the
  publisher immediately before it enters Git. A resumed run answers an already
  staged identity dependency from the staging directory before consulting any
  cache layer, but only when its recorded source quality satisfies the current
  dependency edge; older drafts without that quality field are resolved again.
  Draft lookups by identity are constant time. Source lookups retain every
  match and deterministically select the strongest source quality, then the
  lexicographically smallest source-manifest digest. Assurance requirements
  and attestations name the shared source key and bytes, so they update every
  colliding canonical closure rather than silently promoting only one. Draft directories include the source-manifest digest
  in addition to semantic and canonical-payload digests, so manifests that
  differ only in producer identity coexist. On the canonical staging route the reuse path hands its payload to
  staging instead of copying it, and a freshly computed artifact is staged
  from its verified store object; the payload-carrying queue sink still
  receives its own copy. An adopted hard link is hash-verified before its
  descriptor is exposed, decoded ZIP output is bounded by its declared logical size, and
  transport records reject duplicate repository paths. Retained canonical identity is evaluated
  before the local source-key shortcut, so artifacts with identical source
  keys and logical bytes but different dependency closures remain distinct.
  Reopening accepts both the current source-manifest-namespaced layout and the
  fully bounded two-level layout written by v0.14.1, so persistent author
  staging roots remain resumable after the upgrade.
- Fixed the closure walk selecting the newest candidate under a dependency's
  semantic key and then rejecting it because the content digest differed. Any
  newer artifact under the same key in a higher-precedence layer broke
  publication of every child that named the older one. Dependencies now
  resolve by exact key, content digest, and required quality, selecting the
  same candidate that dependency prefetch prepares.

## 0.14.1

Version 0.14.1 is an amended maintenance release for CCM research capture,
root refinement, finite-sector evidence, and managed cache publication.

This release also moves research-target definitions out of the public toolkit:
target-
dependent work now consumes a private runtime specification, binds only its
opaque SHA-256 identity into artifacts, and is restricted to private cache
publication.

- Fixed managed publication rejecting the closure of artifacts reused from a
  shard. An adopted artifact carries an empty key-based dependency list --
  its dependencies are named by published identity in its retained
  canonical manifest -- so closure staging stopped one level below any
  reused artifact, and publishing to a destination that did not already
  hold the grandparents failed after all computation with "managed
  publication closure is missing exact dependency". Staging now also walks
  retained canonical identity lists, resolving each member from the local
  cache first and any configured shard layer otherwise, so first
  publication to a new destination requires no recomputation. Cache layers
  gained identity-based candidate lookup (local directory scan; shard
  index lookup filtered to the exact canonical manifest digest). Exact
  shard lookup also falls back to the immutable canonical manifest and its
  validated repository-batch proof when an authorized replacement has moved
  the live semantic index to a newer manifest. This preserves old dependency
  closure without allowing historical artifacts to win ordinary semantic
  reuse. Destination preflight now applies the same rule: when a child names
  an exact manifest superseded in the live index, it validates the retained
  manifest and its canonical publication-batch proof instead of falsely
  reporting a missing dependency after computation. Historical inventories
  are read once per shard revision and remain bounded. Regression tests cover
  both the reused-parent/unstaged-grandparent shape and the superseded-index
  dependency that interrupted the Claim 2b HP-1000 sweep.
- Reopened publication staging is completed rather than trusted: closure
  walks traverse the dependencies of already-staged artifacts and suppress
  only their re-recording, so a staging directory left by an interrupted or
  pre-fix run gains its missing closure members on the next attempt.
- The staging sink deduplicates by published identity in addition to source
  key, so an artifact reached through both the key-based and identity-based
  walks -- in either order -- stages exactly once.
- Identity-first dedup now persists the later validated real cache key over
  the temporary `closure/...` adapter provenance. This lets a freshly staged
  child bind its key-addressed dependency to the already staged canonical
  draft instead of failing after reuse with a false "canonical dependency
  draft is missing" error.
- Canonical staging directories are keyed by semantic digest and full
  canonical-payload digest, not by the raw `payload.json` item digest. An
  historical manifest and its active replacement may intentionally retain
  identical numerical bytes while naming different dependency closures; both
  exact identities can now coexist in one publication run without colliding.
  If a new destination lacks both identities, the family batch publishes
  closure-only and older variants first and leaves the newest key-addressable
  replacement as the final live index entry. If the destination already has
  an equal-or-newer live entry, an older closure member is committed only as
  immutable dependency material with its batch proof; it does not replace or
  downgrade that live entry. Non-closure downgrade attempts remain rejected.
- Dependency identities are validated before any index path construction or
  digest slicing, and retained canonical manifests are validated and bound to
  their adapter manifests before their dependency lists are trusted.
- Author publication now stages validated cache-reuse hits as well as fresh
  computations, preserving their exact adapter manifest and payload identity
  and walking the same complete dependency closure. This includes
  `XC_CACHE_MODE=require_reuse`: the mode still never computes a miss, but a
  workstation-only hit can no longer vanish from a successful publication
  run. Destination indexes, canonical manifests, and batch records are checked
  before mutation, so an artifact already published at the requested target is
  a verified no-op rather than a redundant commit.
- `XC_PUBLISH_EXECUTE=true` is fail-closed when the current process observed no
  artifacts, and incomplete execution reports now fail the run after retaining
  their resumable journals. A stale or empty staging directory can therefore
  no longer produce a vacuous publication success.
- Long family publication scans now refresh GitHub write evidence immediately
  before candidate authorization and pass that refreshed session into the
  authorization bundle. Previously the batch path refreshed a local session
  but accidentally prepared candidates from the original family-wide session;
  a scan lasting more than five minutes therefore failed before mutation with
  a stale-permission error. Mutation still performs its independent final
  refresh. This changes no artifact identity or payload bytes.

- Distance/profile capture resolves completed artifacts before constructing an
  eigenstate, sampling a profile, or running quadrature. Refresh, verification,
  and cache-disabled modes retain eager compute-and-compare behavior.
- Distance, profile, residual, resolution, decomposition, and inter-
  discretization artifacts now resolve the canonical managed
  `ccm_weil_eigenpair` and retain it as an exact dependency. The eigenpair
  content digest is part of each affected semantic identity, and retained
  target-distance eigenvalues must exactly equal the canonical eigenvalue.
  Previously these paths independently requested the midpoint of a selected-
  sector Sturm enclosure whose absolute tolerance was too coarse near the
  HP-200 floor; at `(lambda^2, N) = (100, 120)` it retained a negative
  `-3.38270e-211` distance eigenvalue while the residual-validated canonical
  state was `+3.48676e-215`. The numerical claim path was unaffected. All
  affected artifact semantic versions are advanced, so no legacy profile or
  distance payload can be reused under the corrected identities.
- A target-distance capture resolves one managed `gauss_legendre_rule` artifact for each
  unique `(points, working precision)` request and reuses the exact table
  across configurations, distance, norm, and signed-residual analysis. The
  norm replays the bit-identical eigenfunction values recorded by the distance
  pass while retaining its established reduction order.
- Root refinement constructs the HP secular-pole vector once per root window
  and shares it across seeds.
- Root refinement keeps the historical fixed-guard v6/v7 policy as the
  default, preserving existing claim-script arithmetic, identities, and payload
  bytes. Adaptive v9 is explicitly selected with
  `HighPrecConfig::with_adaptive_root_precision()`. It holds the requested
  accuracy fixed, widens only the inexpensive secular-root arithmetic when the
  initial 64-bit reserve is insufficient, and accepts convergence only after a
  wider-precision correction replay at the exact stored root. The payload
  records evaluation/verification precision, escalation count, correction,
  stopping reason, and the explicit `exact_stored_point_source` scope. The key
  binds the exact secular-source digest.
- An adaptive miss always starts from the request's identity-bound seeds. It
  does not use a v6/v7 artifact as a warm start because iterations and adaptive
  escalation evidence are payload fields; allowing a cache-dependent start
  would produce different bytes under one semantic key. Reuse, refresh, and
  verification therefore share one canonical v9 computation path. Strict
  `RequireReuse` mode does not compute a missing v9 artifact.
- Zero and overflowing capture resolutions fail before HP work. Reuse-mode
  payload validation binds schema, shapes, normalization, alpha, ordered rule
  metadata, finite scalar values, and profile ordering to the exact request.
- The `maximum` capture documentation now distinguishes capture volume from
  independent certification or cross-checked assurance.
- Maximum capture adds `ccm_distance_resolution_evidence`: fixed coefficient-
  tail diagnostics at `1e-15`, `1e-30`, and `1e-45`, and same-rule Q/2Q
  refinement of each uniform grid. A deterministic 4Q continuation runs only
  when the `1e-8` relative tolerance is missed. Gauss--Legendre remains the
  independent-family measurement in `ccm_target_distance` and is not doubled.
  Left/right/trapezoid refinements reuse values from Q in 2Q and from 2Q in 4Q
  only at binary-identical MPFR abscissae; midpoint grids bypass this exact
  optimization because they are not nested.
- Maximum capture also adds `ccm_target_residual_analysis`. For every retained
  distance rule it records signed, positive, and negative weighted residual
  mass; on the retained profile grid it records the sign sequence, extrema,
  and strict sign-change brackets. This is diagnostic retention only: it does
  not introduce crossing-refined or piecewise integration.
- Maximum capture adds `ccm_root_conditioning_analysis` in the existing
  `ccm-evidence` family. It records each returned root's signed secular
  derivative, absolute secular-term sum, reciprocal derivative, magnitude
  condition estimate, neighboring retained poles, nearest-pole distance,
  normalized isolation margin, and interval position. The computation is one
  parallel direct pole-sum pass; it does not invoke FLINT, Arb, interval
  Newton, matrix construction, or root refinement. Reuse replays both the term
  sum and derivative from the exact source.
- A current canonical-eigenpair-bound profile and distance can supply missing
  resolution, residual, or decomposition children on a later reuse-first
  capture. Legacy unbound profiles and distances are deliberately not accepted
  under the corrected identities; the canonical eigenpair can still be reused
  without repeating its eigensolve.
- Existing root-range and secular-source/eigenpair artifacts can likewise
  supply a missing root-conditioning child on a later maximum run. Its cache
  identity binds both parent content digests and the exact returned-root
  selection, so existing root keys and payloads do not move.
- Explicit prime-response capture adds
  `ccm_prime_power_response_analysis` to the existing `ccm-evidence` family.
  It is excluded from `maximum` and enabled only through
  `capture_prime_power_response` or `.with_prime_power_response()`. The artifact
  isolates every active prime power's analytic contribution to `dQ/du`, shares
  one reduced-even-sector bordered factorization across the event solves, and
  retains eigenvalue, full eigenvector, and selected-root responses with
  replayable residuals. Schema/semantics v2 binds disjoint indexed HP Sturm
  enclosures for the lowest two even eigenvalues and records the positive gap,
  selected-state residual, and residual-to-gap bound. Unresolved crossings and
  non-even-sector state routes fail closed. At an event edge the response
  reduces to the exact rank-one derivative jump. Existing Tau, eigenpair,
  root-range, and secular-source artifacts are unchanged and act as exact
  parents; the even-sector matrix and indexed eigenvalues are additional exact
  dependencies. Unguarded v1 response payloads are not reusable as v2.
- Explicit complete-flow capture adds `ccm_u_flow_response_analysis` to the
  existing `ccm-evidence` family. It is excluded from `maximum` and enabled by
  `capture_u_flow_response` or `.with_u_flow_response()`. The artifact retains
  analytic pole, archimedean, aggregate-prime, and total Tau-velocity channels;
  full selected-eigenstate transport; fixed-pole root responses; isolated
  secular-pole motion; and the final combined root response. Its analytic
  derivatives are checked against same-family two-sided Tau refinement rather
  than against any single external event formula. It shares the same v2
  same-sector Sturm-isolation precondition, reduced solve, explicit crossing
  failure, and cache separation from unguarded v1 payloads as prime response.
- Explicit finite-sector certification adds `ccm_sector_gap_certificate` to
  the existing `ccm-evidence` family. It is excluded from `maximum` and enabled
  with `.with_sector_gap_certification(...)` on a capture that also retains at
  least two parity-sector eigenpairs. The certified child binds the numerical
  even/odd spectra and GapLog manifests, independently assembles the cutoff-free
  Arb interval matrix and first certifies the raw full matrix's inertia with
  exact-rational interval LDLT. Positive definiteness is therefore
  assumption-free. It separately intersects transpose/reflection symmetry
  orbits and replays exact shifted inertia for the lowest two even eigenvalues
  and lowest odd eigenvalue. Exact separation reports the finite ground parity
  as `even`, `odd`, or `unresolved`; parity, sector ordering, and sector
  simplicity explicitly depend on the recorded premise that the exact
  closed-form CCM matrix is centrosymmetric. Both ordinary numerical guides and
  native cutoff-free midpoint guides are retained for research, but neither is
  trusted by offline proof replay. Existing sector artifacts can parent a later
  opt-in certificate, while a cache miss still pays the interval assembly and
  proof cost. The scope is one finite `(c, N)` matrix, not continuum parity or
  convergence.
- Explicit deviation-decomposition capture adds `ccm_deviation_decomposition`
  to the existing `ccm-distance` family. It is excluded from every named
  capture level and enabled only through `capture_deviation_decomposition` or
  `.with_deviation_decomposition()`. It records the amplitude of a private
  runtime-supplied auxiliary profile in the deviation, with the deviation,
  auxiliary-profile and residual norms and the relative residual, under both readings of
  the distance weight; both are retained because they are not equivalent and
  an amplitude without its metric is not recoverable. A vanishing amplitude
  at a crossing is recorded rather than rejected. The artifact reads only the
  retained profile, so it backfills onto older profile captures without an
  eigensolve, and adds a new artifact without mutating existing records.
- `crate::target` gains generic runtime-supplied base and auxiliary profile
  evaluation in `f64` and high precision. The private specification is loaded
  from `XC_TARGET_SPEC_FILE`; only its SHA-256 digest is retained.
  `crate::deviation` adds the projection those artifacts use.
- Target-derived distance, resolution, and residual artifacts use
  schema/semantics v2; the deviation decomposition uses schema/semantics v3
  because an interim pre-release build of this branch serialized a draft
  projection field name under v2, and the final schema refuses interim
  payloads rather than colliding with them. All four bind the opaque
  target-specification digest. Managed publication withholds these kinds from
  every public leg (see the mixed-visibility routing entry below), and public
  bootstrap layers ignore them. Eigenfunction profiles and inter-discretization
  distances remain target-independent and may still be published publicly.
- Corrected the prolate chi-squared asymptotic predictor. The prefactor
  carried `sqrt(2*pi)` where `2^14*sqrt(2)*pi^5/3` belongs, leaving it low by
  `pi^(9/2) = 172.65`. Fuchs' constant `4*sqrt(pi)*8^4/4!*(2*pi)^(9/2)` is
  `2^15*sqrt(2)*pi^5/3` and governs the concentration deficiency
  `1 - lambda_4`; the predictor is for `1 - chi_2`, and `chi_2^2 = lambda_4`
  gives `1 - chi_2 ~ (1 - lambda_4)/2`, so the retained prefactor is half of
  Fuchs' constant. The corrected prefactor reproduces the
  published `C_0 = 6.373563` and is pinned by a regression test. The value
  is an in-memory research observable and is not serialized into any
  artifact payload or semantic key, so no cache identity moves and no
  retained record changes.
- Fixed `ccm_target_residual_analysis` rejecting every payload it produced.
  The one-sided masses were derived from unrounded working-precision values
  while the reader re-derives them from the retained decimals and requires
  exact string equality; `decimal` emits two digits fewer than an exact
  round trip needs, so no payload could validate. The masses are now derived
  from the values as retained. No artifact of this kind had ever been
  successfully written, so nothing existing is affected. A regression test
  covers a genuinely two-sided residual, which the previous one-sided
  coverage could not exercise.
- The standalone Gauss-Legendre node cache now resolves to
  `$XC_CACHE_ROOT/gl_cache`, falling back to `<cwd>/data/gl_cache` only when
  that variable is unset. It previously always wrote relative to the working
  directory, so a run started inside a checkout deposited binary cache files
  there rather than on the operator's chosen cache volume.

Historical fixed-guard root payloads retain their v6/v7 keys and bytes.
Runtime-target artifacts intentionally receive new identities (v2, and v3 for
the deviation decomposition) and do not reuse earlier embedded-target records. Adaptive root refinement
is a new opt-in v9 identity and schema-5 payload, and the added secular term
scale is a v2 root-conditioning identity and schema-2 payload. The sector-gap
certificate uses its corrected v2 semantic identity and schema-2 payload;
other new evidence kinds retain independent v1 identities and the v0.14.1
producer/reader floor.

- The persisted Weil eigenpair is computed from the canonical initial
  state. Automatic lower-N continuation seeding and cross-N sweep seed
  threading were removed: a seeded solve retained different bytes under the
  same semantic identity, which content addressing forbids. An explicitly
  offered seed is ignored at the compute boundary, and a regression test
  pins byte equality across the seed-absent, seed-available,
  explicit-seed, and refresh paths.

- Mixed-visibility publication routes instead of failing: under a `both`
  target, private-only runtime-target-derived kinds ride the private leg and
  are withheld from the public one, public-eligible kinds publish to both, and
  an empty public leg is skipped. The only remaining hard failure is an
  explicit public-only request in which nothing is public-eligible. The
  staging, planning, and bootstrap guards remain hard backstops.

## 0.14.0

Version 0.14.0 introduced a research-measurement layer that is superseded by
the amended private runtime-target interface in 0.14.1.

- `xc_spectral::target` supplies binary64 and arbitrary-MPFR evaluation for a
  normalized runtime target. In the amended release its research definition is
  private input rather than public source.
- `xc_numerics::grid_integral` adds deterministic uniform-grid integration
  (left/right Riemann, midpoint, trapezoid) on grids uniform in `u` or `ln u`,
  at binary64 and HP. The scheme and grid variable are explicit arguments, and
  tests verify each rule's documented error order.
- `WeightedIntegrationRule` selects between the uniform-grid family and
  Gauss--Legendre for every weighted norm and distance. The two are peers: the
  toolkit fixes no node count and blesses no rule. Gauss--Legendre converges
  spectrally on smooth integrands, but the distance integrand carries an
  absolute value and loses that advantage at any interior residual sign
  change, so the choice belongs to the integrand rather than to convention.
  The rule is part of the retained artifact identity.
- `xc_spectral::distance` reconstructs the normalized even CCM eigenfunction
  (`f(1) = 1`) from `V_n` coefficients and measures weighted `alpha`-norms,
  inter-discretization distances `D_alpha(N, M; lambda)`, and the distance to
  target `d(N, lambda)`. Every result carries the integration rule, grid
  variable, resolution, `alpha`, and precision that produced it.
  `ccm_distance_to_target_hp` measures `d(N, lambda)` end to end from cached or
  computed CCM ground states.
- `xc_spectral::ccm::hp` exports the even and odd parity-sector eigenvector
  expansions with their normalization documented and tested as an isometry.
  `xc_spectral::prolate` gains an eigenvector `xi_l2` norm, making the
  scale-free relative distance recoverable, and an end-to-end educated-guess
  comparison `ccm_prolate_distance_hp`.
- `ccm_discretization_distance_hp` measures `D_alpha(N, M; lambda)` end to
  end for two discretizations of one `lambda^2`. This is the quantity the
  first stage of the program is stated in, and it needs no target function.
  `WeilEigenfunction::from_normalized_coefficients` rebuilds an eigenfunction
  from the coefficients a retained profile carries, so a published artifact is
  usable without repeating the eigensolve. `target_crossings_f64` reports where
  the target residual changes sign: each interior crossing is a derivative
  kink of its absolute value, which is what decides whether Gauss--Legendre keeps its spectral
  advantage for a given configuration.
- A new `ccm-distance` artifact family retains eigenfunction profiles
  (`ccm_eigenfunction_profile`), target-distance measurements
  (`ccm_target_distance`), and inter-discretization distances
  (`ccm_discretization_distance`). Retention is opt-in: it is absent at the `claim`,
  `research`, and `gap` capture levels, is requested explicitly through
  `CcmDistanceCaptureOptions`, and is included by
  `CcmResearchCaptureOptions::maximum`. A capture records **several rules at
  once** and the default spans both families, so a retained measurement always
  shows its convention spread instead of a single unqualified number; the whole
  rule list is part of the semantic key. The retained profile additionally
  carries the normalized `V_n` coefficients, which are lossless: a consumer can
  evaluate the eigenfunction at any abscissa and apply any rule at any
  resolution, rather than being limited to the rules captured here.
- The `target_distance` example prints a cross-check card for line-by-line
  comparison between independent implementations.

### Persistence correctness

Persisted high-precision decimal scalars now carry guard digits sufficient
for exact round-trip. The previous width, the bare ceiling of
`bits * log10(2)`, is one digit short of unique recovery, so a stored root
decoded up to one ulp away from the computed root; replay validation of the
stored residual, which is cancellation-dominated, then failed on every cache
reuse even though the stored mathematics was correct. Root refinement
payloads and eigenpair diagnostics now use the same exact-round-trip width
the deterministic-reduction encoding has always used, and regression tests
pin both the width formula and bit-exact round-trip at the claim precisions.

Root and eigenvector payload values were verified to already round-trip
exactly; the affected surfaces are certification interval strings and solver
diagnostics. An ignored diagnostic test replays a published root-range
payload against its published eigenpair through the production functions, so
a reuse-validation failure can be attributed to a specific artifact rather
than reasoned about.

### Compatibility

Additive release. No existing mathematical semantic key, artifact schema, cache
payload, or compatibility floor is repurposed. Existing claim scripts retain
their distance and fixed-guard root behavior. Root-producing scripts opt into
adaptive v9 with `HighPrecConfig::with_adaptive_root_precision()`.

The `ccm-distance` family retains its 0.14.0 floor; only
`ccm_distance_resolution_evidence`, `ccm_target_residual_analysis`, and
`ccm_deviation_decomposition` start at 0.14.1. All three kinds are registered in the existing family and its existing
public/private shards, so no previously published artifact is affected. Which
lane a measurement reaches remains a publication-time decision.

`ccm_root_conditioning_analysis` starts at 0.14.1 and is routed through the
existing `ccm-evidence` family. Its secular-scale extension is separately keyed
as v2 and does not alter an existing v1 record. Adaptive root schema 5 remains
in the existing `ccm-roots` family; it reuses upstream matrices, eigenpairs, and
secular sources while retaining a distinct source-bound identity.

`ccm_prime_power_response_analysis`, `ccm_u_flow_response_analysis`, and
`ccm_sector_gap_certificate` start at 0.14.1 in the same existing
`ccm-evidence` family. Each has an independent semantic key and payload, and
none requires a new artifact repository. Both response kinds ship directly as
schema/semantics v2; the earlier unguarded development identity is superseded
and cannot satisfy the released request. The sector certificate also has a
0.14.1 reader floor because older readers do not know its exact replay schema.

Published cache batches no longer carry a hardcoded commit identity. Every
site that constructs the Git transport for publication resolves the author
name and email the same way, defaulting to the authenticated principal and its
GitHub no-reply address, so published commits are attributed to that account
instead of appearing as an unlinked author. This covers the private
publication lease ledger carried on the coordination branch, which is written
by a different path from the family batches. `XC_PUBLISH_AUTHOR_NAME` and
`XC_PUBLISH_AUTHOR_EMAIL` override both; an account whose no-reply address
carries a numeric user prefix should set the latter so published commits match
its other commits exactly.

The dependency-preflight transport only reads refs and never publishes, so it
carries an explicitly invalid address and cannot mint a commit that resembles
a published one. A regression test asserts that no hardcoded publication
identity appears anywhere in the publishing module.

Cache acceptance no longer compares an artifact's producer version against the
running toolkit's release line. That comparison, introduced in the v0.13.0
rebuild, was redundant with the explicit compatibility machinery and would have
invalidated every v0.13.x artifact at this version bump. Producer age is
governed where it always was originally: the per-family
`minimum_producer_version` floors enforced during canonical manifest
validation, and each artifact's own declared reader range. This restores the
pre-0.13 floor-based contract — an artifact remains valid under every later
toolkit until a floor is deliberately raised. Artifacts produced by v0.13.x
therefore remain reusable under 0.14.0 without migration, and artifacts
produced by 0.14.0 retain the established minimum reader version.

## 0.13.5

Version 0.13.5 is a payload-preserving performance and observability release.

- High-precision CCM component construction writes directly into final matrix
  storage. This removes full-size intermediate coordinate and result
  collections without changing the established MPFR operation order.
- `XC_PERF_REPORT` enables a process-wide JSON timing sidecar for controlled
  performance studies. Reporting is lazy when disabled, aggregates nested and
  concurrent work, and remains outside cache identity, payloads, manifests,
  publication, and validation evidence.
- An opt-in Gauss--Legendre root schedule can use idle Rayon capacity when a
  cold precompute batch contains too few independent tables. The owning planner
  selects table-level or root-level parallelism, never both.
- Experimental root-level scheduling fails closed on WSL, Windows, and macOS.
  It remains disabled by default and requires native-Linux qualification before
  production use.

### Compatibility

The release does not alter mathematical semantic keys, artifact schemas,
precision targets, solver selection, convergence rules, or default scheduling.
The new runtime-policy field defaults to `false` and is omitted from serialized
policy in that state. Existing compatible cache artifacts and established
default provenance therefore remain reusable without migration.

Exact reference tests compare the optimized component paths with their frozen
materializing counterparts at multiple precisions and Rayon worker counts.
`XC_CACHE_MODE=verify` remains the release authority for exact artifact-payload
comparison during production qualification.

See [Performance reporting](PERFORMANCE_REPORTING.md) for controlled benchmark
guidance and the scope of recorded diagnostics.
