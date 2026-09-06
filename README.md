# Xcelerator Toolkit

> Reusable Rust libraries for high-precision numerical research in analytic
> number theory, spectral methods, variational problems, and adjacent mathematics.

- **Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
- **ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)
- **Contact:** randrewsmath@gmail.com

## 0.14.4: deterministic diagnostics and retained-source analysis

Version 0.14.4 corrects cutoff-free zero-mode endpoint handling, gives the
corrected certificate its own semantics, and adds precision-aware analytic
series budgets. Ordinary quadrature-derived matrices, eigenstates, and roots
retain their numerical contracts and compatibility floors. Default allocation
and validation-memory improvements preserve the established operation order.
Alternative prime aggregation and quadrature bucketing remain separately
identified opt-in research routes, not automatic capture-level changes.

New APIs provide fixed-order positive-definite prefix ladders, inverse-trace
moments, normalization-explicit checkpoint vectors, and cancellation-aware
post-serialization checks. The versioned `CcmCapturePlan::ultra` includes the
retained-source diagnostic phase without implicitly requesting certification.
The paper applications must adopt both capture phases; upgrading the library
alone does not rewrite their CLI behavior. The new `ccm_prefix_analysis` kind is
private-only and does not replace any existing source artifact.

- [Prefix API, formulas, precision, and capture contract](docs/CCM_PREFIX_ANALYSIS.md)
- [Completed qualification and remaining boundaries](docs/CCM_HARDENING.md)
- [Desktop / VS Code Codex handoff](docs/CODEX_HANDOFF.md)

No private warehouse payloads or registry changes are included. This source
integration is not a release tag, a private-campaign reproduction, or a claim
that every configuration/platform has been certified.

## Existing artifact fabric (0.14.3)

Version 0.14.3 adds backward-compatible multi-shard cache rollover, building on
the CCM capture and managed-publication functionality delivered in 0.14.1.
Publishing children of shard-reused artifacts to a new
destination now stages the full dependency closure without recomputation,
including identity-first dependencies later referenced by their real cache
keys. Author publication also accounts for validated workstation and remote
reuse hits, including strict `require_reuse` runs; destination verification
suppresses redundant commits, while an execution that observed no artifacts
fails instead of reporting a vacuous success. Historical exact dependencies
may be added to immutable repository closure without displacing an
equal-or-newer live semantic-index entry; ordinary producer downgrades remain
forbidden. Publication preflight recognizes those superseded exact identities
only when the retained manifest has a canonical batch proof, matching the
reader's historical-resolution rule. Long destination scans refresh GitHub write evidence before both
candidate authorization and remote mutation, so the five-minute authority
window cannot expire merely while a large existing family is inspected.
Cold multipart reuse batches missing Git blobs before concurrent verified
streaming, verifies reused parts during the reconstruction copy, quarantines
and re-downloads a corrupt reused part once with best-effort quarantine
cleanup, removes a corrupt complete package so it can be rebuilt, and reports fetch,
reconstruction, and decode time separately. Workstation ZIP hits read a
compressed object up to the split part size once, within a bounded
process-wide in-memory allowance, and avoid a redundant second hash. Exact published
identities are looked up through a persistent per-semantic-digest inventory
rather than a directory walk. Exact dependency sets are prepared
progressively in bounded repository batches, with independent shard sessions
allowed to proceed concurrently; complete local packages and retained parts
are excluded from remote preparation, and each filtered blob reserves the
caller's exact retained-part bound rather than a generic 100 MB allowance.
Publication staging reuses verified
encoded packages, stages dependencies from those packages without inflating
their payloads, and directly links their verified split parts when available,
instead of recompressing, recopying, or repeatedly decoding artifacts that are
already present. Direct encoded adoption requires the exact persisted encoder
profile; unprofiled legacy objects remain readable but are re-encoded before
publication. The unchanged single-entry route retains its published V1
transport identity; V2 is reserved for packages that actually contain
multiple items on the corrected ZIP64 route.
Dependency closure resolves each member by exact content
digest, so a newer artifact under the same semantic key no longer blocks
publication of children that name the older one.

Cache transport tuning is explicit and bounded. `XC_CACHE_PREFETCH_CONCURRENCY`
controls independent repository preparation and `XC_CACHE_DOWNLOAD_CONCURRENCY`
controls verified part-download workers (both default to 4 and clamp to 1--8).
`XC_CACHE_SINGLE_PASS_ZIP_BYTES` sets the per-object in-memory ZIP threshold
(default 90 MiB), while `XC_CACHE_IN_MEMORY_ZIP_BYTES` sets the process-wide
allowance shared by all such reads (default 256 MiB). Memory overrides must be
between 1 byte and 16 GiB; invalid or zero values retain the defaults.
Warm distance/profile hits avoid fresh eigensolves, quadrature, and sampling
work; distance capture reuses the exact managed `ccm_weil_eigenpair`, binds
its content digest and dependency closure into every affected artifact
identity, and reuses exact managed Gauss--Legendre artifacts across
configurations and exact eigenfunction values
across nested refinement grids; root refinements share their secular-pole
vector; and invalid capture resolutions or semantically mismatched retained
payloads fail early. Maximum capture adds separate numerical-analysis
artifacts. Target-dependent work reads its definition from the private runtime
path named by `XC_TARGET_SPEC_FILE`; only the specification digest enters cache
identity, and those derived artifacts are private-only. Explicit finite sector certification can additionally retain exact
cutoff-free parity, ordering, simplicity, and positivity evidence. The
corrected distance/profile semantic identities supersede legacy artifacts that
were not bound to their canonical eigenpair; unrelated artifact payloads,
schemas, and numerical definitions are unchanged.

---

## Key features

- **Compute-first workflow** — request a result and use it. The toolkit reuses a compatible cache entry when available and computes the result when it is not.
- **Computed assurance by default** — ordinary calculations run their normal validation and diagnostics without the substantial additional cost of rigorous certification.
- **High-precision numerics** — GMP/MPFR-based arithmetic, deterministic reductions, structured linear algebra, root finding, and eigensolvers.
- **Research mathematics** — CCM finite Weil forms, the CCM target function and weighted eigenfunction distances, prolate and Mellin methods, Suzuki screw functions, Yakaboylu operators, Dirichlet L-functions, zeta utilities, and Maynard–Tao variational calculations.
- **Reusable artifacts** — versioned, content-addressed local and remote caching with validation before reuse.
- **Output-preserving optimization** — cache verification compares recomputed payload bytes with current references, while deterministic HP parallelism and retained validated values reduce avoidable work without changing artifact identities.
- **Optional stronger assurance** — independent cross-checks and replayable finite certificates are available for claims that need them.

Finite computations are always reported with their finite scope. They are not presented as proofs of infinite-dimensional conjectures.

---

## Reporting issues & feature requests

Found a bug, hit a limitation, or have an idea for a new capability?

- [Open a GitHub issue](https://github.com/TeamXcelerator/xcelerator-toolkit/issues)
- Or email `randrewsmath@gmail.com`

Please report proposed changes upstream so they can be reviewed and incorporated consistently for everyone using the toolkit. See [CONTRIBUTING.md](CONTRIBUTING.md) for the project’s authorization and review policy.

## Citing this work

If you use Xcelerator Toolkit in research, please cite the exact version or Git commit. Citation metadata is also provided in [CITATION.cff](CITATION.cff).

```bibtex
@software{AndrewsXceleratorToolkit2026,
  author  = {Andrews, Ronnie, Jr.},
  title   = {Xcelerator Toolkit: High-Precision Numerical Libraries for
             Analytic Number Theory and Spectral Methods},
  version = {0.14.3},
  year    = {2026},
  url     = {https://github.com/TeamXcelerator/xcelerator-toolkit}
}
```

---

## Crates

| Crate | Purpose |
|---|---|
| [`xc-core`](crates/xc-core) | Configuration, precision, assurance, provenance, and result contracts. |
| [`xc-numerics`](crates/xc-numerics) | Numerical primitives, high-precision and interval arithmetic, and linear algebra. |
| [`xc-operator`](crates/xc-operator) | Dense, structured, stored, distributed, and matrix-free operators. |
| [`xc-solver`](crates/xc-solver) | Standard, generalized, selected-spectrum, shift-invert, and restarted solvers. |
| [`xc-root`](crates/xc-root) | Root isolation, refinement, interval Newton, and contour-counting services. |
| [`xc-certify`](crates/xc-certify) | Optional finite-dimensional certificate construction and replay. |
| [`xc-cache`](crates/xc-cache) | Artifact identity, validation, and local or remote reuse. |
| [`xc-spectral`](crates/xc-spectral) | CCM, prolate, Mellin, screw, Yakaboylu, and L-function workflows. |
| [`xc-variational`](crates/xc-variational) | Exact and high-precision Maynard–Tao engines. |
| [`xc-zeta`](crates/xc-zeta) | Zeta reference-data loading. |
| [`xc-cli`](crates/xc-cli) | Cache and research-operation command-line tools. |

## Getting started

The minimum supported Rust version is 1.85.

```bash
cargo build --workspace --release --locked
cargo test --workspace --all-targets --locked
```

Small compiled examples include:

```bash
cargo run -p xc-solver --example plan --locked
cargo run -p xc-root --example bracketed_root --locked
cargo run -p xc-spectral --example ccm_window_plan --locked
```

See [Research Workflows](docs/RESEARCH_WORKFLOWS.md) for additional examples and their numerical scope.

## High precision

Ubuntu and WSL2 Ubuntu are the primary high-precision environments. Install the native libraries, then enable the `hp` feature:

```bash
sudo apt install build-essential m4 libgmp-dev libmpfr-dev libmpc-dev libflint-dev pkg-config
cargo build --workspace --release --features hp --locked
```

FLINT/Arb is used by routes that need ball arithmetic or rigorous interval certification. Ordinary high-precision computation uses the appropriate MPFR-based implementation. Explicit `f64` routes remain available for lower-precision work and are never silently substituted for an HP request.

## Assurance

Assurance is selected per result:

- **Computed** is the default: one identified numerical method with its normal validation and diagnostics.
- **Cross-Checked** requires agreement between genuinely independent algorithms or formulations.
- **Certified** produces replayable exact or interval evidence for a finite claim.

Certification may take far longer than the underlying computation. It is intended for selected claims that require rigorous finite bounds, not as a routine prerequisite for computing, caching, or using most artifacts.

Certificates are separate content-addressed evidence artifacts bound to the exact digest of the computed source they certify. They do not overwrite or relabel computed matrices, eigenstates, or root windows. The same source, target, and certification policy reuses one certificate; a different source digest, target range, precision, or method produces a distinct certificate.

The general CCM API discovers roots independently by default. Explicit
reference-seeded refinement is available for reproduction workflows and must
carry a content-bound reference-dataset identity. Seeded refinements and
independent discovery windows use different artifact kinds and semantic keys;
they may share Tau, eigenstate, and secular-source dependencies but can never
satisfy each other's cache requests. Finite-source root certificates remain
independent of either acquisition policy.

Advanced independent discovery can explicitly retain a signed root window and
can return the finite roots actually found when a request exceeds the positive
finite-source reach. These controls are opt-in; ordinary positive, complete
requests retain their established behavior, semantic keys, payload bytes, and
minimum reader version.

## Caching

Caching is automatic for ordinary consumers. The toolkit checks compatible local and public artifacts, validates a candidate before reuse, and computes on a miss. No credentials or cache configuration are required for that normal workflow.

Authenticated publication to private or public cache storage is an optional author operation and is disabled by default. See [Cache Schema](docs/CACHE_SCHEMA.md) for the artifact model. Toolkit changes can be checked against current cached outputs with [cache output validation](docs/OUTPUT_VALIDATION.md).

Managed GitHub routing is registry-driven. A family may span multiple ordered
repositories after rollover: publication uses the family document's active
writable shard, while reuse and exact dependency checks search it and its
read-only predecessors. The root route can remain on a predecessor as a
compatibility pointer for released single-shard readers.

Version 0.13.1 adds generation-fenced publication leases for private shards. Concurrent authors serialize publication per shard, while each content batch and its coordination heartbeat advance atomically.

Version 0.13.2 adds an automatic high-precision CCM eigenstate policy. It reuses an exact current-\(N\) state when available and can use the nearest compatible lower-\(N\) cached state as a shift-invert Krylov starting point without changing claim scripts.

Version 0.13.3 adds opt-in signed and incomplete independent-root discovery.
The numerical v7 artifact stores one canonical complete finite window per
mathematical domain; requested counts, projections, and shortfall policy are
stored separately in compact evidence. Equivalent advanced requests therefore
reuse one numerical artifact, while all existing positive v6 artifacts remain
readable and retain their original identities.

High-precision CCM eigenstates expose three explicit parity policies.
`even-sector` remains the default and preserves every established v0.13
artifact identity. `natural` performs an unrestricted full-space solve.
`adaptive-even` restores the original full-space inverse iteration with an
even projection applied only when the iterate materially drifts away from
even symmetry. Adaptive artifacts use distinct state, secular-source, root,
and evidence identities; they cannot overwrite or satisfy natural or
even-sector requests.

Version 0.13.4 adds `XC_CACHE_MODE=verify`. Verification recomputes each
requested artifact into an isolated cache, follows reference-cache route
selection, compares exact payload bytes, and writes a claim-wide report.
Persisted eigenstates are always computed from the canonical initial state,
never from cached continuation seeds. Missing references, mismatches, nondeterministic recomputation, and
zero-comparison runs fail the validation without publishing or modifying the
production cache.

The v0.13.4 verification path also isolates execution and reference layer sets
through one tested construction boundary. Claim-wide reports preserve
dependency-cascade diagnostics when an upstream payload changes a downstream
semantic key, including `ReferenceAbsent` descendants inherited from the first
divergence. Hermetic CCM, quadrature, and prolate tests exercise the same
managed-session wiring used by production consumers.

High-precision CCM execution in v0.13.4 removes several sources of redundant
work while preserving the established arithmetic order and exact portable
payloads:

- bounded, indexed parallel decimal encoding and decoding retains input order,
  precision, deterministic lowest-error reporting, and bounded scratch memory;
- cache validators retain their decoded runtime matrices, eigenpairs,
  factorizations, spectra, and root windows instead of decoding them again;
- eigenpair validation reuses its already computed residual, and secular
  evaluation reuses one per-run pole table without changing ordered sums;
- the production borrowed dense HP operator computes independent rows in
  parallel while each row retains its original left-to-right MPFR fold; and
- ordinary positive-prefix and index-range root discovery stops once the exact
  required scan extent is satisfied, while unsatisfied and advanced requests
  still exhaust their established finite domain and preserve existing errors.

These are Category A optimizations: semantic keys, schemas, solver selection,
precision, convergence rules, and computed artifact payload bytes remain
unchanged. Performance comparisons should use identical release builds, claim
arguments, cache modes, and Rayon worker counts; `HighPrecResult::elapsed_seconds`
reports the primary toolkit duration, and `XC_CACHE_MODE=verify` remains the
acceptance authority for payload identity.

Version 0.13.5 extends this payload-preserving work in three areas:

- CCM pole, archimedean, prime, and fused component matrices are written
  directly into their final row slots, eliminating full-size coordinate and
  result collections while retaining each entry's established MPFR statement
  order;
- an opt-in process-wide performance sidecar separates cache resolution,
  high-precision construction, validation, encoding, and storage without
  entering artifacts or provenance; and
- an experimental native-Linux Gauss--Legendre schedule can occupy otherwise
  idle Rayon workers inside a single large table without nesting it beneath
  table-level parallelism.

The release does not change mathematical semantic keys, artifact schemas,
precision targets, solver selection, convergence rules, or default scheduling.
Existing compatible artifacts remain reusable. The new scheduling field is
default-false and omitted from serialized runtime policy at that value, so the
established provenance representation is also preserved.

Set `XC_PERF_REPORT` to an ignored `*.performance.json` path to capture a
process-wide stage breakdown for controlled before-and-after measurements. The
sidecar separates high-precision CCM, Gauss-Legendre construction and codecs,
component assembly, artifact computation, encoding, validation, and local
storage without changing cache or artifact identity. See
[performance reporting](docs/PERFORMANCE_REPORTING.md).

Cold CCM runs whose Gauss--Legendre batch contains too few distinct tables to
occupy Rayon can opt into root-level parallelism through
`HpRuntimePolicy::parallel_gl_roots`. The default is `false`. The owning batch
planner selects table parallelism or root parallelism, never both, and records
the selected schedule in performance output and the requested policy in the
execution fingerprint. The default `false` value is omitted from serialized
runtime policy, preserving the established provenance representation and cache
semantics; the opt-in schedule therefore needs a cold cache for a meaningful
construction benchmark. This option requires native-Linux qualification and
is not supported for WSL, where concurrent GMP allocation has exhibited
non-deterministic allocator failures. The numerical runtime fails closed when
this policy is requested on WSL, Windows, or macOS:

```rust
let mut policy = xc_core::HpRuntimePolicy::default();
policy.parallel_gl_roots = true;
xc_numerics::hp_runtime::run_hp_with_policy(&policy, || {
    // Invoke the high-precision workflow here.
})?;
```

## Target-distance measurement

Version 0.14.1 includes the measurement layer for runtime target-distance work:

- `xc_spectral::target` evaluates a generic Gaussian-polynomial lattice-series
  specification at binary64 and arbitrary MPFR precision. The research
  definition is supplied at runtime through `XC_TARGET_SPEC_FILE`; the public
  source contains no target coefficients or research-specific formula.
- `xc_numerics::grid_integral` provides deterministic uniform-grid integration
  (left/right Riemann, midpoint, trapezoid) on grids uniform in `u` or `ln u`,
  at binary64 and HP. `xc_spectral::distance::WeightedIntegrationRule` selects
  between that family and Gauss--Legendre for every weighted norm and distance,
  so a collaborator's rule can be reproduced exactly rather than approximated.
  Neither family is privileged: Gauss--Legendre converges spectrally on smooth
  integrands but loses that advantage at the derivative kinks introduced by
  absolute residuals at interior sign changes.
- `xc_spectral::distance` reconstructs the normalized even CCM eigenfunction
  `f(1) = 1` from its Fourier coefficients and measures the weighted norm,
  the inter-discretization distance `D_alpha(N, M; lambda)`, and the distance
  to target `d(N, lambda)`. Every result records the quadrature scheme, grid
  variable, resolution, `alpha`, and precision that produced it; a value
  separated from its convention is not comparable and should not be reported.
- Every retained profile, target-distance diagnostic, and inter-discretization
  distance names the exact canonical `ccm_weil_eigenpair` content digest in
  its semantic identity and carries that eigenpair as a manifest dependency.
  The target-distance eigenvalue must replay exactly from that parent; an
  independently approximated sector midpoint is never accepted as the state.
- `xc_spectral::distance::hp::ccm_distance_to_target_hp` measures `d(N, lambda)`
  end to end for one CCM configuration, resolving the even ground state through
  the ordinary reuse-first cache routes; the distance computation itself writes
  nothing.

Measurements can be retained as reusable artifacts. The `ccm-distance` family
holds eigenfunction profiles (`ccm_eigenfunction_profile`), target distances
(`ccm_target_distance`), inter-discretization distances
(`ccm_discretization_distance`), numerical resolution evidence
(`ccm_distance_resolution_evidence`), signed/crossing residual diagnostics
(`ccm_target_residual_analysis`), and projection onto a runtime-supplied
auxiliary profile (`ccm_deviation_decomposition`). Retention of the first four is
opt-in — absent at the ordinary capture levels, requested through
`CcmDistanceCaptureOptions`, and included by
`CcmResearchCaptureOptions::maximum`. The deviation decomposition is opt-in
separately and is excluded from every named capture level, including `maximum`,
because adding a new artifact kind to a named level would break `require_reuse`
reproduction of shards that predate it; request it with
`.with_deviation_decomposition()`. Maximum capture records fixed
coefficient-tail diagnostics and same-rule Q/2Q refinement for each uniform
grid, continuing to 4Q only when the `1e-8` relative tolerance is missed. It
also retains signed, positive, and negative residual mass under every requested
rule, together with sampled extrema, signs, and sign-change brackets.
Gauss--Legendre remains the independent-family cross-check in the target-distance
artifact. The quadrature convention is part of artifact identity, so a
measurement cannot be confused with one taken under a different rule, grid,
resolution, or `alpha`. Retained profiles allow norms and inter-discretization
distances to be recomputed downstream without repeating the spectral solve.
They also allow the three diagnostic kinds to be backfilled after the current
canonical profile/distance pair exists. Legacy unbound profile and distance
identities are superseded rather than reused; their canonical eigenpair parent
can still avoid a fresh eigensolve. Managed target-distance capture resolves Gauss--Legendre nodes and
weights through the existing `gauss_legendre_rule` artifact family, so
configurations with the same order and working precision reuse one exact
table. For left/right/trapezoid resolution evidence, Q/2Q/4Q also reuse an
eigenfunction value only when the refined MPFR abscissa is binary-identical;
midpoint grids remain independent.

The four runtime-target-derived kinds -- `ccm_target_distance`,
`ccm_distance_resolution_evidence`, `ccm_target_residual_analysis`, and
`ccm_deviation_decomposition` -- are private-only. Managed publication routes
them to the private leg: under `Both` they are withheld from the public
destination while public-eligible kinds publish to both, and an explicit
`Public`-only request fails when nothing staged is public-eligible. Public
bootstrap layers will not resolve them. The target-independent `ccm_eigenfunction_profile` and
`ccm_discretization_distance` kinds remain eligible for public publication.

Maximum capture also retains `ccm_root_conditioning_analysis` in the existing
`ccm-evidence` family. For every returned root it records the signed secular
derivative, the absolute secular-term sum, its reciprocal and magnitude
condition estimate, and exact geometry relative to the retained uniform pole
grid. The term sum is the measured cancellation scale needed to study the
root-precision floor. This is a direct MPFR replay from the retained root range
and eigenstate, not interval certification; a later maximum run can create a
missing analysis child without repeating the matrix or root solve.

Secular-root refinement continues to default to
`RootPrecisionPolicy::FixedGuard`, retaining the historical v6/v7 arithmetic,
payload, and cache identity exactly. Adaptive v9 refinement is an explicit
opt-in through `HighPrecConfig::with_adaptive_root_precision()`. It holds the
requested target fixed and widens only the inexpensive root layer when the
initial 64-bit reserve cannot support that target. Every converged v9 root is
checked again at a wider precision at the exact stored root. Its artifact
records the evaluation and verification precisions, escalation count,
correction, stopping reason, and the explicit `exact_stored_point_source`
scope. Adaptive keys bind the exact secular-source content digest. Historical
v6/v7 root artifacts are not promoted or used as adaptive starting points;
every v9 miss follows the same canonical computation path, so reuse, refresh,
and verification cannot produce different payload bytes for one semantic key.

Prime-power response is a separate, potentially expensive opt-in and is **not**
implied by `maximum`. Set
`CcmResearchCaptureOptions::capture_prime_power_response = true`, or call
`.with_prime_power_response()`, to retain
`ccm_prime_power_response_analysis` in `ccm-evidence`. At the observation
cutoff the artifact isolates every active prime power's contribution to
`dQ/du`, `u = log(lambda_squared)`, and transports it through the selected
eigenvalue, the full L2 eigenvector, and every retained root. At a prime-power
edge this is Groskin's rank-one right-minus-left derivative jump. Dense event
matrices are not duplicated: each direction is reconstructed analytically, one
reduced-even-sector bordered factorization is shared by every event, and the
retained full response vectors preserve the data needed for later
coefficient-tail research. Response schema v2 first binds independently
indexed HP Sturm enclosures for the lowest two even-sector eigenvalues,
requires a positive same-sector gap and a selected-state residual small enough
relative to that gap, and records the resulting isolation evidence. An
unresolved crossing fails the capture explicitly instead of emitting a
misleading individual eigenvector derivative. Natural and adaptive-even state
routes are likewise rejected because they do not bind the response to one
parity branch. Existing unguarded v1 response artifacts cannot satisfy v2.

Complete CCM flow is a second, independent opt-in and is also **not** implied
by `maximum`. Set `CcmResearchCaptureOptions::capture_u_flow_response = true`,
or call `.with_u_flow_response()`, to retain
`ccm_u_flow_response_analysis` in `ccm-evidence`. It analytically differentiates
the toolkit's full Tau construction with respect to
`u = log(lambda_squared)`, retaining separate pole, archimedean, aggregate
active-prime, and total channels. Each channel carries its action on the
selected state, eigenvalue response, complete eigenvector response, and
fixed-pole root response. The total root response additionally includes motion
of every secular pole `2*pi*n/u`. This decomposition supports convergence,
cancellation, and dominance studies without treating any one prime-event
formula as the research target. It uses the same v2 even-sector isolation gate
and reduced solve as prime-power response capture; near-crossing ambiguity is a
hard capture error, never a residual-passing response payload.

Finite sector-gap certification is a third explicit opt-in and is likewise
**not** implied by `maximum`. It requires an Arb-enabled build and retained
sector analysis with at least two eigenpairs:

```rust
use xc_spectral::ccm::hp::CcmResearchCaptureOptions;
use xc_spectral::ccm::sector_gap_certificate::CcmSectorGapCertificationOptions;

let options = CcmResearchCaptureOptions::maximum(8)
    .with_sector_gap_certification(CcmSectorGapCertificationOptions::default());
```

The resulting `ccm_sector_gap_certificate` stays in the existing
`ccm-evidence` family. It stores the raw exact cutoff-free Tau intervals and an
assumption-free full-matrix interval-LDLT inertia certificate. Positive
definiteness is derived only from that full-matrix proof. The artifact also
stores exact shifted-inertia enclosures for the lowest two even eigenvalues and
lowest odd eigenvalue, the even-sector gap, signed even-versus-odd separation
bounds, and the finite outcome (`even`, `odd`, or `unresolved`). Those parity,
ordering, and sector-simplicity conclusions explicitly depend on the recorded
premise that the exact closed-form CCM matrix is centrosymmetric; the verifier
derives their canonical parity matrix from the raw inertia-certified matrix.
Numerical sector spectra and native cutoff-free midpoint values are retained
only as discovery guides. A later opt-in run can reuse existing numerical
sector parents and create this child, but a cache miss must still perform the
cutoff-free interval assembly and certification. Its scope is one finite
`(c, N)` matrix, not continuum parity or convergence.

```bash
cargo run -p xc-spectral --example target_distance
```

prints the opaque target-definition digest, a normalized runtime-target table,
the weighted target norm, and the same distance under several quadrature
schemes for line-by-line comparison with an authorized private implementation.

## Validation

Release checks run locally; the repository does not require a hosted GitHub Actions workflow. The core public checks use standard Cargo commands:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Maintainers additionally run private release audits and HP validation before tagging a release.

## Scientific limits and reproducibility

A finite CCM spectrum, finite root window, or other bounded calculation does not prove an infinite-dimensional conjecture. Such a conclusion requires a separate validated convergence argument.

Saved results record the toolkit version, source revision, numerical backend, precision, enabled features, effective configuration, inputs, and execution fingerprint needed to understand and reproduce the computation.

## Documentation

- [Toolkit documentation](docs/README.md)
- [Release notes](docs/RELEASE_NOTES.md)
- [Research workflows](docs/RESEARCH_WORKFLOWS.md)
- [Cache output validation](docs/OUTPUT_VALIDATION.md)
- [Performance reporting](docs/PERFORMANCE_REPORTING.md)
- [CLI reference](docs/CLI.md)
- [Security policy](SECURITY.md)

## License

Copyright © 2026 Ronnie Andrews, Jr. / Team Xcelerator Inc. All rights reserved except for the permissions expressly granted in [LICENSE](LICENSE).

This is source-available software, not an open-source license. Reading the repository does not grant permission to modify, redistribute, incorporate, or commercially use the software beyond the license terms.

Repository: <https://github.com/TeamXcelerator/xcelerator-toolkit>
