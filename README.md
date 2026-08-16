# Xcelerator Toolkit

> Reusable Rust libraries for high-precision numerical research in analytic
> number theory, spectral methods, variational problems, and adjacent mathematics.

- **Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
- **ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)
- **Contact:** randrewsmath@gmail.com

Version 0.13.5 preserves existing cache artifacts and mathematical behavior
while reducing high-precision CCM construction overhead, adding opt-in
performance evidence, and introducing a guarded experimental Gauss--Legendre
schedule. Established scheduling remains the default.

---

## Key features

- **Compute-first workflow** — request a result and use it. The toolkit reuses a compatible cache entry when available and computes the result when it is not.
- **Computed assurance by default** — ordinary calculations run their normal validation and diagnostics without the substantial additional cost of rigorous certification.
- **High-precision numerics** — GMP/MPFR-based arithmetic, deterministic reductions, structured linear algebra, root finding, and eigensolvers.
- **Research mathematics** — Connes–Consani–Moscovici finite Weil forms, prolate and Mellin methods, Suzuki screw functions, Yakaboylu operators, Dirichlet L-functions, zeta utilities, and Maynard–Tao variational calculations.
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
  version = {0.13.5},
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
requested artifact into an isolated cache, follows reference-cache routes and
continuation seeds, compares exact payload bytes, and writes a claim-wide
report. Missing references, mismatches, nondeterministic recomputation, and
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
