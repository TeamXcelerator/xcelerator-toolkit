# Xcelerator Toolkit

> Reusable Rust libraries for high-precision numerical research in analytic
> number theory, spectral methods, variational problems, and adjacent mathematics.

- **Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
- **ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)
- **Contact:** randrewsmath@gmail.com

Version 0.13.0 is a breaking release and supports only the current APIs and cache format.

---

## Key features

- **Compute-first workflow** — request a result and use it. The toolkit reuses a compatible cache entry when available and computes the result when it is not.
- **Computed assurance by default** — ordinary calculations run their normal validation and diagnostics without the substantial additional cost of rigorous certification.
- **High-precision numerics** — GMP/MPFR-based arithmetic, deterministic reductions, structured linear algebra, root finding, and eigensolvers.
- **Research mathematics** — Connes–Consani–Moscovici finite Weil forms, prolate and Mellin methods, Suzuki screw functions, Yakaboylu operators, Dirichlet L-functions, zeta utilities, and Maynard–Tao variational calculations.
- **Reusable artifacts** — versioned, content-addressed local and remote caching with validation before reuse.
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
  version = {0.13.0},
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

See [Research Workflows](docs/v0.13.0/RESEARCH_WORKFLOWS.md) for additional examples and their numerical scope.

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

## Caching

Caching is automatic for ordinary consumers. The toolkit checks compatible local and public artifacts, validates a candidate before reuse, and computes on a miss. No credentials or cache configuration are required for that normal workflow.

Authenticated publication to private or public cache storage is an optional author operation and is disabled by default. See [Cache Schema](docs/v0.13.0/CACHE_SCHEMA.md) for the artifact model.

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

- [v0.13.0 documentation](docs/v0.13.0/README.md)
- [Research workflows](docs/v0.13.0/RESEARCH_WORKFLOWS.md)
- [CLI reference](docs/v0.13.0/CLI.md)
- [Security policy](SECURITY.md)

## License

Copyright © 2026 Ronnie Andrews, Jr. / Team Xcelerator Inc. All rights reserved except for the permissions expressly granted in [LICENSE](LICENSE).

This is source-available software, not an open-source license. Reading the repository does not grant permission to modify, redistribute, incorporate, or commercially use the software beyond the license terms.

Repository: <https://github.com/TeamXcelerator/xcelerator-toolkit>
