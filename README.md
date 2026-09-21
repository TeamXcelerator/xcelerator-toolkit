# Xcelerator Toolkit

> Reusable Rust libraries for high-precision numerical research in analytic
> number theory, spectral methods, variational problems, and adjacent mathematics.

- **Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
- **ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)
- **Contact:** randrewsmath@gmail.com

## Retained research with v0.15.1

Xcelerator Toolkit supplies high-precision numerical libraries and reusable,
source-bound research data. Applications can retain a primary calculation once
and use Ultra capture to preserve every applicable implemented diagnostic.
Thirty new artifact kinds cover state geometry, transforms, root responses,
energy and tail models, reference comparisons, signed bands and finite enclosures.

Capture records missing inputs, failed calculations and unresolved numerical
rows explicitly. A completed capture is separate from numerical acceptance.
Campaign target evaluators can be supplied independently through the generic
runtime interface. These diagnostics do not constitute a proof of CCM
convergence or RH.

- [Runtime targets and external providers](docs/RUNTIME_TARGETS.md)
- [Ultra coverage, external inputs and recovery](docs/ULTRA_COMPLETENESS.md)
- [Add diagnostics to existing results](docs/RESEARCH_BACKFILL.md)
- [Release changes and compatibility](docs/RELEASE_NOTES.md)
- [Local validation record](docs/VALIDATION.md)

Large authenticated atom tables, fixed cutoff studies, signed per-ordinal atom
sums and searchable scalar exports are described in the
[atom research guide](docs/ATOM_RESEARCH.md).

## Getting started

The minimum supported Rust version is 1.98. Development uses the stable
channel selected by `rust-toolchain.toml`; v0.15.1 was qualified with Rust
1.98.1. Record the exact compiler version for reproducible experiments.

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

## Capture and interpret research data

Applications integrate `RetainedCcmRun` with the shared `CcmCapturePlan::ultra`.
Updating the library does not change an application's independently hard-coded
capture list. See [capture integration](docs/CAPTURE_LEVELS.md).

Supply a data-only reference file through `XC_RESEARCH_REFERENCE_FILE` when
reference-dependent measurements are needed. The [input example and schema](docs/ULTRA_COMPLETENESS.md#external-reference-preparation)
explain the format. External campaign evaluators are supplied and explicitly authorized by the caller; their implementations are not distributed with the Toolkit.
A reference, atom model or independent configuration cannot be inferred when
its required data is absent.

| Data group | Inputs beyond a retained primary state | Interpretation |
|---|---|---|
| Geometry, origin moments, finite transforms | Retained roots for root-indexed rows | Point measurements with precision/cancellation status |
| Energy, forcing, prefixes and root response | Exact parent matrices and applicable response sources | Finite-source diagnostics with explicit hypotheses |
| Reference projection and signed channels | Identified external reference and basis/jets | Reference-dependent comparison, including unresolved normalization |
| Bands and tail-model energy | Signed atoms or finite zero/tail/lattice forms | Declared finite model; unknown infinite tails stay unknown |
| Cross-configuration comparison | Independently retained compatible states | Measured C/N/P/quadrature changes, not a fitted convergence law |
| Transform enclosures | Arb; optional matching finite source certificate | Finite-function enclosures with source scope stated |

Inspect receipt `numerical_coverage` as well as execution outcomes. Retained,
resolved, qualified and unresolved rows are counted separately. An unavailable
optional input does not invalidate the primary measurement or stop independent
diagnostics. Resource caps and checkpoints are described in the
[Ultra guide](docs/ULTRA_COMPLETENESS.md#resources-progress-and-recovery).

[Additive backfill](docs/RESEARCH_BACKFILL.md) can derive new children from
retained sources without repeating the primary solve or replacing old payloads.
Changing a target or increasing a diagnostic budget creates new source-bound
results; historical receipts keep their original meaning.

## Publication and recovery

Publication reuses verified compressed packages, avoids Git delta searches and
avoids recompressing canonical archive parts during Git import. Verified loose
Git blobs can reuse their SHA-256 check while their storage metadata remains
unchanged. Metadata keeps ordinary Git compression; initial integrity checks,
atomic publication and lease fencing remain enabled.

[Operational reports](docs/PUBLICATION_PERFORMANCE.md) retain per-attempt phase
timings and byte counts separately from scientific artifacts. Preserve the run
journal and staging directory after an interruption. Use
[publication-only recovery](docs/PUBLICATION_RECOVERY.md) to resume missing work.
A numerical claim does not need to be rerun just because publication stopped.

- [Detailed cache and target configuration](docs/CACHE_AND_TARGET_CONFIGURATION.md)
- [Numerical compatibility and historical artifacts](docs/NUMERICAL_COMPATIBILITY.md)
- [Performance measurements and their scope](docs/PUBLICATION_PERFORMANCE.md)

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

See the [documentation index](docs/README.md), [release notes](docs/RELEASE_NOTES.md), and [historical artifact fabric](docs/ARTIFACT_FABRIC_HISTORY.md).

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
  version = {0.15.0},
  year    = {2026},
  url     = {https://github.com/TeamXcelerator/xcelerator-toolkit}
}
```

---

## License

Copyright © 2026 Ronnie Andrews, Jr. / Team Xcelerator Inc. All rights reserved except for the permissions expressly granted in [LICENSE](LICENSE).

This is source-available software, not an open-source license. Reading the repository does not grant permission to modify, redistribute, incorporate, or commercially use the software beyond the license terms.

Repository: <https://github.com/TeamXcelerator/xcelerator-toolkit>

Large retained artifacts can use [explicit resource limits and publication-only recovery](docs/PUBLICATION_RECOVERY.md). This preserves numerical results and the original capture history.
