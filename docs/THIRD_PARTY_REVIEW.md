# Third-party dependencies and implementation provenance

## Cutoff-free CCM mathematics

- Source reviewed: A. Groskin, arXiv:2607.02828, including the published cutoff-free formulas and stated $c=13,N=4$ and $c=100,N=200$ positivity configurations.
- Incorporation decision: mathematical formulas and public benchmark parameters were independently implemented in `xc-spectral::ccm::cutoff_free`. No ancillary source file, generated matrix, pivot list, certificate, or other artifact from the external submission is stored in or compiled by this repository.
- Independence control: the implementation uses project-owned Rust interval arithmetic and component assembly. Tests regenerate matrices from configuration and formulas; they do not compare against an imported certificate as their oracle.
- Attribution: source identity is retained in module documentation and [References](REFERENCES.md).

## FLINT/Arb system library

- Component reviewed: Ubuntu Noble `libflint18t64` / `libflint-dev` 3.0.1, upstream <https://flintlib.org/>.
- License evidence: Ubuntu's installed machine-readable copyright record identifies the upstream library files as `LGPL-2.1+`; package-specific CMake helpers are BSD-2-Clause and Debian packaging files are GPL-2+. The toolkit neither copies those files nor redistributes the Ubuntu package.
- Integration decision: accepted as an optional user-installed shared-library dependency behind the `xc-spectral/arb` feature. The toolkit dynamically links `libflint`; it does not vendor FLINT source or statically incorporate `libflint.a`.
- Boundary: the project-owned C shim exposes only complex digamma and trigamma interval evaluation through MPFR endpoints. Rust owns input validation, endpoint storage, error propagation, and all remaining CCM formula operations.
- Distribution control: a toolkit binary built with `xc-spectral/arb` has a runtime dependency on the separately replaceable system shared library. Release packaging must preserve this notice, disclose the runtime dependency, and must not bundle FLINT without a fresh package-content and license review.
- Validation environment: Ubuntu 24.04 WSL with system FLINT 3.0.1, MPFR/GMP, `pkg-config`, and Rust 1.98.1 for v0.15.0 qualification.

## Contribution and distribution review

Dependency or source-code additions follow the review requirements in
[CONTRIBUTING.md](../CONTRIBUTING.md). Expanding the FLINT ABI, vendoring code,
static linking or redistributing native libraries requires a review of the
new incorporation and distribution scope.

## Retained-source diagnostics in v0.15.0

Prefix LDLT, block-inverse trace recurrences, normalized checkpoint exports
and preservation checks use algebraic identities and existing toolkit
numerical/cache interfaces. Implementation and tests were developed with AI
assistance. No external source implementation or unpublished research payload
is incorporated in this extension.

The exact-rational Gauss-Jordan oracle is implemented separately from the
prefix recurrence. Tests use synthetic positive-definite matrices and small
public CCM configurations; this algorithmic comparison does not imply an
independent human review or replication of a research dataset. See
[release validation](VALIDATION.md) for the executed checks and scope.

This extension adds no dependency requirement or locked external package.
It reuses Rug, MPFR/GMP, Rayon, serde and the optional dynamically linked FLINT
integration described above.
