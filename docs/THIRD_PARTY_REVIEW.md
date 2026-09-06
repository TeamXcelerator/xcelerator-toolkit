# Third-party review record

## Cutoff-free CCM mathematics

- Source reviewed: A. Groskin, arXiv:2607.02828, including the published cutoff-free formulas and stated $c=13,N=4$ and $c=100,N=200$ positivity configurations.
- Incorporation decision: mathematical formulas and public benchmark parameters were independently implemented in `xc-spectral::ccm::cutoff_free`. No ancillary source file, generated matrix, pivot list, certificate, or other artifact from the external submission is stored in or compiled by this repository.
- Independence control: the implementation uses project-owned Rust interval arithmetic and component assembly. Tests regenerate matrices from configuration and formulas; they do not compare against an imported certificate as their oracle.
- Attribution: source identity is retained in module documentation and owner-managed review evidence.

## FLINT/Arb system library

- Component reviewed: Ubuntu Noble `libflint18t64` / `libflint-dev` 3.0.1, upstream <https://flintlib.org/>.
- License evidence: Ubuntu's installed machine-readable copyright record identifies the upstream library files as `LGPL-2.1+`; package-specific CMake helpers are BSD-2-Clause and Debian packaging files are GPL-2+. The toolkit neither copies those files nor redistributes the Ubuntu package.
- Integration decision: accepted as an optional user-installed shared-library dependency behind the `xc-spectral/arb` feature. The toolkit dynamically links `libflint`; it does not vendor FLINT source or statically incorporate `libflint.a`.
- Boundary: the project-owned C shim exposes only complex digamma and trigamma interval evaluation through MPFR endpoints. Rust owns input validation, endpoint storage, error propagation, and all remaining CCM formula operations.
- Distribution control: a toolkit binary built with `xc-spectral/arb` has a runtime dependency on the separately replaceable system shared library. Release packaging must preserve this notice, disclose the runtime dependency, and must not bundle FLINT without a fresh package-content and license review.
- Validation environment: Ubuntu 24.04 WSL with system FLINT 3.0.1, MPFR/GMP, `pkg-config`, and the declared Rust 1.85 minimum toolchain.

## Workspace-wide review gate

The owner maintains a machine-readable review record for every direct external
Rust dependency, separately installed native dependency, external validation
tool, and named published algorithm source used in v0.14.0. Records bind exact
manifest requirements, license expressions, authoritative package pages,
incorporation decisions, and implementation paths. Public contributions must
submit the same evidence through the owner-managed review process described in
`CONTRIBUTING.md`; internal review records are not part of the public source
distribution.

Expanding the FLINT ABI, vendoring upstream code, static linking, redistributing
native libraries, or incorporating any source fragment requires a new review;
the current independent-formula decisions do not authorize those changes.

## 0.14.4 retained-source diagnostic extension

- Owner authorization: Ronnie Andrews, Jr. / TeamXceleratorDev explicitly
  requested all Toolkit audit and research improvements and main-branch
  integration in the September 6, 2026 CCM research conversation.
- Original/generated implementation: fixed-order prefix LDLT, block-inverse
  trace recurrences, normalized checkpoint exports, capture-plan integration,
  and preservation checks were implemented with AI assistance from algebraic
  identities and the existing project-owned numerical/cache interfaces.
  The prototype from the same owner-authorized conversation was promoted and
  tested; no outside source implementation or unpublished campaign payload is
  incorporated. The exact-rational Gauss-Jordan test oracle is a separate
  independently implemented algorithm, not an independent human review.
- Dependencies: no direct or transitive dependency requirement or locked
  external package was added or upgraded by this extension. Existing Rug,
  MPFR/GMP, Rayon, serde, and optional dynamically linked FLINT are reused.
- Validation fixtures: generated synthetic positive-definite matrices and
  small public CCM configurations only. Runtime source snapshots and local
  compiler/native-library bundles used for testing are not tracked or shipped
  in this repository.
- Scope: tests and byte-identity checks support the documented finite sample;
  private release audit, independent reviewer approval, and missing research
  checkpoint reproduction are not asserted. Existing owner review policy
  remains in effect for release approval.
