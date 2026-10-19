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
- Tested with system FLINT 3.0.1 on Ubuntu 24.04.

## Contribution and distribution review

Dependency or source-code additions follow the review requirements in
[CONTRIBUTING.md](../CONTRIBUTING.md). Expanding the FLINT ABI, vendoring code,
static linking or redistributing native libraries requires a review of the
new incorporation and distribution scope.

## Exact products and optional CUDA integration

The stored-dyadic conversion, exact integer CPU oracle, modular residue/product
tree and CRT code, binary HP adapter, and typed two-pass Tau decoder are original
implementations developed with AI assistance. The integer/CRT identities and
binary scaling are implemented by project-owned Rust code; no third-party matrix
kernel source, research dataset, or external solver implementation is copied.
The HP path reuses Rug and its existing GMP/MPFR dependencies.

`xc-numerics/modular-product` adds optional direct dependencies pinned to
`num-bigint` 0.4.8, `num-integer` 0.1.46, and `num-traits` 0.2.19. These package
versions already occur in the workspace lockfile. All three declare
`MIT OR Apache-2.0`; their original notices and distribution terms apply.
They supply integer and numeric traits, while the toolkit owns residue selection,
product bounds, reconstruction, resource checks, and backend selection.

`xc-numerics/cuda-product`, also enabled by `hp`, uses separately installed NVIDIA CUDA runtime and
cuBLAS shared libraries. Its original, narrow Rust FFI follows the documented
ABI for allocation/copy/free, cuBLAS handle lifetime, and `cublasGemmEx` integer
products. Sources reviewed are NVIDIA's [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html)
and [CUDA runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html).
No NVIDIA implementation, header, SDK package, or binary is vendored or bundled;
no CUDA source is compiled by this feature. The original runtime-loader code
resolves CUDA 12/13 and cuBLAS shared-library entry points from the user's
installation; neither library is a build/link dependency. Device selection,
per-device handles and buffers, and the prime-task scheduler are project-owned
implementations. CPU execution remains available without a CUDA installation.

NVIDIA software remains subject to its separate
[CUDA license terms](https://docs.nvidia.com/cuda/eula/index.html). This integration
does not grant redistribution rights or authorize bundling vendor libraries.
Binary distribution must disclose the optional runtime dependency and undergo
a fresh package-content/license review before any vendor component is included.
The toolkit's source-available license and contribution authorization requirements
remain unchanged. ABI and integer-GEMM behavior require qualification against the
actual installed driver, libraries, and hardware; numerical tests alone do not
establish compatibility across every CUDA version or device.

## Retained-source diagnostics in v0.15.0

Prefix LDLT, block-inverse trace recurrences, normalized checkpoint exports
and preservation checks use algebraic identities and existing toolkit
numerical/cache interfaces. Implementation and tests were developed with AI
assistance. No external source implementation or unpublished research payload
is incorporated in this extension.

The exact-rational Gauss-Jordan oracle is implemented separately from the
prefix recurrence. Tests use synthetic positive-definite matrices and small
public CCM configurations; this algorithmic comparison does not imply an
independent human review or replication of a research dataset.

This extension adds no dependency requirement or locked external package.
It reuses Rug, MPFR/GMP, Rayon, serde and the optional dynamically linked FLINT
integration described above.

## Implementation provenance

Numerical repairs and extensions after v0.15.0 (exact and directed inertia
counts, interval and domain guards, scaled inverse iteration and Jacobi
rotations, Sonin compression, reduction and operator comparisons, root
execution evidence, checked report/LU/quadrature boundaries, Mellin kernels
and scans, CCM rank-one and native secular routes, retained-data arithmetic,
schema contracts, and the external target protocol checks) are original
implementations developed with AI assistance. No external source
implementation was copied.

Mathematical definitions are taken from the cited publications:

- Suzuki screw kernel: Suzuki (2023), Eq. (1.1), https://doi.org/10.1112/jlms.12785 .
- Maynard central projectors: the border-strip rule in Richard Stanley's
  publication, section 7, https://math.mit.edu/~rstan/pubs/pubfiles/59.pdf .
- Reconstruction, prolate operator and CCM rank-one finite operator: Connes,
  Consani and Moscovici, Zeta Spectral Triples, arXiv:2511.22755v1, sections 5
  and 7 (Lemma 5.4), https://arxiv.org/html/2511.22755v1 .
- Kernel definition: Yakaboylu, arXiv:2408.15135v15, Eq. (52),
  https://arxiv.org/html/2408.15135v15#S4 . This does not endorse or validate
  the paper's full operator/RH argument.

### Development-only reference tools

SymPy, mpmath, Python `fractions`, Arb, and PARI/GP were used to generate
independent test references. They are development tools, not runtime
dependencies, and their use does not constitute independent human review.

The PARI/GP eigenvalue fixture is generated by an original project-authored GP
script from four exact rational matrices; PARI/GP 2.15.4 computes real roots of
the exact characteristic polynomials at 2,000 decimal digits, independently of
the toolkit's MPFR eigensolver. The fixture contains numerical outputs and
project-authored inputs only. PARI/GP is licensed GPL version 2 or later; it is
neither vendored nor redistributed, and no runtime dependency is added. See the
[fixture provenance](../crates/xc-numerics/tests/fixtures/README.md) and
[PARI function contract](https://pari.math.u-bordeaux.fr/dochtml/html/Polynomials_and_power_series.html#polrootsreal).
Fixture SHA-256 `491cd23a2f9045eeb462d0c275dfdd883439684be1a39afd84fee3b4dd936797`;
generator SHA-256 `29a13d35759e302dbf17842f430db3f9bbde59c3b67c51bc72056343aae4758d`.

### Rust dependencies added after v0.15.0

- `xc-solver` depends directly on `num-bigint` 0.4.8 (MIT OR Apache-2.0), already
  present in the workspace and external-consumer lockfiles. It supplies integer
  arithmetic; the toolkit owns the matrix and count logic.
- `xc-spectral` depends directly on `getrandom` 0.3.4, already pinned in the
  workspace, to bind each external target protocol request to 256 bits of
  operating-system entropy.

Neither addition introduces a new resolved package version; the original
dependency notices and distribution terms continue to apply.

### Native library linkage

The optional `arb` feature uses shared system FLINT, GMP, and MPFR throughout
the dependency graph, including Rug. The direct optional `gmp-mpfr-sys` 1.7
dependency enables its `use-system-libs` feature; this is the same resolved
package already supplied through Rug, with unchanged distribution terms.
Builds without Arb retain Rug's bundled GMP 6.3.0 and MPFR 4.2.2.
The qualified Arb system stack uses FLINT 3.0.1, GMP 6.3.0, and MPFR 4.2.1.

The Arb build requires matching development headers and runtime libraries and
executes a probe requiring thread-safe MPFR. The bridge requires TLS-enabled
FLINT headers at compile time. It checks versions, ABI,
exponent context, and serial FLINT settings before generation and proof replay.
Deployment must preserve one shared GMP/MPFR pair, without bundled native symbol
definitions or allocator/interposition overrides. Numerical tests alone do not
establish this packaging contract. See the
[native build and linkage policy](../crates/xc-numerics/ARBITRARY_PRECISION_QUADRATURE.md).
