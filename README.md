# Xcelerator Toolkit

> Reusable libraries for high-precision numerical research in analytic
> number theory, spectral methods, and adjacent areas.

**Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.Â®)  
**ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)  
**Contact:** randrewsmath@gmail.com

## Crates

This is a Cargo workspace containing three sub-crates:

| Crate | Purpose |
|---|---|
| [`xc-numerics`](crates/xc-numerics) | High-precision numerical primitives: GL quadrature (f64 + HP with `<cwd>/data/gl_cache/` disk cache, JSON or zip-compressed), LU factorization, inverse iteration, root-finding, prime sieve, HP symmetric eigendecomposition, HP formatting / comparison helpers. |
| [`xc-zeta`](crates/xc-zeta) | Riemann zeta function utilities: reference zero loading (HP strings, f64, rug::Float), path-parameterized. |
| [`xc-spectral`](crates/xc-spectral) | Spectral methods: CCM Weil-form construction (f64 + HP), prolate-wave operators (f64 + HP), Mellin transforms (f64 + HP), Yakaboylu W-positivity framework (f64 + HP), Dirichlet L-function extensions. |

### Module inventory

**xc-numerics:**
- `quadrature` â€” Gauss-Legendre at f64 (configurable N-point) and HP. The HP path caches nodes/weights to `<cwd>/data/gl_cache/` and supports both uncompressed JSON and zip-compressed JSON fixtures (auto-decompressed on first read). A [`CacheMode`] parameter on `gauss_legendre_nodes(n, prec, mode)` selects the lookup strategy: `Off` (always compute), `JsonOnly` (local `.json` only), `JsonZip` (local `.json` then `.json.zip`), or `DynamicFetch` (default â€” local, then a remote download of the specific fixture from the public [`xcelerator-gl-cache`](https://github.com/TeamXcelerator/xcelerator-gl-cache) repo via `curl`, then compute). Remote fetch fires only on local cache miss and falls through to compute if `curl`/network/the fixture is unavailable. Per-cwd layout means each paper repo / reproduction script gets its own independent cache, and pre-computed cache fixtures can be checked into a repo to skip the cold-start cost of Newton iteration. Cache hits are structurally validated (Î£w = 2, Î£xÂ·w = 0, antisymmetric nodes); corrupt or wrong-precision files are skipped with a stderr warning. Public audit API: `verify_gl_cache_dir`.
- `root_finding` â€” f64 bisection with configurable tolerance and max iterations. Endpoint-zero handled correctly (returns the zero endpoint, no walk-away).
- `primes` â€” Sieve of Eratosthenes, prime counting function Ï€(x).
- `linalg` (HP-gated) â€” Dense LU factorization with partial pivoting, LU solve, banded tridiagonal LU (Thomas with partial pivoting; O(n) factor and solve), inverse iteration (with optional forced-even projection; rustdoc documents both convergence floors), â„“Â² normalization, Rayleigh quotient. Inner reductions and matvec parallelized. `lu_solve` parallelizes its inner triangular-solve reductions by default; `lu_solve_with(..., parallel)` exposes a serial/parallel toggle for tiny matrices or deterministic single-threaded benchmarking. The parallel reductions in `lu_solve`, `normalize_l2`, and `rayleigh_quotient` use a parallel map followed by a fixed index-order sequential fold (not rayon `.reduce()`), so HP results are **bit-identical run-to-run** despite the non-associativity of HP addition â€” required so the Weil eigenvector Î¾ is reproducible and cacheable.
- `fmt` (HP-gated) â€” `display_hp` (decimal scientific notation at any sig-digit count, no f64 underflow), `sign_of` (HP sign without f64), `matching_digits` and `relative_difference` (HP comparison helpers). Use these wherever you'd otherwise call `to_f64()` for display or comparison.
- `eigen` (HP-gated) â€” HP symmetric eigendecomposition: `tridiag_eigenvalues_hp` (symmetric tridiagonal QR with implicit Wilkinson shifts; allocation-optimized inner loop), `tridiag_eigenvector_for_value_hp` (shifted inverse iteration with `TridiagEigvecOptions { max_steps, early_termination, solver: Banded | Dense }`), `dense_symmetric_eigenvector_for_value_hp` (shifted inverse iteration on dense input), `householder_tridiag_hp` (dense â†’ tridiagonal reduction with parallel reductions, matvec, symmetric update, and Q accumulation), `dense_symmetric_eigenvalues_hp` (full pipeline). Truly dynamic in working precision (verified at HP-1000 against PARI/GP 2000-digit reference for both dense and tridiagonal cases; matches to â‰¥500 digits across 9 reference matrices including Hilbert and Wilkinson W11).

**xc-zeta:**
- `zeros` â€” Load reference zeros as HP strings, f64, or `rug::Float`; path-parameterized for flexibility.

**xc-spectral:**
- `ccm` â€” CCM construction: `CcmParams`, `CcmResult`, `prime_powers_up_to`, `run_f64`, `solve_spectrum_f64`.
- `ccm::hp` (HP-gated) â€” `HighPrecConfig`, `HighPrecResult`, `run`, `save_xi_json`, `load_xi_json`, `measure_evenness`, full Weil-form matrix assembly at arbitrary precision. The Ï„-matrix construction is cached automatically to `<cwd>/data/tau_cache/` (uncompressed JSON, single zip, or byte-split `.partXX` for files exceeding GitHub's 100 MB limit). The smallest-eigenvalue Weil eigenvector Î¾ is cached to `<cwd>/data/weil_eigvec_cache/` (uncompressed JSON or single zip â€” Î¾ is small, â‰²2 MB, so no byte-split tier), governed by the same `CacheMode` as the GL/Ï„ caches with a remote-fetch tier from the public [`xcelerator-weil-eigvec-cache`](https://github.com/TeamXcelerator/xcelerator-weil-eigvec-cache) repo. The Î¾ cache check sits *after* the Ï„ build so a cached `(Î¾, Îµ_N)` is validated against the in-hand Ï„ via the eigen-residual `â€–Ï„Î¾ âˆ’ Îµ_NÂ·Î¾â€–`; a hit skips the dominant `O(NÂ³)` LU factorization. Î¾ is bit-reproducible run-to-run because the inverse-iteration reductions (`lu_solve`, `normalize_l2`, `rayleigh_quotient`) use a fixed index-order fold rather than a runtime-ordered parallel reduction. Symmetrize loop, Newton-per-seed loop, and evenness reduction are all parallelized. Public audit API: `verify_tau_cache_dir`.
- `prolate` â€” Prolate-wave operator PW_Î». f64 prototype (`build_pw_matrix_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`) and HP submodule `prolate::hp` (`build_pw_matrix`, `compute_k_lambda`, `compare_xi_to_k_lambda`) using the HP eigensolver from `xc-numerics::eigen`. The eigenvalue spectrum from `tridiag_eigenvalues_hp` (the dominant cost in `compute_k_lambda` at HP-1000) is cached to `<cwd>/data/prolate_eigvals_cache/`. HP u-grid evaluation parallelized. Public audit API: `verify_prolate_eigvals_cache_dir`.
- `mellin` â€” Truncated completed eta function `Î›_Î»(s)`, Î¾-weighted Mellin `G(s)`, parallelized critical-line zero scanner. Full f64 (`*_f64`) and HP (`*_hp`) parity: `omega_f64` / `omega_hp`, `truncated_lambda_f64` / `truncated_lambda_hp`, `xi_weighted_mellin_f64` / `xi_weighted_mellin_hp`, `scan_critical_line_zeros_f64` / `scan_critical_line_zeros_hp`. The HP scan also runs in parallel.
- `yakaboylu` â€” Yakaboylu's Hilbert-PÃ³lya framework. f64 prototype (`v_r_matrix_element_f64`, `build_w_matrix_f64`, `test_w_positivity_f64`, `WPositivityResultF64`) and HP submodule `yakaboylu::hp` (`build_w_matrix`, `test_w_positivity`, `HpWPositivityResult`) using `dense_symmetric_eigenvalues_hp`. HP outer-row matrix build is parallelized.
- `lfunction` â€” Dirichlet L-function character specs (Ï‡â‚ƒ, Ï‡â‚„, Ï‡â‚…, Ï‡â‚‡), twisted prime-power enumeration. `chi_at` and `chi_at_prime_power` return exact `i8` values (precision-agnostic) alongside the `_f64` variants.

## Tests

All magic numbers are extracted to documented public constants. All
public APIs have unit tests.

```bash
# f64-only (Windows/Linux/macOS â€” no system dependencies):
cargo test --workspace
# 54 tests pass, 0 ignored

# Full HP tier (Linux/WSL/macOS â€” requires libgmp-dev libmpfr-dev libmpc-dev):
cargo test --workspace --features hp
# 171 tests pass, 2 ignored (PARI-fixture heavy tests);
# plus 1 ignored live-network test (remote_fetch_live) â€” run it with:
#   cargo test -p xc-numerics --features hp -- --ignored remote_fetch_live
```

### HP eigensolver verification (3 layers)

The HP symmetric eigensolver in `xc-numerics::eigen` is verified at three
independent levels:

1. **Closed-form structured matrices** (Strang's tridiagonal, Hilbert,
   rotated diagonal, clustered eigenvalues, Wilkinson W21) â€” closed-form
   eigenvalues at HP-256 and HP-1000.
2. **PARI/GP cross-check** â€” `tests/eigen_reference.rs` loads
   `tests/fixtures/eigen_reference.json` (9 reference matrices generated
   by PARI at 2000-digit precision via `polrootsreal(charpoly(M))`) and
   verifies every eigenvalue matches our HP-1000 result to â‰¥500 decimal
   digits.
3. **Property-based** â€” random symmetric matrices (deterministically
   seeded HP LCG, no f64) verified to satisfy trace, determinant,
   eigenequation, normalization, orthogonality, and decomposition
   properties.

To regenerate the PARI fixture (only needed if the test cases change):

```bash
sudo apt install pari-gp  # if not already installed
cd crates/xc-numerics/tests/fixtures
gp -q generate_eigen_reference.gp > eigen_reference.json
```

The committed `eigen_reference.json` is 448 KB; the test passes vacuously
on machines without PARI but verifies fully wherever the JSON is present.

## Performance

The HP code paths are parallelized with [rayon](https://github.com/rayon-rs/rayon)
throughout. Most parallelization is unconditional â€” there are no `if n >
threshold` guards â€” with one exception: `lu_solve` runs its inner
triangular-solve reduction serially for short rows (below a small fixed
threshold) where rayon's dispatch overhead would exceed the work, and in
parallel for longer rows. Small-n tests pay a small constant overhead, but
production workloads scale across all available cores.

| Layer | Parallelized hot spots |
|---|---|
| `xc-numerics::eigen` | Householder reduction: â€–xâ€– and â€–vâ€–Â² reductions, matvec `p = Î²Â·AÂ·v`, symmetric rank-2 update `A â† A âˆ’ vÂ·qáµ€ âˆ’ qÂ·váµ€`, váµ€p reduction, Q accumulation `Q â† Q Â· H`. |
| `xc-numerics::linalg` | `lu_factor` Schur-complement update; `lu_solve` inner forward/back-substitution reductions (per-row, length-thresholded); `normalize_l2` (parallel sum-of-squares + per-element divide), `rayleigh_quotient` (parallel row evaluation + final reduction), `inverse_iteration` initial guess and forced-even projection. |
| `xc-spectral::ccm::hp` | `run` and `measure_evenness` symmetrize loops (parallel pair compute, sequential write to avoid mirror-cell aliasing), Newton-per-seed loop in `run` (~50 independent refinements), combined `diff_sq + norm_sq` reduction in `measure_evenness`. |
| `xc-spectral::yakaboylu::hp` | `build_w_matrix` outer-row loop. |
| `xc-spectral::prolate::hp` | `compute_k_lambda` u-grid evaluation, `compare_xi_to_k_lambda` Î¾-value reconstruction and dot reductions. |
| `xc-spectral::mellin` | Critical-line scan grid evaluation in both `scan_critical_line_zeros_f64` and `scan_critical_line_zeros_hp`. |

The toolkit ships HP-everywhere by policy. f64 fast paths exist where
explicitly requested (suffixed `_f64`); they remain useful for
quick-iteration smoke tests but cannot reach the precisions needed for
publication-grade convergence claims.



In your `Cargo.toml`:

```toml
[dependencies]
xc-spectral = { git = "https://github.com/TeamXcelerator/xcelerator-toolkit", subpath = "crates/xc-spectral" }
xc-zeta     = { git = "https://github.com/TeamXcelerator/xcelerator-toolkit", subpath = "crates/xc-zeta" }
xc-numerics = { git = "https://github.com/TeamXcelerator/xcelerator-toolkit", subpath = "crates/xc-numerics" }
```

Pin to a specific commit for reproducibility:

```toml
xc-spectral = { git = "https://github.com/TeamXcelerator/xcelerator-toolkit", rev = "<commit-sha>", subpath = "crates/xc-spectral" }
```

## Build

```bash
cargo build --workspace --release
cargo build --workspace --release --features hp
```

System dependencies for HP tier:
```bash
sudo apt install build-essential m4 libgmp-dev libmpfr-dev libmpc-dev
```

## HP / f64 boundary policy

The toolkit follows a strict rule: **HP everywhere unless f64 is
explicitly requested**. f64 underflows below ~10â»Â³â°â¸, which is
routinely violated by HP values our papers produce (e.g. eigenvalues
of magnitude 10â»Â¹â°â°â° at Î»Â²=1000).

### Naming convention

Every public function that operates at f64 precision has `_f64` in its
name. There is no ambiguity at the call site: if you don't see `_f64`,
the function is HP (or precision-agnostic).

| Pattern | Examples |
|---|---|
| `_f64` suffix â†’ f64-only | `gauss_legendre_64pt_f64`, `bisect_f64`, `omega_f64`, `truncated_lambda_f64`, `xi_weighted_mellin_f64`, `scan_critical_line_zeros_f64`, `solve_spectrum_f64`, `chi_at_f64`, `chi_at_prime_power_f64`, `build_w_matrix_f64`, `smallest_eigenvalue_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`, `run_f64`, `build_tau_f64`, `v_r_matrix_element_f64`, `build_pw_matrix_f64`, `first_n_f64`, `to_f64_result`, `bisect_zero_f64`, `legendre_p_deriv_f64`, `gl_nodes_weights_f64`, `parity_of_f64`, `count_nodes_f64`, `interp_grid_f64`, `build_pw_dense_f64` |
| `_hp` suffix â†’ HP | `omega_hp`, `truncated_lambda_hp`, `xi_weighted_mellin_hp`, `scan_critical_line_zeros_hp`, `tridiag_eigenvalues_hp`, `householder_tridiag_hp`, `dense_symmetric_eigenvalues_hp`, `dense_symmetric_eigenvector_for_value_hp`, `tridiag_eigenvector_for_value_hp` |
| no suffix â†’ HP-default | `ccm::hp::run`, `ccm::hp::measure_evenness`, `inverse_iteration`, `lu_factor`, `lu_solve`, `normalize_l2`, `rayleigh_quotient`, `display_hp`, `sign_of`, `matching_digits`, `relative_difference`, `chi_at` (returns exact `i8`), `chi_at_prime_power` (returns exact `i8`), `gauss_legendre_nodes` (in the `hp` submodule) |

### Allowed f64 in HP code

f64 is permitted in HP-claiming code only at:

- **Documented f64 boundary fields** in result structs (e.g. `CcmResult`,
  `LoadedXi.xi_f64`, `LoadedXi.weil_min_eigenvalue` â€” paired with `xi_hp`
  and `weil_min_eigenvalue_hp` for HP consumers).
- **`to_f64_result()`** â€” explicit lossy conversion.
- **Wall-clock metadata** (`elapsed_seconds: f64`).
- **CLI input parameters** (e.g. `CcmParams::from_lambda(lambda: f64, ...)`).
  These are user inputs at f64 precision by contract; the precomputed
  `lambda_sq_int: u64` field is what HP code paths actually consume.
- **Digit-to-bit conversion** (`bits = digits * DIGITS_TO_BITS_FACTOR`)
  where the output is an integer (`u32`) â€” no precision loss.

### Forbidden in HP code

- `to_f64()` on an HP value used in display, comparison, or computation
  that could underflow. Use `xc_numerics::fmt::display_hp` (formatting),
  `sign_of` (sign inspection), `matching_digits` or `relative_difference`
  (HP-native comparison) instead.
- f64 arithmetic on HP-derived values where the result is consumed
  by HP code (no `(hp_value.to_f64() / divisor).round() as i32` etc.).
- f64 thresholds compared against HP `Float` values without explicit
  HP construction (build the threshold once as `Float::with_val(prec,
  Float::parse("1e-10").unwrap())`).

Full guideline (private):
[`xcelerator-research/research/methods/HP_F64_GUIDELINES.md`](https://github.com/TeamXcelerator/xcelerator-research).


## Version History

- **v0.9.2** — τ-cache remote fetch now checks two repos (`xcelerator-tau-cache` then `xcelerator-tau-cache-2`) to support cache overflow beyond the first repo's storage limit. Adding further overflow repos is a one-line change to `REMOTE_BASES`.

- **v0.9.1** — Zip-only cache: τ, GL, and Weil caches read directly from `.json.zip` in memory; no decompressed `.json` is written to disk, halving local cache storage overhead.

- **v0.9.0** — DynamicFetch cache tier (GL, τ, and Weil eigenvector caches fetch from public repos on demand). Weil eigenvector (ξ) cache with bit-reproducible determinism. `force_even` field on `HighPrecConfig` (natural-eigenvector testing). Cache keys on `force_even` (forced/natural ξ stored separately).

- **v0.8.0** — Parallel `lu_solve` (inner triangular-solve reductions via rayon). Eliminates the last single-core bottleneck in inverse iteration at large N.

- **v0.7.0** — GL cache writes `.json.zip` alongside `.json` (distribution artifact for git checkin).

- **v0.6.0** — Cache infrastructure (GL, τ, prolate), heavy testing audit, API consolidation (`tridiag_eigenvector_for_value_hp` single entry point), rustdoc pass.

- **v0.5.0** — Tridiagonal LU + banded shifted inverse iteration. O(n) per-step solve, memory-efficient eigenvector recovery for HP-scale tridiagonal systems.

- **v0.4.0** — HP-everywhere (no silent f64 leaks), comprehensive rayon parallelization, HP symmetric eigendecomposition (Householder + QR + inverse iteration), `xc-numerics::fmt` module.

- **v0.3.0** — `ccm::hp::measure_evenness()` for eigenvector symmetry measurement.

- **v0.2.0** — Breaking: `ccm::hp::run()` takes `&[Float]` seeds (eliminates f64 truncation in Newton seeding).

- **v0.1.0** — Initial release. CCM construction, prolate, Mellin, Yakaboylu, L-functions, HP numerics.

## Used by

- [`ccm-reproduction-and-convergence`](https://github.com/TeamXcelerator/ccm-reproduction-and-convergence) — Independent reproduction of the CCM zeta spectral triple, with eigenvalue match to 999 digits.
- [`ccm-convergence-rate-falsifications`](https://github.com/TeamXcelerator/ccm-convergence-rate-falsifications) — Empirical convergence-rate study and falsification of proposed convergence-rate predictions.

## License

Source-available for academic verification, study, and citation.
See [LICENSE](LICENSE) for terms.
