# Xcelerator Toolkit

> Reusable libraries for high-precision numerical research in analytic
> number theory, spectral methods, and adjacent areas.

**Author:** Ronnie Andrews, Jr. (Team Xcelerator Inc.®)  
**ORCID:** [0009-0003-9724-3104](https://orcid.org/0009-0003-9724-3104)  
**Contact:** randrewsmath@gmail.com

## Crates

This is a Cargo workspace containing three sub-crates:

| Crate | Purpose |
|---|---|
| [`xc-numerics`](crates/xc-numerics) | High-precision numerical primitives: GL quadrature (f64 + HP with `<cwd>/data/gl_cache/` disk cache), LU factorization, inverse iteration, root-finding, prime sieve, HP symmetric eigendecomposition, HP formatting / comparison helpers. |
| [`xc-zeta`](crates/xc-zeta) | Riemann zeta function utilities: reference zero loading (HP strings, f64, rug::Float), path-parameterized. |
| [`xc-spectral`](crates/xc-spectral) | Spectral methods: CCM Weil-form construction (f64 + HP), prolate-wave operators (f64 + HP), Mellin transforms (f64 + HP), Yakaboylu W-positivity framework (f64 + HP), Suzuki screw function (HP), Dirichlet L-function extensions. |

### Module inventory

**xc-numerics:**
- `quadrature` — Gauss-Legendre at f64 (configurable N-point; `gauss_legendre_64pt_f64`, `gauss_legendre_npt_f64`, `gl_nodes_weights_f64` for callers that need raw nodes/weights) and HP. The HP path caches nodes/weights to `<cwd>/data/gl_cache/` and supports both uncompressed JSON and zip-compressed JSON fixtures. A [`CacheMode`] parameter on `gauss_legendre_nodes(n, prec, mode)` selects the lookup strategy: `Off` (always compute), `JsonOnly` (local `.json` only), `JsonZip` (local `.json` then `.json.zip`), or `DynamicFetch` (default — local, then a remote download from the public [`xcelerator-gl-cache`](https://github.com/TeamXcelerator/xcelerator-gl-cache) repo via `curl`, then compute). Cache hits are structurally validated (Σw = 2, Σx·w = 0, antisymmetric nodes); corrupt or wrong-precision files are skipped with a stderr warning. Public audit API: `verify_gl_cache_dir`.
- `root_finding` — f64 bisection with configurable tolerance and max iterations. Endpoint-zero handled correctly. No panics on degenerate inputs (zero bracket width, max_iter=0).
- `primes` — Sieve of Eratosthenes, prime counting function π(x).
- `linalg` (HP-gated) — Dense LU factorization with partial pivoting, LU solve, banded tridiagonal LU (Thomas with partial pivoting; O(n) factor and solve), inverse iteration (with optional forced-even projection and warm-start from a nearby cached eigenvector), ℓ² normalization, Rayleigh quotient. All inner reductions parallelized. The parallel reductions in `lu_solve`, `normalize_l2`, and `rayleigh_quotient` use a parallel map followed by a fixed index-order sequential fold (not rayon `.reduce()`), so HP results are **bit-identical run-to-run** despite the non-associativity of HP addition — required so the Weil eigenvector ξ is reproducible and cacheable.
- `fmt` (HP-gated) — `display_hp` (decimal scientific notation at any sig-digit count, no f64 underflow), `sign_of` (HP sign without f64), `matching_digits` and `relative_difference` (HP comparison helpers).
- `eigen` (HP-gated) — HP symmetric eigendecomposition: `tridiag_eigenvalues_hp` (symmetric tridiagonal QR with implicit Wilkinson shifts), `tridiag_eigenvector_for_value_hp` (shifted inverse iteration with `TridiagEigvecOptions { max_steps, early_termination, solver: Banded | Dense }`), `dense_symmetric_eigenvector_for_value_hp`, `householder_tridiag_hp` (dense → tridiagonal reduction), `dense_symmetric_eigenvalues_hp` (full pipeline). Truly dynamic in working precision; verified at HP-1000 against PARI/GP 2000-digit reference for both dense and tridiagonal cases (≥500 digits across 9 reference matrices including Hilbert and Wilkinson W11).

**xc-zeta:**
- `zeros` — Load reference zeros as HP strings, f64, or `rug::Float`; path-parameterized. All three loaders have tests including the `first_n_hp` HP precision path.

**xc-spectral:**
- `ccm` — CCM construction: `CcmParams` (`from_lambda_sq_integer` / `from_lambda_sq_fractional`), `LambdaSq` (integer/fractional λ² mode), `CcmResult`, `prime_powers_up_to`, `run_f64`, `solve_spectrum_f64`.
- `ccm::hp` (HP-gated) — `HighPrecConfig`, `HighPrecResult`, `run`, `measure_evenness`, full Weil-form matrix assembly at arbitrary precision. The τ-matrix is cached to `<cwd>/data/tau_cache/` (JSON, single zip, or byte-split `.partXX` for GitHub's 100 MB limit). The Weil eigenvector ξ is cached to `<cwd>/data/weil_eigvec_cache/` with a remote-fetch tier from the public [`xcelerator-weil-eigvec-cache`](https://github.com/TeamXcelerator/xcelerator-weil-eigvec-cache) repo. The eigensolver uses adaptive Newton steps, f64 warm seeds, Halley's method (selectable via `XCELERATOR_SOLVER=halley`), warm-start inverse iteration (`XCELERATOR_WARM_START=1`), and auto-detects even/odd eigenvector symmetry. Newton convergence failures return `None` (type: `Vec<Option<Float>>`) rather than silently returning a bad result. Public audit API: `verify_tau_cache_dir`. Also exposes `weil_spectrum_hp(params, cfg, include_primes)` — the full dense HP spectrum of the localized Weil-form τ-matrix (smallest positive eigenvalue = the plunge `ε_N`), with an archimedean-only mode (`include_primes = false` drops the prime-power sum `w_p` from every τ entry) for prefactor-decomposition studies. `weil_plunge_cancellation_hp(params, cfg) -> PlungeCancellation` finds the full plunge eigenvector ξ and returns the plunge `ε_N` together with the archimedean and prime Rayleigh quotients on that same ξ (`ε_N = arch_rayleigh − prime_rayleigh` by linearity), quantifying the archimedean↔prime cancellation that sets the floor. `weil_spectrum_sonin_hp(params, cfg, omega, n_drop)` (with `SoninRestriction`) and `band_concentration_matrix_hp(params, cfg, omega)` add the time-frequency (Slepian/prolate) band-concentration operator in the same `V_n` basis (eigenvalues `χ∈(0,1)`): the latter exposes the band-concentrated vs Sonin-like (anti-band) split, and the former deflates the top `n_drop` band-concentrated modes to return the archimedean Weil spectrum on the Sonin subspace (where archimedean positivity holds, Connes Thm 7.1).
- `screw` (HP-gated) — Suzuki screw function `g(t)` (arXiv:2606.09096): `ScrewKernel::new(a_max, prec)` / `.eval(&t)`, HP throughout (closed-form `ψ(1/4) = −γ − π/2 − 3 ln 2` and `Φ(1,2,1/4) = π² + 8·G` from rug constants; Hurwitz–Lerch `Φ(z,2,1/4)` by HP series; von Mangoldt sum via `prime_powers_up_to`). Validated to 1e-38 against an independent reference implementation. The continuous kernel of the localized Weil-form convolution operator `G_a`.
- `prolate` — Prolate-wave operator PW_λ. f64 prototype and HP submodule `prolate::hp` using the HP eigensolver from `xc-numerics::eigen`. Eigenvalue spectrum cached to `<cwd>/data/prolate_eigvals_cache/`. Public audit API: `verify_prolate_eigvals_cache_dir`.
- `mellin` — Truncated completed eta function Λ_λ(s), ξ-weighted Mellin G(s), parallelized critical-line zero scanner. Full f64 (`*_f64`) and HP (`*_hp`) parity. Uses `xc_numerics::quadrature::gl_nodes_weights_f64` for the f64 path (no internal duplication).
- `yakaboylu` — Yakaboylu's Hilbert-Pólya framework. f64 prototype and HP submodule using `dense_symmetric_eigenvalues_hp`. HP matrix build is parallelized.
- `lfunction` — Dirichlet L-function character specs (χ₃, χ₄, χ₅, χ₇), twisted prime-power enumeration. `chi_at` and `chi_at_prime_power` return exact `i8` values (precision-agnostic) alongside the `_f64` variants.

## Tests

All magic numbers are extracted to documented public constants. All
public APIs have unit tests covering both normal operation and boundary
conditions.

```bash
# f64-only (Windows/Linux/macOS — no system dependencies):
cargo test --workspace
# 56 tests pass

# Full HP tier (Linux/WSL/macOS — requires libgmp-dev libmpfr-dev libmpc-dev):
cargo test --workspace --features hp --release -- --test-threads=1
# ~213 tests pass, ~23 ignored (HP compute, slow Mellin scans, live-network);
# test-threads=1 required on WSL2 (GMP arena limit)
#
# To run everything including the heavy HP compute tests:
#   RAYON_NUM_THREADS=2 cargo test --features hp --release -- --test-threads=1 --include-ignored
```

### HP eigensolver verification (3 layers)

The HP symmetric eigensolver in `xc-numerics::eigen` is verified at three
independent levels:

1. **Closed-form structured matrices** (Strang's tridiagonal, Hilbert,
   rotated diagonal, clustered eigenvalues, Wilkinson W21) — closed-form
   eigenvalues at HP-256 and HP-1000.
2. **PARI/GP cross-check** — `tests/eigen_reference.rs` loads
   `tests/fixtures/eigen_reference.json` (9 reference matrices generated
   by PARI at 2000-digit precision) and verifies every eigenvalue matches
   our HP-1000 result to ≥500 decimal digits.
3. **Property-based** — random symmetric matrices (deterministically
   seeded HP LCG, no f64) verified to satisfy trace, determinant,
   eigenequation, normalization, orthogonality, and decomposition
   properties.

To regenerate the PARI fixture (only needed if the test cases change):

```bash
sudo apt install pari-gp  # if not already installed
cd crates/xc-numerics/tests/fixtures
gp -q generate_eigen_reference.gp > eigen_reference.json
```

## Performance

The HP code paths are parallelized with [rayon](https://github.com/rayon-rs/rayon)
throughout.

| Layer | Parallelized hot spots |
|---|---|
| `xc-numerics::eigen` | Householder reduction: ‖x‖ and ‖v‖² reductions, matvec, symmetric rank-2 update, Q accumulation. |
| `xc-numerics::linalg` | LU Schur-complement update; `lu_solve` inner reductions (length-thresholded); `normalize_l2`, `rayleigh_quotient`, inverse iteration initial guess and forced-even projection. |
| `xc-spectral::ccm::hp` | Symmetrize loops (parallel compute, sequential write), Newton-per-seed loop, evenness reduction. |
| `xc-spectral::yakaboylu::hp` | `build_w_matrix` outer-row loop. |
| `xc-spectral::prolate::hp` | u-grid evaluation, ξ-value reconstruction and dot reductions. |
| `xc-spectral::mellin` | Critical-line scan grid evaluation (both f64 and HP). |

The toolkit ships HP-everywhere by policy. f64 fast paths exist where
explicitly requested (suffixed `_f64`); they remain useful for
quick-iteration smoke tests but cannot reach the precisions needed for
publication-grade convergence claims.

## Usage

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
explicitly requested**. f64 underflows below ~10⁻³⁰⁸, which is
routinely violated by HP values our papers produce (e.g. eigenvalues
of magnitude 10⁻¹⁰⁰⁰ at λ²=1000).

### Naming convention

Every public function that operates at f64 precision has `_f64` in its
name. There is no ambiguity at the call site: if you don't see `_f64`,
the function is HP (or precision-agnostic).

| Pattern | Examples |
|---|---|
| `_f64` suffix → f64-only | `gauss_legendre_64pt_f64`, `gl_nodes_weights_f64`, `bisect_f64`, `omega_f64`, `truncated_lambda_f64`, `xi_weighted_mellin_f64`, `scan_critical_line_zeros_f64`, `solve_spectrum_f64`, `chi_at_f64`, `chi_at_prime_power_f64`, `build_w_matrix_f64`, `smallest_eigenvalue_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`, `run_f64`, `first_n_f64` |
| `_hp` suffix → HP | `omega_hp`, `truncated_lambda_hp`, `xi_weighted_mellin_hp`, `scan_critical_line_zeros_hp`, `tridiag_eigenvalues_hp`, `householder_tridiag_hp`, `dense_symmetric_eigenvalues_hp`, `tridiag_eigenvector_for_value_hp` |
| no suffix → HP-default | `ccm::hp::run`, `inverse_iteration`, `lu_factor`, `lu_solve`, `normalize_l2`, `rayleigh_quotient`, `display_hp`, `matching_digits`, `chi_at`, `gauss_legendre_nodes` |


## Version History

- **v0.11.2** — WSL2 HP reliability fix. Adds `xc-numerics::hp_runtime`:
  a WSL2-aware execution context that detects `/proc/version` and, on WSL2 only,
  routes all HP entry points through a single shared rayon pool (2 workers,
  256 MB per-worker stack) and a 256 MB `spawn_scoped` outer thread. Fixes the
  hard-abort (exit 1, no panic) that occurred on WSL2 when 32 default rayon
  workers exhausted the GMP arena/mmap limit during dense LU and GL-node
  Newton compute. All five public HP entry points in `ccm::hp` and
  `scan_critical_line_zeros_hp` in `mellin` are wrapped with `run_hp(||…)`.
  On Vast / native Linux the wrapper is a zero-overhead passthrough — full
  parallelism and behavior unchanged. Also marks slow/intensive HP compute
  tests `#[ignore]` with explicit run instructions (`RAYON_NUM_THREADS=2
  cargo test --features hp --release -- --include-ignored --test-threads=1`)
  so the default test run (`cargo test --features hp`) completes cleanly in
  ~60 s on WSL2 without process crashes. **Backwards compatible — no public
  API changed; all callers of `run`, `weil_spectrum_hp`, etc. are unaffected.**

- **v0.11.1** — New `ccm::hp::band_concentration_matrix_hp(params, cfg, omega)`:
  the time-frequency (Slepian/prolate) band-concentration operator in the
  localized Weil-form `V_n` basis, with eigenvalues `χ ∈ (0,1)` separating the
  band-concentrated (prolate) subspace from the Sonin-like (anti-band) subspace.
  New `ccm::hp::weil_spectrum_sonin_hp(params, cfg, omega, n_drop)` (with the
  `SoninRestriction` result): deflates the top `n_drop` band-concentrated modes
  and returns the archimedean Weil spectrum on the Sonin subspace (where
  archimedean positivity holds, Connes Thm 7.1). Adds the `weil_sonin` example.
  **Backwards compatible — additive only** (no public API changed). Adds 2 unit
  tests. Also scrubs internal shorthand from a few code comments (no behavior
  change).

- **v0.11.0** — New `xc-spectral::screw` module: the Suzuki screw function
  `g(t)` (arXiv:2606.09096) in HP, validated to 1e-38 against an independent
  reference. New `ccm::hp::weil_spectrum_hp(params, cfg, include_primes)`:
  the full HP spectrum of the localized Weil-form τ-matrix, with an
  archimedean-only mode (`include_primes = false`) that drops the prime-power
  sum for prefactor-decomposition studies. Also adds
  `ccm::hp::weil_plunge_cancellation_hp` (with the `PlungeCancellation` result)
  — decomposes the plunge `ε_N` on the full eigenvector ξ into archimedean and
  prime Rayleigh quotients (`ε_N = arch − prime`), and the `weil_cancellation`
  example. **Backwards compatible — additive
  only** (no public API changed; the one internal signature change is to a
  private function whose sole caller is unchanged). Adds 4 unit tests.

- **v0.10.0** — Eigensolver overhaul (adaptive Newton steps, f64 warm seeds,
  Halley's method, warm-start inverse iteration, auto-detect even/odd symmetry,
  Newton cross-over detection). Newton non-convergence now returns `None`
  (breaking API change: `eigenvalues_pos: Vec<Option<Float>>`).
  `LambdaSq` struct: explicit integer/fractional λ² mode throughout all
  cache layers. Primary constructors: `CcmParams::from_lambda_sq_integer(13,
  n_modes)` and `from_lambda_sq_fractional(12.5, n_modes)` — pass λ² directly.
  `from_lambda` and `LAMBDA_SQ_ROUNDING_EPS` removed. Cache JSON envelopes:
  `schema_version`, `toolkit_version`, `min_compatible_version`,
  `lambda_sq_mode` on all 4 cache types (GL, τ, Weil eigvec, prolate).
  `save_xi_json`, `load_xi_json`, `LoadedXi` removed (dead API).
  226 tests, 0 failures.

- **v0.9.2** — τ-cache remote fetch checks two repos (`xcelerator-tau-cache`
  then `xcelerator-tau-cache-2`) to support cache overflow.

- **v0.9.1** — Zip-only cache: τ, GL, and Weil caches read directly from
  `.json.zip` in memory; no decompressed `.json` written to disk, halving
  local cache storage overhead.

- **v0.9.0** — DynamicFetch cache tier (GL, τ, and Weil caches fetch from
  public repos on demand). Weil eigenvector (ξ) cache with bit-reproducible
  determinism. `force_even` field on `HighPrecConfig`. Cache keys on
  `force_even` (forced/natural ξ stored separately).

- **v0.8.0** — Parallel `lu_solve` (inner triangular-solve reductions via
  rayon). Eliminates the last single-core bottleneck in inverse iteration
  at large N.

- **v0.7.0** — GL cache writes `.json.zip` alongside `.json` (distribution
  artifact for git checkin).

- **v0.6.0** — Cache infrastructure (GL, τ, prolate), heavy testing audit,
  API consolidation (`tridiag_eigenvector_for_value_hp` single entry point),
  rustdoc pass.

- **v0.5.0** — Tridiagonal LU + banded shifted inverse iteration. O(n)
  per-step solve, memory-efficient eigenvector recovery for HP-scale
  tridiagonal systems.

- **v0.4.0** — HP-everywhere (no silent f64 leaks), comprehensive rayon
  parallelization, HP symmetric eigendecomposition (Householder + QR +
  inverse iteration), `xc-numerics::fmt` module.

- **v0.3.0** — `ccm::hp::measure_evenness()` for eigenvector symmetry
  measurement.

- **v0.2.0** — Breaking: `ccm::hp::run()` takes `&[Float]` seeds
  (eliminates f64 truncation in Newton seeding).

- **v0.1.0** — Initial release. CCM construction, prolate, Mellin,
  Yakaboylu, L-functions, HP numerics.

## Used by

- [`ccm-reproduction-and-convergence`](https://github.com/TeamXcelerator/ccm-reproduction-and-convergence) — Independent reproduction of the CCM zeta spectral triple, with eigenvalue match to 999 digits.
- [`ccm-convergence-rate-falsifications`](https://github.com/TeamXcelerator/ccm-convergence-rate-falsifications) — Empirical convergence-rate study and falsification of proposed convergence-rate predictions.

## License

Source-available for academic verification, study, and citation.
See [LICENSE](LICENSE) for terms.
