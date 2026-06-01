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
| [`xc-numerics`](crates/xc-numerics) | High-precision numerical primitives: GL quadrature (f64 + HP with `<cwd>/data/gl_cache/` disk cache, JSON or zip-compressed), LU factorization, inverse iteration, root-finding, prime sieve, HP symmetric eigendecomposition, HP formatting / comparison helpers. |
| [`xc-zeta`](crates/xc-zeta) | Riemann zeta function utilities: reference zero loading (HP strings, f64, rug::Float), path-parameterized. |
| [`xc-spectral`](crates/xc-spectral) | Spectral methods: CCM Weil-form construction (f64 + HP), prolate-wave operators (f64 + HP), Mellin transforms (f64 + HP), Yakaboylu W-positivity framework (f64 + HP), Dirichlet L-function extensions. |

### Module inventory

**xc-numerics:**
- `quadrature` — Gauss-Legendre at f64 (configurable N-point) and HP. The HP path caches nodes/weights to `<cwd>/data/gl_cache/` and supports both uncompressed JSON and zip-compressed JSON fixtures (auto-decompressed on first read). A [`CacheMode`] parameter on `gauss_legendre_nodes(n, prec, mode)` selects the lookup strategy: `Off` (always compute), `JsonOnly` (local `.json` only), `JsonZip` (local `.json` then `.json.zip`), or `DynamicFetch` (default — local, then a remote download of the specific fixture from the public [`xcelerator-gl-cache`](https://github.com/TeamXcelerator/xcelerator-gl-cache) repo via `curl`, then compute). Remote fetch fires only on local cache miss and falls through to compute if `curl`/network/the fixture is unavailable. Per-cwd layout means each paper repo / reproduction script gets its own independent cache, and pre-computed cache fixtures can be checked into a repo to skip the cold-start cost of Newton iteration. Cache hits are structurally validated (Σw = 2, Σx·w = 0, antisymmetric nodes); corrupt or wrong-precision files are skipped with a stderr warning. Public audit API: `verify_gl_cache_dir`.
- `root_finding` — f64 bisection with configurable tolerance and max iterations. Endpoint-zero handled correctly (returns the zero endpoint, no walk-away).
- `primes` — Sieve of Eratosthenes, prime counting function π(x).
- `linalg` (HP-gated) — Dense LU factorization with partial pivoting, LU solve, banded tridiagonal LU (Thomas with partial pivoting; O(n) factor and solve), inverse iteration (with optional forced-even projection; rustdoc documents both convergence floors), ℓ² normalization, Rayleigh quotient. Inner reductions and matvec parallelized. `lu_solve` parallelizes its inner triangular-solve reductions by default; `lu_solve_with(..., parallel)` exposes a serial/parallel toggle for tiny matrices or deterministic single-threaded benchmarking. The parallel reductions in `lu_solve`, `normalize_l2`, and `rayleigh_quotient` use a parallel map followed by a fixed index-order sequential fold (not rayon `.reduce()`), so HP results are **bit-identical run-to-run** despite the non-associativity of HP addition — required so the Weil eigenvector ξ is reproducible and cacheable.
- `fmt` (HP-gated) — `display_hp` (decimal scientific notation at any sig-digit count, no f64 underflow), `sign_of` (HP sign without f64), `matching_digits` and `relative_difference` (HP comparison helpers). Use these wherever you'd otherwise call `to_f64()` for display or comparison.
- `eigen` (HP-gated) — HP symmetric eigendecomposition: `tridiag_eigenvalues_hp` (symmetric tridiagonal QR with implicit Wilkinson shifts; allocation-optimized inner loop), `tridiag_eigenvector_for_value_hp` (shifted inverse iteration with `TridiagEigvecOptions { max_steps, early_termination, solver: Banded | Dense }`), `dense_symmetric_eigenvector_for_value_hp` (shifted inverse iteration on dense input), `householder_tridiag_hp` (dense → tridiagonal reduction with parallel reductions, matvec, symmetric update, and Q accumulation), `dense_symmetric_eigenvalues_hp` (full pipeline). Truly dynamic in working precision (verified at HP-1000 against PARI/GP 2000-digit reference for both dense and tridiagonal cases; matches to ≥500 digits across 9 reference matrices including Hilbert and Wilkinson W11).

**xc-zeta:**
- `zeros` — Load reference zeros as HP strings, f64, or `rug::Float`; path-parameterized for flexibility.

**xc-spectral:**
- `ccm` — CCM construction: `CcmParams`, `CcmResult`, `prime_powers_up_to`, `run_f64`, `solve_spectrum_f64`.
- `ccm::hp` (HP-gated) — `HighPrecConfig`, `HighPrecResult`, `run`, `save_xi_json`, `load_xi_json`, `measure_evenness`, full Weil-form matrix assembly at arbitrary precision. The τ-matrix construction is cached automatically to `<cwd>/data/tau_cache/` (uncompressed JSON, single zip, or byte-split `.partXX` for files exceeding GitHub's 100 MB limit). The smallest-eigenvalue Weil eigenvector ξ is cached to `<cwd>/data/weil_eigvec_cache/` (uncompressed JSON or single zip — ξ is small, ≲2 MB, so no byte-split tier), governed by the same `CacheMode` as the GL/τ caches with a remote-fetch tier from the public [`xcelerator-weil-eigvec-cache`](https://github.com/TeamXcelerator/xcelerator-weil-eigvec-cache) repo. The ξ cache check sits *after* the τ build so a cached `(ξ, ε_N)` is validated against the in-hand τ via the eigen-residual `‖τξ − ε_N·ξ‖`; a hit skips the dominant `O(N³)` LU factorization. ξ is bit-reproducible run-to-run because the inverse-iteration reductions (`lu_solve`, `normalize_l2`, `rayleigh_quotient`) use a fixed index-order fold rather than a runtime-ordered parallel reduction. Symmetrize loop, Newton-per-seed loop, and evenness reduction are all parallelized. Public audit API: `verify_tau_cache_dir`.
- `prolate` — Prolate-wave operator PW_λ. f64 prototype (`build_pw_matrix_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`) and HP submodule `prolate::hp` (`build_pw_matrix`, `compute_k_lambda`, `compare_xi_to_k_lambda`) using the HP eigensolver from `xc-numerics::eigen`. The eigenvalue spectrum from `tridiag_eigenvalues_hp` (the dominant cost in `compute_k_lambda` at HP-1000) is cached to `<cwd>/data/prolate_eigvals_cache/`. HP u-grid evaluation parallelized. Public audit API: `verify_prolate_eigvals_cache_dir`.
- `mellin` — Truncated completed eta function `Λ_λ(s)`, ξ-weighted Mellin `G(s)`, parallelized critical-line zero scanner. Full f64 (`*_f64`) and HP (`*_hp`) parity: `omega_f64` / `omega_hp`, `truncated_lambda_f64` / `truncated_lambda_hp`, `xi_weighted_mellin_f64` / `xi_weighted_mellin_hp`, `scan_critical_line_zeros_f64` / `scan_critical_line_zeros_hp`. The HP scan also runs in parallel.
- `yakaboylu` — Yakaboylu's Hilbert-Pólya framework. f64 prototype (`v_r_matrix_element_f64`, `build_w_matrix_f64`, `test_w_positivity_f64`, `WPositivityResultF64`) and HP submodule `yakaboylu::hp` (`build_w_matrix`, `test_w_positivity`, `HpWPositivityResult`) using `dense_symmetric_eigenvalues_hp`. HP outer-row matrix build is parallelized.
- `lfunction` — Dirichlet L-function character specs (χ₃, χ₄, χ₅, χ₇), twisted prime-power enumeration. `chi_at` and `chi_at_prime_power` return exact `i8` values (precision-agnostic) alongside the `_f64` variants.

## Tests

All magic numbers are extracted to documented public constants. All
public APIs have unit tests.

```bash
# f64-only (Windows/Linux/macOS — no system dependencies):
cargo test --workspace
# 54 tests pass, 0 ignored

# Full HP tier (Linux/WSL/macOS — requires libgmp-dev libmpfr-dev libmpc-dev):
cargo test --workspace --features hp
# 171 tests pass, 2 ignored (PARI-fixture heavy tests);
# plus 1 ignored live-network test (remote_fetch_live) — run it with:
#   cargo test -p xc-numerics --features hp -- --ignored remote_fetch_live
```

### HP eigensolver verification (3 layers)

The HP symmetric eigensolver in `xc-numerics::eigen` is verified at three
independent levels:

1. **Closed-form structured matrices** (Strang's tridiagonal, Hilbert,
   rotated diagonal, clustered eigenvalues, Wilkinson W21) — closed-form
   eigenvalues at HP-256 and HP-1000.
2. **PARI/GP cross-check** — `tests/eigen_reference.rs` loads
   `tests/fixtures/eigen_reference.json` (9 reference matrices generated
   by PARI at 2000-digit precision via `polrootsreal(charpoly(M))`) and
   verifies every eigenvalue matches our HP-1000 result to ≥500 decimal
   digits.
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

The committed `eigen_reference.json` is 448 KB; the test passes vacuously
on machines without PARI but verifies fully wherever the JSON is present.

## Performance

The HP code paths are parallelized with [rayon](https://github.com/rayon-rs/rayon)
throughout. Most parallelization is unconditional — there are no `if n >
threshold` guards — with one exception: `lu_solve` runs its inner
triangular-solve reduction serially for short rows (below a small fixed
threshold) where rayon's dispatch overhead would exceed the work, and in
parallel for longer rows. Small-n tests pay a small constant overhead, but
production workloads scale across all available cores.

| Layer | Parallelized hot spots |
|---|---|
| `xc-numerics::eigen` | Householder reduction: ‖x‖ and ‖v‖² reductions, matvec `p = β·A·v`, symmetric rank-2 update `A ← A − v·qᵀ − q·vᵀ`, vᵀp reduction, Q accumulation `Q ← Q · H`. |
| `xc-numerics::linalg` | `lu_factor` Schur-complement update; `lu_solve` inner forward/back-substitution reductions (per-row, length-thresholded); `normalize_l2` (parallel sum-of-squares + per-element divide), `rayleigh_quotient` (parallel row evaluation + final reduction), `inverse_iteration` initial guess and forced-even projection. |
| `xc-spectral::ccm::hp` | `run` and `measure_evenness` symmetrize loops (parallel pair compute, sequential write to avoid mirror-cell aliasing), Newton-per-seed loop in `run` (~50 independent refinements), combined `diff_sq + norm_sq` reduction in `measure_evenness`. |
| `xc-spectral::yakaboylu::hp` | `build_w_matrix` outer-row loop. |
| `xc-spectral::prolate::hp` | `compute_k_lambda` u-grid evaluation, `compare_xi_to_k_lambda` ξ-value reconstruction and dot reductions. |
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
explicitly requested**. f64 underflows below ~10⁻³⁰⁸, which is
routinely violated by HP values our papers produce (e.g. eigenvalues
of magnitude 10⁻¹⁰⁰⁰ at λ²=1000).

### Naming convention

Every public function that operates at f64 precision has `_f64` in its
name. There is no ambiguity at the call site: if you don't see `_f64`,
the function is HP (or precision-agnostic).

| Pattern | Examples |
|---|---|
| `_f64` suffix → f64-only | `gauss_legendre_64pt_f64`, `bisect_f64`, `omega_f64`, `truncated_lambda_f64`, `xi_weighted_mellin_f64`, `scan_critical_line_zeros_f64`, `solve_spectrum_f64`, `chi_at_f64`, `chi_at_prime_power_f64`, `build_w_matrix_f64`, `smallest_eigenvalue_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`, `run_f64`, `build_tau_f64`, `v_r_matrix_element_f64`, `build_pw_matrix_f64`, `first_n_f64`, `to_f64_result`, `bisect_zero_f64`, `legendre_p_deriv_f64`, `gl_nodes_weights_f64`, `parity_of_f64`, `count_nodes_f64`, `interp_grid_f64`, `build_pw_dense_f64` |
| `_hp` suffix → HP | `omega_hp`, `truncated_lambda_hp`, `xi_weighted_mellin_hp`, `scan_critical_line_zeros_hp`, `tridiag_eigenvalues_hp`, `householder_tridiag_hp`, `dense_symmetric_eigenvalues_hp`, `dense_symmetric_eigenvector_for_value_hp`, `tridiag_eigenvector_for_value_hp` |
| no suffix → HP-default | `ccm::hp::run`, `ccm::hp::measure_evenness`, `inverse_iteration`, `lu_factor`, `lu_solve`, `normalize_l2`, `rayleigh_quotient`, `display_hp`, `sign_of`, `matching_digits`, `relative_difference`, `chi_at` (returns exact `i8`), `chi_at_prime_power` (returns exact `i8`), `gauss_legendre_nodes` (in the `hp` submodule) |

### Allowed f64 in HP code

f64 is permitted in HP-claiming code only at:

- **Documented f64 boundary fields** in result structs (e.g. `CcmResult`,
  `LoadedXi.xi_f64`, `LoadedXi.weil_min_eigenvalue` — paired with `xi_hp`
  and `weil_min_eigenvalue_hp` for HP consumers).
- **`to_f64_result()`** — explicit lossy conversion.
- **Wall-clock metadata** (`elapsed_seconds: f64`).
- **CLI input parameters** (e.g. `CcmParams::from_lambda(lambda: f64, ...)`).
  These are user inputs at f64 precision by contract; the precomputed
  `lambda_sq_int: u64` field is what HP code paths actually consume.
- **Digit-to-bit conversion** (`bits = digits * DIGITS_TO_BITS_FACTOR`)
  where the output is an integer (`u32`) — no precision loss.

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

| Version | Changes |
|---|---|
| `v0.9.0` | **GL and τ caches gain a `CacheMode` and remote-fetch tier.** `gauss_legendre_nodes` takes a `CacheMode` argument (signature change), and `HighPrecConfig` gains a `cache_mode` field (default `DynamicFetch`) that governs both the GL-node and τ-matrix caches: `Off` / `JsonOnly` / `JsonZip` / `DynamicFetch`. DynamicFetch adds a last-resort tier that downloads the specific fixture from the public consolidated cache repos ([`xcelerator-gl-cache`](https://github.com/TeamXcelerator/xcelerator-gl-cache), [`xcelerator-tau-cache`](https://github.com/TeamXcelerator/xcelerator-tau-cache)) via `curl`, before falling back to a fresh compute. |
| | • **GL lookup order (DynamicFetch):** local `.json` → local `.json.zip` → remote `.json.zip` → compute. |
| | • **τ lookup order (DynamicFetch):** local `.json` → local single `.json.zip` → local multi-part `.json.zip.partXX` → **remote** (probe single zip first, then probe `.part00`, `.part01`, … until a part 404s; concatenate + decompress) → compute. |
| | • **Remote URLs are deterministic** from the cache keys using each repo's bucketed layout. GL: `gl_cache/prec{P}/npts{B}-{B+999}/...`. τ: `tau_cache/prec{P}/lambda_sq{L}/nmodes{B}-{B+999}/...` with `B = (N/1000)*1000`. |
| | • **`CacheMode::Off`** computes always and never reads/writes disk or network. **`JsonZip`** reproduces the exact pre-v0.9.0 behavior. Downloads go to a temp path and rename on success so a failed download never leaves a truncated file. |
| | • Call sites updated: `mellin` (×2) and `ccm::hp` τ-build + GL precompute pass the configured mode. |
| | • **New tests:** GL `cache_mode_off_never_touches_disk`, `cache_mode_json_only_ignores_zip`, `remote_url_uses_bucketed_layout`; τ `tau_remote_url_uses_bucketed_layout`; plus two `#[ignore]`-gated live end-to-end fetch tests (`remote_fetch_live_downloads_and_validates` for GL, `tau_remote_fetch_live_downloads_and_validates` for τ's byte-split path) — run with `cargo test --features hp -- --ignored`. The pre-existing GL/τ cache tests pin `JsonZip`/`Off` as appropriate. |
| | • **Motivation:** lets fresh cloud (Vast) runs pull only the specific GL and τ fixtures they need from the public cache repos on demand — no giant clone, no recompute. See `xcelerator-research/research/ccm/PERFORMANCE.md`. |
| | • **`HighPrecConfig` gains a `force_even: bool` field** (default `true`). When `true` (the default), inverse iteration projects onto the even subspace at each step — the standard CCM path. When `false`, the natural (unprojected) smallest eigenvector is used. This allows testing whether the smallest eigenvector is naturally even without forcing, supporting the empirical conjecture that forced-even projection is unnecessary above the precision floor. |
| | • **Doctest fix:** the `verify_gl_cache_dir` and `tridiag_eigenvector_for_value_hp` doc examples use ` ```text ``` ` fences (they reference undefined locals and must not compile under `--include-ignored`). |
| | • **Weil eigenvector (ξ) cache + determinism.** The parallel reductions in `lu_solve`/`normalize_l2`/`rayleigh_quotient` switched from rayon `.reduce()` (runtime-ordered) to a parallel-map + fixed index-order sequential fold, making ξ **bit-identical run-to-run** (HP addition is non-associative, so reduction order matters). On that foundation, `ccm::hp::run` gains a `weil_eigvec_cache` to `<cwd>/data/weil_eigvec_cache/` keyed on `(λ², N, prec)`, governed by the same `CacheMode`: `weil_eigvec_lambda_sq{L}_nmodes{N}_prec{P}.json[.zip]` locally, remote-fetched from the public [`xcelerator-weil-eigvec-cache`](https://github.com/TeamXcelerator/xcelerator-weil-eigvec-cache) repo (single zip, no `.partXX` — ξ is ≲2 MB). |
| | • **ξ lookup order (DynamicFetch):** local `.json` → local `.json.zip` → remote `.json.zip` → compute. The check runs **after** the τ build so the cached `(ξ, ε_N)` is validated against τ via the eigen-residual `‖τξ − ε_N·ξ‖` below the working-precision floor; a hit skips the dominant `O(N³)` LU factorization. Schema mirrors `save_xi_json` (`schema_version: 1`: ξ strings + `weil_min_eigenvalue` + metadata). |
| | • **Self-heal:** a structurally-invalid or residual-failing cached ξ is skipped (warned), recomputed, and overwritten in place — matching the GL/τ caches. |
| | • **New tests:** ξ `weil_eigvec_remote_url_uses_bucketed_layout`, `weil_eigvec_parse_json_validates`, `weil_eigvec_residual_check_discriminates`, `weil_eigvec_save_load_round_trip`, plus negative tests `weil_eigvec_load_skips_structurally_invalid_json` and `weil_eigvec_load_handles_corrupt_zip_gracefully`, and an `#[ignore]`-gated live fetch. Also **backfilled τ load-path negative tests** (`tau_load_skips_structurally_invalid_json`, `tau_load_handles_corrupt_zip_gracefully`) to bring τ to parity with GL. cwd-mutating tests serialized via a `CwdGuard` mutex (mirrors the GL test module). |
| | • **New example** `gen_weil_eigvec_fixture` (HP-gated): runs the real pipeline to mint a genuine ξ cache fixture (`cargo run --release --features hp --example gen_weil_eigvec_fixture -- <lambda_sq> <n_modes> <digits>`). |
| | • Test counts: f64 16 + 37 + 1 = 54 pass on Windows MSVC. HP: full suite green; 2 `#[ignore]`-gated live-network tests (GL + τ) pass when run with `--ignored`. |
| `v0.8.0` | **`lu_solve` parallelizes its inner triangular-solve reductions (behavior change).** The forward/back-substitution inner sums are now split across rayon for rows longer than a small fixed threshold; the result is identical to working precision but the computation is multi-threaded and the HP reduction order is no longer deterministic. This is the per-step hot path in `inverse_iteration`, where it was the one remaining serial bottleneck at large dimension — every other HP hot path was already parallel, so on a many-core box the entire inverse-iteration phase previously ran on a single core. |
| | • **New `lu_solve_with(factors, b, dim, prec, parallel: bool)`.** `lu_solve` now delegates to it with `parallel = true` (the default for all callers, including `inverse_iteration`). Pass `parallel = false` for tiny matrices, deterministic single-threaded benchmarking, or callers already saturating cores at a higher level. |
| | • **Outer row loops remain sequential** — forward substitution row `i` depends on `y[0..i]` and back substitution on `x[i+1..]`; only the inner reduction `Σ_j lu[i,j]·{y,x}[j]` is parallelized. Short rows (below `PAR_SOLVE_MIN_ROW = 32`) stay serial to avoid rayon dispatch overhead exceeding the work. |
| | • **New test `lu_solve_serial_parallel_equivalence`** (HP-gated): on an n=80 Strang system (rows on both sides of the threshold) the serial and parallel paths, and the `lu_solve` default vs explicit `lu_solve_with(parallel=true)`, agree to 1e-60 at HP-256. |
| | • **Motivation:** cutting per-config wall-clock for the convergence-knob (Paper C) Vast runs, where large-`N` configs at high precision spend most of their time in `inverse_iteration`'s `lu_solve`. See `xcelerator-research/research/ccm/KNOB_SCALING_FORMULA.md`. |
| | • Test counts: f64 16 + 37 + 1 = 54 pass on Windows MSVC (unchanged; the new test is HP-gated). HP path: 168 pass, 2 ignored (was 167 + new equivalence test). |
| `v0.7.0` | **GL cache also writes `.json.zip` alongside `.json`.** Mirrors the τ-cache "always write both" pattern. No public API change; no behavior change for existing readers (the `.json` is still the fast next-read path; the new `.zip` is for distribution / git checkin). |
| | • `save_gl_cache` now writes both `.json` (canonical fast-read) and `.json.zip` (distribution artifact) on every fresh compute. Mirrors `tau_cache::save`. |
| | • Lets paper repositories check in compressed GL cache fixtures alongside the τ-cache fixtures using a single uniform pattern (`data/*_cache/*.json.zip`). The decompressed `.json` files are gitignored on the consumer side; fresh clones get full cache benefit on first read by auto-decompressing the committed `.zip`. |
| | • New test `cache_fresh_compute_writes_json_and_zip` (HP-gated) verifies both files appear on fresh compute, that the zip is smaller than the json, and that round-trip through the zip-only path returns bit-identical nodes/weights. |
| | • Test counts: f64 16 + 37 + 1 = 54 pass on Windows MSVC (unchanged; the new test is HP-gated). HP path: 167 expected on Linux (was 166; +1 new test). |
| `v0.6.0` | **Cache infrastructure, HEAVY testing, perf, API consolidation, rustdoc pass.** Squashed release of v0.5.1 through v0.5.13. Public API breaking vs v0.5.0 (eigenvector recovery consolidates three entry points into one). |
| | • **API consolidation:** `tridiag_eigenvector_for_value_hp` is the single entry point. Takes `TridiagEigvecOptions { max_steps, early_termination, solver: TridiagSolver::{Banded, Dense} }`. Banded LU is the default; dense is retained for cross-validation. The v0.4.x/v0.5.0 wrappers are removed. |
| | • **HEAVY testing audit:** three-layer validation (closed-form + cross-check + property) on every core numeric — dense LU, banded LU, tridiag QR (PARI/GP cross-check at 2000 digits, JSON fixture committed), Householder, inverse iteration, root finding (`bisect_f64` endpoint-sign bug fixed), GL quadrature, vector ops, HP cache fixtures. |
| | • **Cache infrastructure:** three disk caches share the same shape (structural validation on load, “preserve but discard” on corruption, public `verify_*_cache_dir` audit API): `<cwd>/data/gl_cache/` (Gauss-Legendre nodes/weights), `<cwd>/data/prolate_eigvals_cache/` (`compute_k_lambda` eigenvalues), `<cwd>/data/tau_cache/` (τ-matrix; custom byte-split `.json.zip.partXX` for compressed payloads exceeding GitHub’s 100 MB hard limit). |
| | • **Cache wall-time impact:** Paper A and Paper B re-runs at the same `(λ², N, prec)`: first run unchanged (cache miss → compute → save); subsequent runs skip τ-matrix construction and the prolate tridiag QR entirely. λ²=100 N=4001 HP-1000 prolate eigenvalues drop from ~27 minutes to ~5 seconds on a hit. |
| | • **Performance:** `tridiag_eigenvalues_hp` allocation reduction — 20 scratch Floats hoisted out of the QR sweep, ~14 fewer MPFR allocs per Givens step (~10¹⁰ fewer at HP-1000 N=8001). `CwdGuard` mutex poison recovery so a single panicking test doesn’t cascade. |
| | • **Documentation:** module-level rustdoc on every crate; field-level docs on every public struct, variant docs on every public enum, function docs on every public fn. |
| | • **Test counts:** f64 16 + 37 + 1 = 54 tests pass on Windows MSVC; HP 166 tests pass on Linux. |
| `v0.5.0` | **Tridiagonal LU + banded shifted inverse iteration.** Squashed release of v0.4.1 through v0.5.0. Memory-efficient eigenvector recovery for HP-scale tridiagonal systems. |
| | • **`xc-numerics::linalg::tridiag_lu_factor_hp` / `tridiag_lu_solve_hp`** — Thomas algorithm with partial pivoting at HP. O(n) factor and O(n) per-step solve vs the dense path’s O(n³)/O(n²). |
| | • **`eigen::tridiag_eigenvector_for_value_hp`** is now banded by default; the dense path is retained for cross-validation. Resident memory at HP-1000 N=8001 drops from ~26 GB (dense) to a few KB (banded). |
| | • **Opt-in early termination** on inverse iteration via the `|⟨v_k, v_{k-1}⟩|` convergence proxy (cheap O(n) per step). For well-conditioned, well-separated eigenvalues this typically cuts step count from 200 to 20-50. |
| | • **Progress visibility** — long-running HP iterations print elapsed-time progress every 25 steps so reviewers can distinguish “still iterating” from “wedged” on multi-hour runs. |
| | • **GL cache** moved from `~/.cache/ccm_gl/` to `<cwd>/data/gl_cache/` (per-cwd, parallel-run safe) and supports zip-compressed cache files. |
| `v0.4.0` | **HP-everywhere unless explicitly requested + comprehensive rayon parallelization.** |
| | • **Naming:** every public f64 function has `_f64` in its name; every HP function is unsuffixed (default) or `_hp`-suffixed where a parallel f64 version exists. No silent f64 leaks in HP code paths. |
| | • **`xc-numerics::eigen`** — new HP symmetric eigendecomposition (Householder + tridiagonal QR with Wilkinson shifts + shifted inverse iteration). Verified across 3 layers: closed-form structured matrices (Strang, Hilbert, Wilkinson W21), PARI/GP cross-check at 2000 digits (committed JSON fixture, 9 reference matrices), property-based tests with deterministic random matrices (trace, determinant, eigenequation, normalization, orthogonality, decomposition reconstruction). |
| | • **`xc-numerics::fmt`** — new module with HP-only formatting (`display_hp`) and comparison helpers (`sign_of`, `matching_digits`, `relative_difference`) that operate without an f64 round-trip. |
| | • **`xc-spectral::prolate::hp`** and **`yakaboylu::hp`** submodules promote both pipelines to pure HP using the new HP eigensolver. |
| | • **`xc-spectral::mellin`** gains full HP parity: `omega_hp`, `truncated_lambda_hp`, `xi_weighted_mellin_hp`, `scan_critical_line_zeros_hp` with parallel scan grid. |
| | • **Rayon parallelization** across the HP hot path (Householder, linalg primitives, ccm::hp symmetrize/Newton/evenness, yakaboylu W-matrix, prolate u-grid, mellin scan). Unconditional — no small-n guards. |
| | • **`prime_powers_up_to`** now returns `(k, p, j)` triples (was `(k, log_p_f64)`); HP code paths read `p` directly and compute `log p` in HP. |
| | • **`CcmParams`** gains a precomputed `lambda_sq_int: u64` field; HP code paths consume that integer instead of recomputing from f64. |
| | • **`LFunctionSpec::chi_at`** and **`chi_at_prime_power`** — new exact `i8`-returning variants alongside the `_f64` ones; usable in HP code without an f64 cast. |
| | • **Inverse-iteration seed vector** built entirely in HP. Internal `fl(prec, v: f64)` helpers replaced with integer literals and `Float::parse` for non-integer constants. |
| | • **`LoadedXi.weil_min_eigenvalue_hp`** field added (paired with the existing f64 view). |
| | • Test counts: 47 (f64-only) / 111 (full HP). |
| `v0.3.0` | Add `ccm::hp::measure_evenness()` for eigenvector symmetry measurement. 60 tests on Vast. |
| `v0.2.0` | **Breaking:** `ccm::hp::run()` now takes `&[Float]` seeds instead of `&[f64]`. Eliminates f64 truncation in Newton seeding that caused divergence at high eigenvalue index (k > ~100). |
| `v0.1.0` | Initial release. CCM construction, prolate, Mellin, Yakaboylu, L-functions, HP numerics. 58 tests pass. |

## Used by

- [`ccm-reproduction-and-convergence`](https://github.com/TeamXcelerator/ccm-reproduction-and-convergence) — Paper A: independent reproduction of CCM zeta spectral triple, with eigenvalue match to 999 digits at HP-1000 (λ²=1000, N=800) against a 2000-digit PARI/GP reference.
- [`ccm-convergence-rate-falsifications`](https://github.com/TeamXcelerator/ccm-convergence-rate-falsifications) — Paper B: empirical convergence-rate study at HP-1000 (CCM Lemma 7.2 falsified — rel×λ² grows by ~1.116×10⁶ across λ²∈[13,1000]; Śliwiński Conjecture 4.1 unsupported across κ∈[50,500]; CCM is 10⁹⁹⁹× more accurate than naive Mellin truncation at λ²=1000).

## License

Source-available for academic verification, study, and citation.
See [LICENSE](LICENSE) for terms.

Modification, redistribution, and commercial use require explicit
written permission. Contact: randrewsmath@gmail.com

## Trademarks

"Team Xcelerator Inc." is a registered trademark of Team Xcelerator Inc.
All other trademarks are the property of their respective owners.
