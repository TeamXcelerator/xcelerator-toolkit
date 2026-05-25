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
- `quadrature` — Gauss-Legendre at f64 (configurable N-point) and HP. The HP path caches nodes/weights to `<cwd>/data/gl_cache/` and supports both uncompressed JSON and zip-compressed JSON fixtures (auto-decompressed on first read). Per-cwd layout means each paper repo / reproduction script gets its own independent cache, and pre-computed cache fixtures can be checked into a repo to skip the cold-start cost of Newton iteration.
- `root_finding` — f64 bisection with configurable tolerance and max iterations
- `primes` — Sieve of Eratosthenes, prime counting function π(x)
- `linalg` (HP-gated) — LU factorization with partial pivoting, LU solve, inverse iteration (with optional forced-even projection), ℓ² normalization, Rayleigh quotient. Inner reductions and matvec parallelized.
- `fmt` (HP-gated) — `display_hp` (decimal scientific notation at any sig-digit count, no f64 underflow), `sign_of` (HP sign without f64), `matching_digits` and `relative_difference` (HP comparison helpers). Use these wherever you'd otherwise call `to_f64()` for display or comparison.
- `eigen` (HP-gated) — HP symmetric eigendecomposition: `tridiag_eigenvalues_hp` (symmetric tridiagonal QR with implicit Wilkinson shifts), `tridiag_eigenvector_for_value_hp` and `dense_symmetric_eigenvector_for_value_hp` (shifted inverse iteration), `householder_tridiag_hp` (dense → tridiagonal reduction with parallel reductions, matvec, symmetric update, and Q accumulation), `dense_symmetric_eigenvalues_hp` (full pipeline). Truly dynamic in working precision (verified at HP-1000 against PARI/GP 2000-digit reference; matches to ≥500 digits across 9 reference matrices including Hilbert and Wilkinson W11).

**xc-zeta:**
- `zeros` — Load reference zeros as HP strings, f64, or `rug::Float`; path-parameterized for flexibility

**xc-spectral:**
- `ccm` — CCM construction: `CcmParams`, `CcmResult`, `prime_powers_up_to`, `run_f64`, `solve_spectrum_f64`
- `ccm::hp` (HP-gated) — `HighPrecConfig`, `HighPrecResult`, `run`, `save_xi_json`, `load_xi_json`, `measure_evenness`, full Weil-form matrix assembly at arbitrary precision. Symmetrize loop, Newton-per-seed loop, and evenness reduction are all parallelized.
- `prolate` — Prolate-wave operator PW_λ. f64 prototype (`build_pw_matrix_f64`, `compute_k_lambda_f64`, `compare_xi_to_k_lambda_f64`) and HP submodule `prolate::hp` (`build_pw_matrix`, `compute_k_lambda`, `compare_xi_to_k_lambda`) using the HP eigensolver from `xc-numerics::eigen`. HP u-grid evaluation parallelized.
- `mellin` — Truncated completed eta function `Λ_λ(s)`, ξ-weighted Mellin `G(s)`, parallelized critical-line zero scanner. Full f64 (`*_f64`) and HP (`*_hp`) parity: `omega_f64` / `omega_hp`, `truncated_lambda_f64` / `truncated_lambda_hp`, `xi_weighted_mellin_f64` / `xi_weighted_mellin_hp`, `scan_critical_line_zeros_f64` / `scan_critical_line_zeros_hp`. The HP scan also runs in parallel.
- `yakaboylu` — Yakaboylu's Hilbert-Pólya framework. f64 prototype (`v_r_matrix_element_f64`, `build_w_matrix_f64`, `test_w_positivity_f64`, `WPositivityResultF64`) and HP submodule `yakaboylu::hp` (`build_w_matrix`, `test_w_positivity`, `HpWPositivityResult`) using `dense_symmetric_eigenvalues_hp`. HP outer-row matrix build is parallelized.
- `lfunction` — Dirichlet L-function character specs (χ₃, χ₄, χ₅, χ₇), twisted prime-power enumeration. `chi_at` and `chi_at_prime_power` return exact `i8` values (precision-agnostic) alongside the `_f64` variants.

## Tests

All magic numbers are extracted to documented public constants. All
public APIs have unit tests.

```bash
# f64-only (Windows/Linux/macOS — no system dependencies):
cargo test --workspace
# 47 tests pass, 0 ignored

# Full HP tier (Linux/WSL/macOS — requires libgmp-dev libmpfr-dev libmpc-dev):
cargo test --workspace --features hp
# 111 tests pass, 0 ignored
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
throughout. Parallelization is unconditional — there are no `if n >
threshold` guards. Small-n tests pay a small constant overhead, but
production workloads scale across all available cores.

| Layer | Parallelized hot spots |
|---|---|
| `xc-numerics::eigen` | Householder reduction: ‖x‖ and ‖v‖² reductions, matvec `p = β·A·v`, symmetric rank-2 update `A ← A − v·qᵀ − q·vᵀ`, vᵀp reduction, Q accumulation `Q ← Q · H`. |
| `xc-numerics::linalg` | `normalize_l2` (parallel sum-of-squares + per-element divide), `rayleigh_quotient` (parallel row evaluation + final reduction), `inverse_iteration` initial guess and forced-even projection. |
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
| `v0.5.0` | **Tridiagonal LU + banded shifted inverse iteration.** Architectural change: HP eigenvector recovery on tridiagonal matrices no longer densifies. |
| | • **`xc-numerics::linalg::tridiag_lu_factor_hp`** — Thomas algorithm with partial pivoting at HP. Stores L (sub-diagonal multipliers) and U (main + super + super-super diagonals to capture pivot fill-in) as four short vectors of length ~n, plus a row permutation. O(n) factor cost vs dense LU's O(n³). |
| | • **`xc-numerics::linalg::tridiag_lu_solve_hp`** — Forward + back substitution against the banded factored form. O(n) solve cost vs dense O(n²). |
| | • **`xc-numerics::eigen::tridiag_eigenvector_for_value_hp_banded`** — drop-in alternative to `tridiag_eigenvector_for_value_hp_with_options` that uses the banded LU instead of densifying T - λI + εI. At HP-1000 with N=8001 the per-eigenvector wall-time drops from hours to seconds and resident memory from ~26 GB to a few KB. |
| | • **`xc-spectral::prolate::hp::compute_k_lambda`** opts into the banded variant. Numerical output is unchanged (banded LU produces an eigenvector that satisfies T·v = λv to working precision, same as dense LU); the change is purely architectural. |
| | • **Heavy testing** to mirror the HP eigensolver's three-layer validation: banded vs dense LU equivalence on Strang's tridiagonal n=10 (HP-256), Wilkinson W11 + shift (HP-512), property test on a deterministic-random asymmetric tridiagonal n=50, partial-pivoting test on a zero-diagonal first row, HP-1000 production residual check (n=20, ‖T·v - λv‖_∞ < 10⁻⁹⁰⁰). |
| | • The dense variant `tridiag_eigenvector_for_value_hp_with_options` remains in the public API for backward compatibility and cross-validation purposes. |
| `v0.4.3` | **Better progress timing + opt-in early termination on inverse iteration.** |
| | • **`xc-numerics::eigen::tridiag_eigenvector_for_value_hp_with_options`** — new explicit-options entry point with an `early_termination: bool` flag. The original `tridiag_eigenvector_for_value_hp` function is unchanged in behaviour (delegates to `_with_options(..., false)`); existing callers get bit-identical numerics. |
| | • **Convergence test** when `early_termination=true`: tracks `|⟨v_k, v_{k-1}⟩|` (cheap O(n) per step) and breaks the inverse-iteration loop as soon as the change drops below the working-precision threshold. For well-conditioned, well-separated eigenvalues this typically cuts step count from 200 to 20-50, a 4-10× speedup on the iteration phase. |
| | • **`xc-spectral::prolate::hp::compute_k_lambda`** opts into early termination — prolate eigenvalues are widely separated at small k, so the iteration converges quickly. The published numerical output of `prolate-compare` is unchanged (the iteration still runs to convergence; it just stops as soon as it gets there). |
| | • **Improved progress timing.** The previous v0.4.2 prints only timed the inverse-iteration loop, missing the dense-matrix build (~6.6 GB at N=4001, ~26 GB at N=8001) and the LU factor (the actual O(N³) cost). v0.4.3 wraps the entire `tridiag_eigenvector_for_value_hp` body in a phase timer and prints `[HP eigvec] dense matrix built in Xs` and `[HP eigvec] LU factor done in Xs` separately. Each per-step progress line now also reports both the iter-only and total-phase elapsed times. |
| | • Backward compatible: numerical output identical to v0.4.2 in all configurations; new HP cache tests still pass; the opt-in flag means callers that don't pass it get the conservative full-`max_steps` behaviour. |
| `v0.4.2` | **Progress visibility for long-running HP iterations.** |
| | • **`xc-numerics::eigen::tridiag_eigenvector_for_value_hp`** — adds an `eprintln!` every 25 inverse-iteration steps reporting `(step, max_steps, N, elapsed_seconds)`. Lets users distinguish "still iterating" from "wedged" on multi-hour runs at large N. No behavior change to the numerical output; pure observability. |
| | • **`xc-numerics::linalg::inverse_iteration`** — same per-iteration progress line every 25 steps; also prints a final line on convergence. |
| | • **`xc-spectral::prolate::hp::compute_k_lambda`** — bracket prints around each phase: tridiagonal build, full eigenvalue compute (tridiag QR), eigenvector search loop with per-iteration line, k_λ sampling. Each phase reports its own elapsed time. |
| | • **`xc-spectral::prolate::hp::compare_xi_to_k_lambda`** — bracket prints with elapsed time. |
| | • Triggered by 2026-05-25 Paper B Claim 1 retest cycle: a `prolate-compare` run sat silent for ~12 hours under multi-process contention with no visible signal whether it was making progress. The new prints would have made the wedge state visible within minutes. |
| | • Backward compatible: numerical output identical to v0.4.1; tests unchanged. New `eprintln!` lines are stderr only (don't disturb stdout result parsing). |
| `v0.4.1` | **GL cache: per-cwd layout + zip-compressed cache support.** |
| | • **`xc-numerics::quadrature`** — HP GL cache directory moved from `~/.cache/ccm_gl/` to `<cwd>/data/gl_cache/`. Per-cwd makes parallel runs across multiple servers and concurrent processes safer (no shared mutable state in `$HOME`), and lets paper repositories ship pre-computed cache fixtures alongside their reproduction scripts. |
| | • **Zip-compressed cache files** — the toolkit now also reads `prec{prec}_npts{n}.json.zip` (zip archive containing a single entry of the same name without `.zip`). Lookup priority: uncompressed `.json` first, then `.json.zip` (auto-decompressed on first read; the decompressed copy is also written next to the `.zip` so future reads hit the fast path), then compute fresh. |
| | • Compresses HP-1000 GL caches by ~3-4× in practice; lets paper repositories check in pre-warmed cache fixtures without bloating `git clone`. |
| | • Backward compatible: existing callers of `gauss_legendre_nodes(n, prec)` get the same return type and the same caching semantics, just from a different directory. |
| | • New `zip` (v2.2, deflate-only) workspace dependency, gated by the `hp` feature. |
| | • New unit tests in `quadrature::hp_cache_tests`: lookup priority, zip fallback with auto-decompress, fresh-compute round-trip, integration sanity. All gated `#[cfg(all(test, feature = "hp"))]`. |
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

- [`ccm-reproduction-and-convergence`](https://github.com/TeamXcelerator/ccm-reproduction-and-convergence) — Paper A: independent reproduction of CCM zeta spectral triple at 460 matching digits.
- [`ccm-convergence-rate-falsifications`](https://github.com/TeamXcelerator/ccm-convergence-rate-falsifications) — Paper B: empirical falsification of CCM Lemma 7.2 and Śliwiński Conjecture 4.1.

## License

Source-available for academic verification, study, and citation.
See [LICENSE](LICENSE) for terms.

Modification, redistribution, and commercial use require explicit
written permission. Contact: randrewsmath@gmail.com

## Trademarks

"Team Xcelerator Inc." is a registered trademark of Team Xcelerator Inc.
All other trademarks are the property of their respective owners.
