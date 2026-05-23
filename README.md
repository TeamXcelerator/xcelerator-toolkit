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
| [`xc-numerics`](crates/xc-numerics) | High-precision numerical primitives: GL quadrature (f64 + HP with disk cache), LU factorization, inverse iteration, root-finding, prime sieve. |
| [`xc-zeta`](crates/xc-zeta) | Riemann zeta function utilities: reference zero loading (HP strings, f64, rug::Float), path-parameterized. |
| [`xc-spectral`](crates/xc-spectral) | Spectral methods: CCM Weil-form construction (f64 + HP), prolate-wave operators, Mellin transforms, Yakaboylu W-positivity framework, Dirichlet L-function extensions. |

### Module inventory

**xc-numerics:**
- `quadrature` — Gauss-Legendre at f64 (configurable N-point) and HP (with disk cache)
- `root_finding` — f64 bisection with configurable tolerance and max iterations
- `primes` — Sieve of Eratosthenes, prime counting function π(x)
- `linalg` (HP-gated) — LU factorization with partial pivoting, LU solve, inverse iteration (with optional forced-even projection), ℓ² normalization, Rayleigh quotient

**xc-zeta:**
- `zeros` — Load reference zeros as HP strings, f64, or `rug::Float`; path-parameterized for flexibility

**xc-spectral:**
- `ccm` — CCM construction: `CcmParams`, `CcmResult`, `prime_powers_up_to`, `run_f64`, `solve_spectrum`
- `ccm::hp` (HP-gated) — `HighPrecConfig`, `HighPrecResult`, `run`, `save_xi_json`, `load_xi_json`, full Weil-form matrix assembly at arbitrary precision
- `prolate` — Prolate-wave operator PW_λ, eigenfunction identification (h₀, h₄), ℰ map, comparison against ξ_λ
- `mellin` — Truncated completed eta function Λ_λ(s), ξ-weighted Mellin G(s), parallelized critical-line zero scanner, HP variants
- `yakaboylu` — Yakaboylu's Hilbert-Pólya framework: V̂_R matrix elements, W-positivity tests, synthetic off-line zero detection
- `lfunction` — Dirichlet L-function character specs (χ₃, χ₄, χ₅, χ₇), twisted prime-power enumeration

## Tests

All magic numbers are extracted to documented public constants. All
public APIs have unit tests.

```bash
# f64-only (Windows/Linux/macOS — no system dependencies):
cargo test --workspace
# 47 tests pass, 0 ignored

# Full HP tier (Linux/WSL/macOS — requires libgmp-dev libmpfr-dev libmpc-dev):
cargo test --workspace --features hp
# 56 tests pass, 0 ignored
```

## Using from another crate

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
