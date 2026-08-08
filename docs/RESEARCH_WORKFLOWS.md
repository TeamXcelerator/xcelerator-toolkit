# Research workflow entry points

Version target: `0.13.4`

These examples are stable, nonmutating entry points for the three release-required research workflows. Run them from the repository root. Successful commands write machine-readable JSON to standard output and return status 0; failures write Rust error context to standard error and return nonzero. None reads credentials, publishes cache data, or mutates a remote repository.

## CCM observation planning

The high-precision CCM API uses the managed cache without consumer setup. A full tau hit is loaded directly. On a miss, the toolkit resolves or computes compact archimedean integrals and the prime-component matrix, assembles and validates the full tau matrix, and records the dependency chain. `CcmParityPolicy::EvenSector` remains the default and reuses the established reduced even-sector operator and LU identities. `Natural` performs an unrestricted full-space solve. `AdaptiveEven` performs the original full-space inverse iteration and conditionally projects only an iterate that materially drifts away from even symmetry. The three policies have disjoint selected-state and downstream artifact identities, while sharing compatible Tau and full-factorization dependencies. Normal runs retain the selected ground eigenpair, secular source, one bundled and one-based indexed root range, and configuration evidence; evenness runs retain both natural and even-sector states plus their validation record. Author-mode publication stages the same objects to their configured public/private family shards without requiring the consuming application to implement a publishing pipeline.

Parity-sector research is an explicit operation because dense low-spectrum recovery is unnecessary overhead for an ordinary reproduction. Calling `xc_spectral::ccm::hp::analyze_sector_gap` derives and caches both parity matrices from the same exact Tau dependency, retains the requested lowest even and odd eigenpairs as `ccm_sector_spectrum` artifacts, and retains the derived `ccm_sector_gap` evidence. The odd basis matches the historical `(e_k-e_-k)/sqrt(2)` convention. The result reports GapLog, the direct eigenvalue difference and ordering, and an even-sector simplicity margin as separate values.

`xc_spectral::ccm::hp::run` is reference-free by default. It discovers a positive prefix from the finite secular source and then performs HP refinement; it does not accept a zero table. `run_independent` accepts prefix, one-based index-range, and height-window targets. Computed assurance uses pole-aware MPFR discovery directly on the full-precision secular source to obtain starting points and therefore makes no rigorous completeness claim. Certified production discovery uses exact cumulative finite-source counts, FLINT/Arb complete root isolation, and interval Newton through `certify_production_independent_ccm_roots`. `build_source` explicitly requests a source-only computation.

Halley's method is the ordinary HP refinement route; Newton remains an explicit comparison option and is never an automatic fallback. Root refinement and CCM inverse iteration each default to a 2,000-iteration ceiling. A root is converged only after meeting the requested-accuracy correction target; an additional 64 MPFR guard bits absorb secular-sum cancellation without changing that target. An unchanged MPFR point, a two-cycle, or 128 consecutive iterations without a smaller correction is classified as stagnation. Slow monotone improvement may use the full iteration budget. Computed runs retain finite, ordered stagnated and iteration-limited approximate values with their correction, residual, iteration count, and achieved digits, print their status, and continue the requested window; they are never relabeled as converged. A failed root with no value remains fatal, and cross-checked or certified assurance requires every root to converge. Inverse iteration separately records its configured limit, unshifted steps, convergence flag, final Rayleigh change, shifted-refinement outcome, and replayed relative Tau residual. Reaching the unshifted limit remains visible in ordinary output and run evidence even when shifted refinement successfully rescues the eigenstate. The CCM eigenstate must pass its Tau residual check before root refinement begins.

`run_indexed_seeded` and `run_indexed_seeded_via_cache` are explicit comparison/refinement APIs. Their supplied values are reference seeds, their artifacts are `ccm_root_refinement`, and they cannot satisfy an independent-discovery request. Reference datasets may be attached only through a separate post-discovery comparison artifact after the independent root window has been frozen. The lowest Tau eigenpair remains the standard CCM secular source; higher Tau eigenpairs must not be treated as later zeta zeros.

Independent and seeded refinement have the same final HP refinement and normally converge to the same finite-source roots when the computed discovery finds the complete requested sequence. Refinement is faster because it skips discovery and begins close to each root, but it is circular evidence and cannot establish that CCM found the roots. Computed independent discovery adds a pole-aware MPFR scan whose cost remains small relative to Tau construction. Certified independent discovery is substantially more expensive because it builds the exact finite numerator and proves count, completeness, existence, and uniqueness. Certification remains selective rather than the default assurance.

```powershell
cargo run -p xc-spectral --example ccm_window_plan --locked
```

The result is a serialized `ObservationPlan` for the first 100 zeros. It reports the estimated height, minimum mode reach, recommended precision, guard digits, and assumptions. This is a feasibility observation, not a proof of continuum convergence.

To replay a complete finite f64 observation previously created by `run_saved_ccm_f64_observation`, supply the saved record and a freshly captured execution fingerprint:

```powershell
cargo run -p xc-spectral --example ccm_reproduce --locked -- saved-observation.json current-fingerprint.json
```

Replay reconstructs the Weil matrix, smallest eigenpair, normalized even source, and rational spectrum from the saved configuration. It refuses an execution-fingerprint mismatch and compares the timing-independent numerical payload exactly. The saved answer is comparison evidence only and is never supplied to the computation. Success reproduces a finite binary64 observation, not an HP certificate or an infinite-dimensional conclusion.

## Finite certificate construction and verification

```powershell
cargo run -p xc-certify --example finite_certificate --locked
```

The example constructs a finite-dimensional positive-definiteness certificate, recomputes its canonical certificate identity, independently runs the bundle verifier, and emits the verified `CertificateBundle`. The JSON separates its finite claim, achieved assurance, backend, precision, matrix identity, inertia evidence, assumptions, and provenance.

## Exact Maynard–Tao lower bound

This route needs the `hp` feature and therefore a supported GNU/Linux toolchain with GMP/MPFR, such as the project's existing HP WSL environment:

```bash
cargo run -p xc-variational --example mk_constant --features hp --locked
```

The result is a serialized `MkRayleighCertificate` for the constant degree-zero candidate at `k = 2`. Its exact numerator, denominator, and quotient establish a rigorous lower bound within the declared finite polynomial search space. For the larger symmetric discovery-plus-exact-evaluation example, run:

```bash
cargo run -p xc-variational --example mk_symmetric --features hp --locked
```

The exploratory eigensolver in `mk_symmetric` proposes coefficients, but the reported lower bound is recomputed after exact rationalization and does not inherit assurance from floating-point discovery.

## Library-level normal use

Every production workspace crate has a compiled normal-use target recorded in `EXAMPLE_INVENTORY.json`. The smaller library examples can be run independently:

```powershell
cargo run -p xc-numerics --example quadrature --locked
cargo run -p xc-operator --example matrix_action --locked
cargo run -p xc-root --example bracketed_root --locked
cargo run -p xc-solver --example plan --locked
cargo run -p xc-core --example publication_export --locked
cargo run -p xc-zeta --example reference_zeros --locked
```

The cache examples separately demonstrate overlay resolution, permission failure, publication planning, and verification without executing remote mutation. `xc` is a binary-first crate, so its compiled normal-use target is the binary rather than an artificial example wrapper.

## Validation commands

The bounded default examples are exercised directly with:

```powershell
cargo run -p xc-spectral --example ccm_window_plan --locked
cargo run -p xc-certify --example finite_certificate --locked
```

The HP examples are compiled and run on a supported GNU/Linux HP environment. Windows MSVC is intentionally not presented as an HP route because `gmp-mpfr-sys` does not support that target.
