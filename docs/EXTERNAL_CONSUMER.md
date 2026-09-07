# Standalone public-API consumer

Version target: `0.15.0`

`tests/external-consumer` is intentionally excluded from the toolkit workspace and has its own manifest and dependency lock. It models an adjacent mathematical application rather than a toolkit crate. Its direct toolkit dependencies are `xc-core`, `xc-operator`, `xc-solver`, and `xc-certify` (plus supporting public cache/numerics types); it does not depend on `xc-spectral` and its source contains no CCM implementation or import.

The consumer also exercises automatic capture accounting through the public
cache API, retaining a missing outcome alongside a measured value.

The default workflow constructs a positive diagonal operator through `xc-operator`, solves its algebraic minimum through the public `EigenSolverF64` contract, creates a finite positive-definiteness bundle, and independently invokes `verify_bundle`. The HP feature additionally constructs an exact rational interval matrix through public numerics types, builds a portable interval-inertia certificate, and invokes the standalone verifier.

Run it independently from the repository root:

```powershell
cargo test --manifest-path tests/external-consumer/Cargo.toml --locked
```

On the supported GNU/Linux or WSL HP tier:

```bash
cargo test --manifest-path tests/external-consumer/Cargo.toml --all-features --release --locked
```

Run strict Clippy separately because this fixture is outside the workspace:

```sh
cargo clippy --manifest-path tests/external-consumer/Cargo.toml --all-targets --locked -- -D warnings
cargo clippy --manifest-path tests/external-consumer/Cargo.toml --all-targets --features hp --locked -- -D warnings
```

The [manual qualification workflow](../.github/workflows/ccm-qualification.yml)
includes the HP consumer test. The [release validation summary](VALIDATION.md)
records the native and HP consumer results.
