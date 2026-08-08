# Standalone public-API consumer

Version target: `0.13.4`

`tests/external-consumer` is intentionally excluded from the toolkit workspace and has its own manifest and dependency lock. It models an adjacent mathematical application rather than a toolkit crate. Its direct toolkit dependencies are `xc-core`, `xc-operator`, `xc-solver`, and `xc-certify` (plus supporting public cache/numerics types); it does not depend on `xc-spectral` and its source contains no CCM implementation or import.

The default workflow constructs a positive diagonal operator through `xc-operator`, solves its algebraic minimum through the public `EigenSolverF64` contract, creates a finite positive-definiteness bundle, and independently invokes `verify_bundle`. The HP feature additionally constructs an exact rational interval matrix through public numerics types, builds a portable interval-inertia certificate, and invokes the standalone verifier.

Run it independently from the repository root:

```powershell
cargo test --manifest-path tests/external-consumer/Cargo.toml --locked
```

On the supported GNU/Linux or WSL HP tier:

```bash
cargo test --manifest-path tests/external-consumer/Cargo.toml --all-features --release --locked
```

`check_external_consumer.py` rejects a missing independent lock, missing public solver/certification calls, or any `xc-spectral` dependency/domain import. Both release drivers format, test, and run strict Clippy on the fixture separately from the workspace, including its HP exact-certificate path on the HP tier.
