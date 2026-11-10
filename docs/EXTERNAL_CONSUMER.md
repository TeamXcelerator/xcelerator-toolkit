# Standalone public-API consumer

Version target: `0.17.1`.

`tests/external-consumer` is intentionally excluded from the toolkit workspace and has its own manifest and dependency lock. It models an adjacent mathematical application rather than a toolkit crate. Its direct toolkit dependencies include `xc-core`, `xc-operator`, `xc-solver`, `xc-certify`, `xc-cache`, optional HP `xc-numerics`, and `xc-spectral`. The solver example is domain independent; a separate public CCM capture-plan check verifies the current Ultra contract. It imports public CCM types without implementing CCM internals.

The consumer also exercises automatic capture accounting through the public
cache API, retaining a missing outcome alongside a measured value.

The default workflow constructs a positive diagonal operator through `xc-operator`, solves its algebraic minimum through the public `EigenSolverF64` contract, creates a finite positive-definiteness bundle, and independently invokes `verify_bundle`. The HP feature additionally constructs an exact rational interval matrix through public numerics types, builds a portable interval-inertia certificate, and invokes the standalone verifier.

Run it independently from the repository root:

```powershell
cargo test --manifest-path tests/external-consumer/Cargo.toml --locked
```

On GNU/Linux, or in the [configured native Windows environment](WINDOWS.md),
enable high precision with the same shared native arithmetic libraries:

```bash
cargo test --manifest-path tests/external-consumer/Cargo.toml --features hp,xc-numerics/arb --release --locked
```

Run strict Clippy separately because this fixture is outside the workspace:

```sh
cargo clippy --manifest-path tests/external-consumer/Cargo.toml --all-targets --locked -- -D warnings
cargo clippy --manifest-path tests/external-consumer/Cargo.toml --all-targets --features hp,xc-numerics/arb --locked -- -D warnings
```

Release checks include the native and HP consumer tests.
