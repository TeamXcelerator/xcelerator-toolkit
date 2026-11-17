# Optional CPU execution tuning

Toolkit defaults remain portable. CPU instruction selection, residue storage,
worker counts and output tile sizes are execution choices; they do not change
the exact integer product, declared precision, assurance policy or cache key.
Enable and measure options on the machine that will run the workload.

## Runtime options

| Variable | Default | Optional values |
| --- | --- | --- |
| `XC_CPU_PRODUCT_KERNEL` | `portable` | `packed`, `avx2`, `avx512`, `auto` |
| `XC_RESIDUE_LAYOUT` | `nested` | `flat` |
| `RAYON_NUM_THREADS` | Rayon's default | Explicit worker count, within the run worker allowance |
| `XC_LU_TILE_EDGE` | `1024` | Positive maximum output tile edge, still subject to workspace admission |

`portable` retains the established CPU residue-product implementation. `packed`
centers residues into signed 16-bit buffers and transposes the right operand so
both dot-product inputs are contiguous. `avx2` and `avx512` use the same layout
with exact integer SIMD instructions. AVX-512 requires AVX-512F and AVX-512BW.
Explicit unsupported ISA choices return an error when a CPU product is selected.
`auto` dispatches at runtime to AVX-512, AVX2, or packed scalar. On non-x86-64
targets it selects packed scalar. The widest available ISA is not necessarily
the fastest choice for every shape or processor.

All SIMD accumulators widen adjacent signed products to 64 bits before summing.
The admitted bounds cover every partial sum; unaligned operands and scalar
tails preserve all terms. This is integer arithmetic, not approximate
floating-point vectorization. No fast-math mode is enabled.

`flat` stores each input's residue table in one contiguous allocation instead
of allocating a vector for every integer. It also applies when the selected
product backend is CUDA. It retains the established conservative memory
reservation; it does not relax resource limits or change the CRT output.
CPU kernel selection applies only to CPU residue products and does not replace
an explicitly selected CUDA backend.

The first use of each nondefault execution combination prints a bounded
`[CPU execution]` summary. CUDA stage timings remain available through
`XC_CUDA_TIMINGS=1`. These messages are operational evidence, not artifact data.

Applications can use `modular_product::multiply_with_execution` with
`CpuExecutionOptions` to select options explicitly without changing process
environment variables. `multiply` reads the environment settings above.
`CpuExecutionOptions::default()` selects portable kernels and nested storage.
The standalone API requires the `modular-product` feature; `hp` also enables it.
CPU-only applications can select `modular-product` without enabling CUDA or HP.

## Optional compilation for the local CPU

Rust can optimize rebuilt Rust code for the build machine:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release --workspace --all-features --locked
```

For a configured native Windows build, in a fresh PowerShell process:

```powershell
. .\tools\windows-env.ps1
$env:RUSTFLAGS = '-C target-cpu=native'
cargo build --release --workspace --all-features --locked
```

Follow the ordinary native-library setup first. Preserve any other required
Rust flags when adding this option. `rustc --print target-cpus` lists explicit
architecture names. A CPU-targeted binary may require instructions unavailable
on another machine; retain generic builds for portable distribution or use the
runtime-dispatched kernels above. These choices are not enabled by repository
defaults.

Toolkit's release profile already uses optimization level 3, Thin LTO and one
code-generation unit. A consuming application's root Cargo profile controls
its dependency builds; it does not inherit Toolkit's profile. Optional
`CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1` and `CARGO_PROFILE_RELEASE_LTO=fat`
overrides allow controlled comparisons without editing an application's public
defaults. Measure full LTO against Thin LTO rather than assuming an improvement.

Profile-guided optimization is another optional build workflow: instrument,
run representative workloads, merge profiles using compatible LLVM tools, then
rebuild the same source and options with the merged profile. Bind training data
to source, features, compiler and target flags. A profile from a different
application or obsolete source is not qualified for a production build. See
the [Rust PGO guide](https://doc.rust-lang.org/rustc/profile-guided-optimization.html).

Rust flags do not rebuild precompiled GMP, MPFR or FLINT libraries, and do not
retune CUDA kernels. Some dependencies already perform runtime CPU dispatch.

## Cache locality and measurement

Hardware cache is used automatically. Its topology matters: a shared cache may
serve only a subset of the cores, and an aggregate cache size is not a per-worker
budget. Applications may use their operating system's affinity facilities when
starting a workload. Toolkit does not alter process affinity by default.
Compare physical-core placement and SMT with the same total worker allowance;
coordinate any outer processes and avoid nested oversubscription.

Output tiling preserves complete exact inner sums. The fixed LU panel width is
part of the numerical arithmetic contract and is not a tuning control. Smaller
output tiles may improve locality but increase GPU transfers or reduce device
utilization. Keep CPU preparation, GPU products and CRT reconstruction timings
separate, and distinguish warm hardware caches from numerical artifact reuse.

Before adopting a setting, compare complete numerical outputs, CPU/GPU equality,
cold/warm artifact reuse, memory admission and repeated timing distributions.
Retain CPU/compiler choices in execution provenance when payloads are identical;
do not add hardware settings to numerical cache identities.
