# High-precision matrix products and quadrature

HP CCM primary, evenness, and bordered response solves use blocked LU with
exact Schur products by default for every factorization dimension greater than
one, including small claims.
CUDA is discovered at runtime; CPU execution implements the same arithmetic.
`xc-numerics` also exposes the underlying exact integer and stored-MPFR product
APIs for direct use.

## Default CCM LU and cache compatibility

Managed and standalone primary inverse iteration, Krylov polishing, standalone
evenness solves, and their shifted-refinement factorizations use
`linalg::lu_factor_auto` through `inverse_iteration_auto_detailed`. The bordered
solves for prime-power response and u-flow response also use `lu_factor_auto`.
Panel width is the deterministic rule
`min(512, max(1, dimension / 4))`. Partial pivoting and panel arithmetic use MPFR;
each trailing update rounds `C - exact(L * U)` once to the working precision.
Neither tiling nor the selected backend changes that rule. The scalar case
retains the original route. The explicit low-level `linalg::lu_factor` and
legacy inverse-iteration APIs remain available with their original arithmetic.

The numerical identity is
`dense_lu_partial_pivot_dimension_panel512_exact_schur_v1`. Factorization and
eigenpair keys bind it before lookup, including `Require`-policy eigenpairs.
Standalone filenames, payload validation, and nearby-precision warm-start
discovery also bind it. These primary artifacts require reader 0.17.0; they
do not reuse historical unstamped LU/eigenpair artifacts. Descendants follow
their source identities. No cache reset is required. Unchanged quadrature and
matrix artifacts remain reusable under their existing identities.

Both response envelopes independently bind the same arithmetic at their
bordered dimension `N + 2` before lookup and require reader 0.17.0. This prevents
an old response from being reused even when its retained primary state happens
to have identical numerical bytes. Their source and residual checks remain in
place.

Exact products retain arbitrary-precision input bits and cancellation, but
the new rounding sequence can change pivots, eigenvectors, diagnostics, and
serialized bytes relative to the old sequential LU. CPU, one-GPU, and multi-GPU
execution share the new identity. Backend, device count, and output tile size
are execution controls, not numerical-identity parameters. Residual,
ground-index, and root acceptance checks still apply.

GPU work is limited to admitted exact modular products in these LU updates.
Panel operations, conversions, CRT reconstruction, and other pipeline
stages still require CPU work. This does not move quadrature, inertia,
certification, or every research diagnostic onto GPUs. Neither device discovery
nor a successful kernel test demonstrates a complete-claim speedup.

## Runtime selection and controls

The `hp` feature includes `cuda-product`; CUDA is not a build prerequisite.
By default, `Auto` selects admitted compatible devices for each product.
It may select CPU before dispatch when CUDA is unavailable, a product does not
meet the configured threshold, device workspace cannot be admitted, or the
LU product exceeds the finite modular reconstruction domain. Invalid
configuration and explicit device-selection errors refuse. Once GPU work is
selected, allocation, transfer, self-test, kernel, cleanup, and device-restoration
errors propagate; a failed GPU calculation is not silently retried on CPU.
Forced `cuda` refuses when its requested execution is unavailable or unsupported.

The HP modular adapter handles products wider than the single-pass residue family using signed-magnitude
binary limbs. Each limb-pair product fits the same proved CRT domain and uses
the selected backend; complete integer products are shifted and summed exactly
before the sole final Float conversion. This also applies to exact-output LU
updates, preserving the single-rounded Schur subtraction and cached arithmetic
identities. Forced CUDA continues to require accelerator execution and propagate
device errors. Automatic LU retains its existing exact CPU route for these
wide products, including previously admitted tight workspace budgets.
The low-level `modular_product::multiply` API and its report
continue to describe one finite-family product.

Wide-product admission charges the retained full-width accumulator, limb
operands, and extraction/recombination scratch, then gives each sequential child
the remaining workspace. The aggregate operation bound includes every limb pair.
A later child's allocation refusal can repeat completed child work when an LU
output tile is split; the LU matrix is updated only after every child succeeds.
No partial wide product or reduced-precision substitute is accepted.

LU splits output tiles when either exact preparation or the subsequent
integer-bridge/modular planner exceeds the temporary workspace allowance.
Only typed pre-execution workspace refusals trigger splitting, which retains
each complete panel inner sum and its rounding rule. If even a one-element
output tile cannot fit, automatic LU may release the failed factorization and
restart with a larger reservation. This can repeat earlier successful products
within that factorization. Growth is bounded by the original available run
memory and the automatic ceiling; it never changes the configured run budget.
An explicit `XC_LU_WORKSPACE_BYTES` remains a hard limit. Numerical, device
execution, and shared-reservation failures propagate without this retry.

Public CCM primary and natural-evenness entry points inherit an existing
`RunResources` owner or establish one from `XC_RESEARCH_WORKING_BYTES` and
the current numerical pool. The evenness owner spans source construction and
both the natural and forced-even solves, including the explicit-cache route.
Nested work shares the caller's live reservations and worker limit; the
previous scope is restored on return or unwinding. Automatic LU therefore
uses the remaining configured run capacity on these routes instead of the
1 GiB availability fallback for unscoped low-level calls. The automatic
4 GiB ceiling and explicit workspace override rules are unchanged. A run can
now refuse when its actual shared reservations or pool exceed the declared
limits; admission does not change a successful result's arithmetic or identity.

Each admitted device owns its cuBLAS handle and buffers on its host worker.
Product stages reuse the current Rayon pool when its thread count matches the
requested worker allowance. A different explicit allowance uses a bounded
dedicated pool. Reused worker stacks remain caller-owned; the workspace model
retains a conservative stack allowance rather than measuring those stacks.
One distinct prime-residue product is reserved for each device before the
remaining primes enter a dynamic shared queue. A slower device initialization
therefore still receives work, and faster devices can process more of the queue.
The products are independent; peer access and NVLink are not required. Complete
CRT reconstruction retains every prime result in canonical order, yielding the
same exact integer output on CPU, one GPU, or multiple GPUs.

The selected device count is bounded by the worker allowance, number of primes,
remaining aggregate workspace, and each device's free memory after headroom.
`XC_CUDA_DEVICES=0,1` restricts consideration to those visible ordinals; both must
still pass admission to run together. Successful results expose
`cuda_device_work`, whose `successful_moduli` counts are positive for every
admitted device and sum to the product's modulus count. Compact completion
summaries accumulate these counts; verbose multi-device logs report each
product's counts, providing evidence of actual work on each device.
Successful return also requires checked cleanup and restoration of the host
worker's previous CUDA device.

Set controls before the first call; loaded CUDA libraries are cached for the
process lifetime.

Both CUDA engines zero-pad the reduction dimension to a multiple of 32.
Some Ampere/cuBLAS kernels reject rectangular products with only 16-element
reduction alignment even when the API's four-byte alignment rules are met.
Workspace admission and allocation use the same padded shape. Added terms are
exact zeros, so integer reconstruction, rounding, and artifact identities stay
unchanged. Device/runtime errors still propagate without changing backends.

| Variable | Behavior |
|---|---|
| `XC_DENSE_BACKEND` | Automatic CCM LU backend, including evenness and bordered responses: `auto` (default), `cpu`, or `cuda`. |
| `XC_LU_WORKSPACE_BYTES` | Positive hard temporary-workspace allowance. Without an override, the initial allowance scales with dimension, precision, and workers and may grow on typed admission refusals, capped by the original available run memory and 4 GiB (1 GiB available without a run scope). An explicit override remains subject to the run budget and never grows automatically. |
| `XC_LU_TILE_EDGE` | Positive output tile edge, default 1024; does not alter panel width or arithmetic. |
| `XC_CUDA_DEVICES` | Comma-separated unique visible device ordinals, for example `0,1`; ordinals are relative to `CUDA_VISIBLE_DEVICES`. Omission considers all compatible visible devices. |
| `XC_CUDA_MIN_SCALAR_PRODUCTS` | Nonnegative Auto threshold for `rows * inner * columns * number_of_primes`; default 0. Forced CUDA bypasses this threshold. |
| `XC_CUDA_LOG` | `summary` (default) logs the first CUDA admission and rate-limited cumulative completions. `verbose` retains per-product admission and multi-device residue completion lines. Other values warn once and use `summary`; this setting cannot turn a logging error into a numerical refusal. |
| `XC_CUDA_TIMINGS` | Set to `1` to log each completed exact product's shape, actual backend, selected GPU count, modulus count, bound and reconstruction bits, and planning/conversion/products/reconstruction seconds. |
| `XC_CUDA_PRODUCT_PIPELINE` | Unset prefers the two-lane pipeline at 128 or more residue primes per admitted device, subject to the checks below. `0` forces synchronous execution; `1` requires the pipeline. Other values refuse when CUDA selection is considered. |
| `XC_CUDA_HEADROOM_MIB` | Device free-memory reserve, default 512 MiB; at least 10% of total device memory is always reserved. |
| `XC_CUDA_LIB_DIR` | Platform-separated list of installed CUDA shared-library directories searched before `CUDA_PATH` and the platform loader. |
| `RAYON_NUM_THREADS` | CPU pool size; coordinate with the shared run worker allowance and outer process limits. |

`modular_product::cuda_status()` reports discovered driver/runtime/cuBLAS
versions, compatible visible device ordinals, physical PCI bus IDs, free memory,
and unavailability reasons. PCI bus IDs identify the hardware behind ordinals
that can be remapped by `CUDA_VISIBLE_DEVICES`. Driver version is not a substitute
for the installed runtime version. Product results record the actual backend,
devices, per-device successful modulus counts, and stage timings. Explicit
product APIs use their `ProductBackend` argument; `XC_DENSE_BACKEND` controls
`lu_factor_auto`.

Compact logging retains the first admitted product's shape, devices, library
versions, execution mode, and workspace even if execution subsequently fails.
It emits a summary after the first successful exact product and at most once
every 30 seconds afterward, on a successful product completion. These are
process-wide cumulative counts of exact products that passed reconstruction
and range checks, synchronous and pipelined products, and actual per-device
residue work. Device counters retain the first 64 distinct ordinals; additional
ordinals contribute to explicit overflow assignment and modulus totals. Counters
saturate with a visible indication rather than wrapping. Summaries are neither
a background heartbeat nor a guaranteed final flush, so the last line may omit
the final interval's products. Warnings and errors remain visible. Detailed
`XC_CUDA_TIMINGS=1` output is independent of this summary policy. Logging does
not change the per-product report or numerical and artifact identities.

For dimension `n`, precision `p`, and `w` workers, the initial default workspace is
`min(available, 4 GiB, max(128 MiB, n²(8p + 4096) + 64 MiB) + 4 MiB × w)`.
This avoids reserving several GiB for a small solve. If actual exponent spans
exceed an output tile's preparation allowance, the output coordinates split
adaptively while preserving the complete panel sum. Only pre-computation
workspace refusals permit splitting; numerical and accelerator failures propagate.
If splitting is exhausted, the automatic allowance doubles up to the original
available-memory ceiling, attempting a smaller final increment when necessary.
Each attempt reserves its full allowance after the preceding attempt has released
its factors and temporaries. These execution choices preserve the arithmetic
identity and do not turn resource failures into mathematical solver refusals.

## CUDA product overlap

The product pipeline uses two nonblocking streams per admitted device.
Each lane owns pinned host staging, device buffers, a cuBLAS handle, and a
bounded workspace. One strided batched call computes the three exact digit
products for a residue. Both bounded lanes are packed first, then both input
transfers and products are submitted before the output transfers. Two residues
can be queued before synchronization. This permits transfers and GPU execution
to overlap when the installed hardware supports it. Dynamic work distribution and canonical
prime-order reconstruction remain unchanged; no peer access is required.

The pipeline consumes more admitted host and device storage than synchronous
execution. Automatic selection first admits synchronous execution and adopts
the pipeline only with at least 128 residue primes per admitted device,
available optional API symbols, and the same ordered admitted device set under
the extra storage charge. Otherwise it keeps synchronous execution before any
product work begins. An explicit `1` requires pipeline admission and may admit
fewer devices under the same budget. Input,
accumulation, reconstruction, and cleanup checks still apply. An unresolved
cleanup failure quarantines CUDA for the process; restart is required before
further device work. Neither operational setting changes artifact identity.

Hardware qualification covered one RTX 5090 on Windows with loaded CUDA runtime
13.4 and cuBLAS 13.4, including wide CRT, lane reuse, independent exact integer
comparisons, the automatic threshold, tight-budget fallback, and API sanitizer
checks. A transfer/kernel overlap trace was also observed. Smaller products can
be slower with the pipeline, which motivates the threshold. These measurements
do not establish Linux or physical multi-GPU performance, or an equal speedup
for a complete workload. CPU tests cover multi-device scheduling and admission.
For deployment qualification, the explicit hardware regression is:

```sh
XC_CUDA_PRODUCT_PIPELINE=1 cargo test --locked -p xc-numerics \
  --features cuda-product --lib \
  modular_product_cuda_pipeline_preserves_wide_crt_and_multi_device_order \
  -- --ignored --test-threads=1
```

## Certified Gauss-Legendre generation

Build `xc-spectral` with the `arb` feature to opt in to certified table
generation during Archimedean assembly. This feature also enables
`xc-numerics/arb` and requires FLINT headers and the FLINT shared library at
build and run time. The default generator remains unchanged even when the
feature is enabled.

```rust
use xc_numerics::quadrature::GaussLegendreGenerator;
use xc_spectral::ccm::hp::HighPrecConfig;

let mut config = HighPrecConfig::for_decimal_digits(1000);
config.archimedean_gl_generator = GaussLegendreGenerator::ArbCertified;
```

Use the ordinary managed source or run APIs with this configuration. The
generator is bound into the quadrature, integral, and Tau identities before
cache lookup. Descendant identities follow their exact source dependencies.
Existing legacy artifacts remain available to the default route. There is no
cache reset or migration. Direct research assembly also records the generator
in its returned identity. Concentration quadrature retains its existing route.

Arb isolates each Legendre root and weight with interval arithmetic and proves
that rounding to the requested MPFR precision is unambiguous. Nodes and weights
are rounded to nearest, ties to even, with an exact positive-zero central node
for odd orders. A bounded guard ladder refuses if it cannot establish that
rounding. Retained evidence is checked on reuse, and the existing point-rule
validator remains part of admission. Builds without `arb` reject this selection
before a parent-cache lookup; they do not silently substitute another method.

The admitted table domain is 1 through 16,384 nodes and 64 through 16,384 bits,
subject to a checked workspace estimate and the active run budget. Admission
is not a large-dimension performance qualification. Retained payloads contain
the uniquely rounded points and stable validation metadata; reuse reconstructs
the indexed root and weight enclosures rather than trusting stored ball text.

These values generally differ in their last bits from the legacy Newton
generator. Certified table rounding does **not** establish a quadrature
truncation bound, a finite-form assembly bound, or a continuum theorem. The
standard ground-index, residual, and root checks still apply to the selected
stored matrix. Select the generator explicitly and preserve its source graph
when comparing runs.

See the [quadrature API contract](../crates/xc-numerics/ARBITRARY_PRECISION_QUADRATURE.md)
for native requirements, failure behavior, and validation details.

## Features and entry points

| Features | Entry point | Operands and execution |
|---|---|---|
| `hp` | `exact_product::prepare`, then `PreparedProduct::cpu_product` | `rug::Float`; exact serial GMP integer arithmetic and final MPFR rounding |
| `modular-product` | `modular_product::multiply` | `num_bigint::BigInt`; exact CPU residues and CRT reconstruction; MPFR is not required |
| `cuda-product` | `modular_product::multiply` with `ProductBackend::Cuda` | Exact integer product using CUDA INT8 products; implies `modular-product` |
| `hp` | `PreparedProduct::modular_product` | Stored MPFR operands bridged to CPU, Auto, or forced CUDA; `hp` includes both product features |
| `hp` | `linalg::lu_factor_auto`, `inverse_iteration_auto_detailed` | Default dimension-bound CCM LU and refinement arithmetic, including evenness and bordered responses |

All matrices are row-major. A product has shape `(rows, inner, cols)` and
operand lengths `rows * inner` and `inner * cols`. The integer modular API
requires positive dimensions. The HP API also defines empty axes: a zero inner
dimension produces positive zeros, and an empty output dimension produces an
empty vector. Its modular adapter handles these cases without calling a backend.

## Arithmetic contract

The HP preparation preserves each finite input's **stored binary value** as an
integer times a power of two. It removes significand padding before aligning
exponents, so a large absolute exponent need not require a large integer.
The product is an exact sum of exact integer products, followed by one selected
rounding to the requested output precision:

- `Round::Nearest`: nearest, with ties to even.
- `Round::Down`: toward negative infinity.
- `Round::Up`: toward positive infinity.

Exact algebraic zero becomes positive zero, including products containing
negative-zero inputs. Other rounding modes refuse. Nonfinite inputs, unsupported
shapes or precisions, exceeded budgets, and unsupported exponent domains refuse.
The output conversion checks the intermediate integer exponent, the final
scaled exponent, and a possible rounding carry. It refuses unsupported
underflow/overflow rather than silently producing a zero or infinity.
Changing the active MPFR exponent range after preparation also refuses.

**An exact dot product with one final rounding generally differs from a legacy
sequence of rounded multiplications and additions.** Agreement on the real
arithmetic formula does not establish identical eigenvectors, pivot histories,
diagnostics, serialized records, or cache identities. Any solver integration
must preserve the required contract or identify its changed arithmetic explicitly.
The API proves no assembly-error, eigenvalue, inertia, or continuum statement by
itself; its inputs are the stored points supplied by the caller.

The modular engine chooses small prime moduli whose product is strictly larger
than twice `inner * max(abs(left)) * max(abs(right))`. Complete CRT reconstruction
then uniquely determines each signed integer answer. It does not truncate low
digits under cancellation. The CUDA backend splits centered residues into two
balanced base-128 digits and uses three INT8-to-INT32 products per prime. Checked
accumulation and integer-recombination bounds protect all intermediate sums.
Backend errors propagate; an error after CUDA selection does not silently
select the CPU backend.

## HP usage

The following example requires `hp`. The input Floats are caller-owned.

```rust
use rug::{float::Round, Float};
use xc_numerics::exact_product::{prepare, ExactProductLimits, ProductShape};

fn product() -> anyhow::Result<Vec<Float>> {
    let left = [Float::with_val(256, 3), Float::with_val(256, -5)];
    let right = [Float::with_val(256, 7), Float::with_val(256, 11)];
    let prepared = prepare(
        &left,
        &right,
        ProductShape { rows: 1, inner: 2, cols: 1 },
        128,
        ExactProductLimits {
            max_workspace_bytes: 64 << 20,
            max_scalar_products: 1_000,
            max_exponent_span: 4_096,
            max_integer_bits: 16_000,
        },
    )?;
    prepared.cpu_product(Round::Nearest)
}
```

With `hp,modular-product`, the same prepared value can call:

```rust,ignore
use xc_numerics::modular_product::{ProductBackend, ProductLimits};

let result = prepared.modular_product(
    Round::Down,
    ProductBackend::Cpu,
    ProductLimits {
        max_workspace_bytes: 128 << 20,
        max_input_bits: 16_000,
        max_modular_scalar_products: 1_000_000,
        workers: 1,
    },
)?;
```

Select `ProductBackend::Cuda` explicitly when built with `cuda-product`.
The bridge converts signed binary digits directly; it does not use decimal text
or binary64. Immutable prepared integer operands and their scales are available
for inspection. No public API accepts arbitrary externally computed integers as
an authenticated product.

The integer API takes the two `BigInt` slices, dimensions, backend, and
`ProductLimits`. Its `ExactModularProduct` result includes the exact entries,
dimensions, modulus/reconstruction sizes, workspace estimate, and separate
planning, conversion, product, and reconstruction timings. These are invocation
stage measurements, not a claim of complete-solver acceleration.

## Resource and implementation limits

`ExactProductLimits` bounds aligned integer width, exponent spread, exact scalar
multiply-add count, and a checked workspace estimate. The count is not a
bit-complexity bound. `ProductLimits` separately bounds modular input width,
workspace, the number of scalar products across all primes, and worker count.
The finite prime set, padded kernel dimensions, and accumulator bounds can
refuse a request even when its caller-selected limits would otherwise allow it.

For the HP modular adapter, `max_workspace_bytes` is a **combined** budget.
The prepared product's integer/Float workspace estimate and the binary bridge
estimate are subtracted before the remaining allowance reaches the modular
engine. Original caller-owned input Floats are excluded. The engine accounts
for its configured worker stacks and arithmetic arrays, while allocator overhead
and vendor CUDA context/library storage require process-level admission. These
policy estimates are not measured RSS limits or complete device-memory limits.
Callers must coordinate concurrent operations and handle allocation/backend
failures. CUDA conversion, transfers, and CRT reconstruction can dominate a
small product; no speedup is promised by the backend choice.

## Build and CUDA boundary

CPU builds use the existing Rust dependencies; HP additionally needs the
repository's Rug/GMP/MPFR build prerequisites.

```sh
cargo build --locked -p xc-numerics --features modular-product
cargo build --locked -p xc-numerics --features hp,modular-product
```

The CUDA feature requires no CUDA headers, link libraries, or `nvcc` at build
time. To execute on a GPU, install a compatible NVIDIA driver, a device with
compute capability at least 7.5, and CUDA runtime and cuBLAS shared libraries
from supported major versions 12 or 13. Runtime loading searches
`XC_CUDA_LIB_DIR`, then `CUDA_PATH/bin/x64` and `CUDA_PATH/bin` on Windows or
`CUDA_PATH/lib64` on Unix, then the platform loader. Windows loading excludes
the working directory from its implicit search. Libraries are never bundled.

The loader recognizes `cudart64_13.dll` / `cudart64_12.dll` and
`cublas64_13.dll` / `cublas64_12.dll` on Windows, and the corresponding
`libcudart.so.13`, `.12`, or unversioned name and `libcublas.so.13`, `.12`, or
unversioned name on Unix. Required symbols and reported versions are checked;
selected devices run an exact integer-product self-test before workload dispatch.
Compatibility with an actual driver/library/device combination remains a runtime
qualification, not a promise inferred from a major version number.

Local hardware validation used one RTX 5090 with CUDA runtime 13.2. Two-device
hardware qualification and complete-workload timing remain for server validation.

```sh
cargo build --locked -p xc-numerics --features cuda-product
cargo build --locked -p xc-spectral --features hp
```

The project-owned FFI loads device/version/memory queries, `cudaMalloc`,
`cudaFree`, `cudaMemcpy`, and the cuBLAS handle, workspace, and `cublasGemmEx`
entry points. It uses signed-byte
inputs, signed-32-bit outputs/accumulation, explicit layout conversion and
padding, and owned resource cleanup. Compatibility must be qualified for the
installed library, driver, and device; it is not inferred from the feature flag.
See NVIDIA's [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html) and
[CUDA runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html).

Focused CPU tests can be run without a CUDA device:

```sh
cargo test --locked -p xc-numerics --features modular-product --lib modular_product
cargo test --locked -p xc-numerics --features hp,modular-product --lib exact_product
```

The CUDA regression is ignored by default and must be selected explicitly:

```sh
cargo test --locked -p xc-numerics --features cuda-product --lib modular_product_cuda_exact_cancellation_and_rectangular -- --ignored --test-threads=1
```

The tests cover independent exact references, cancellation over thousands of
bits, signs, rectangular layouts, directed rounding, and resource refusals.
They validate these primitives; complete primary behavior also requires solver,
cache-identity, and acceptance regressions. Distribution and dependency scope are recorded in
[Third-party review](THIRD_PARTY_REVIEW.md).
