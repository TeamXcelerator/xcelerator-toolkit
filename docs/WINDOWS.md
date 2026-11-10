# Native Windows high-precision builds

The v0.17.1 Windows build lane uses a native x64 UCRT toolchain with Rust
1.99.1, GMP, MPFR, and FLINT. It does not require WSL, a Linux container, or a
virtual machine. MPFR still performs arbitrary-precision arithmetic with the
declared rounding and precision; CUDA products retain their exact integer
reconstruction and acceptance checks.

## Install the native dependencies

Use a checkout and native dependency directory without whitespace, such as
`C:\src\xcelerator-toolkit`. The arithmetic binding's native build discovery
does not support spaced paths in this lane; the scripts reject them early.

From PowerShell in that checkout, open a shell whose execution policy applies
only to that process:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass
```

In the new shell, install the dependencies and build:

```powershell
.\tools\windows-bootstrap.ps1
. .\tools\windows-env.ps1
cargo build --workspace --release --features hp,arb --locked
```

The bootstrap creates a private MSYS2 installation under
`target/native-windows/msys64`. It verifies the initial archive's SHA-256 and
uses signed packages from the official repository. It does not register an
installation, edit the global PATH, change the default Rust toolchain, or
invoke GitHub Actions. The execution-policy argument affects only that child
PowerShell process; it does not change the machine or user policy.

MSYS2 supplies native Windows compilers and libraries. Its build utilities do
not introduce a WSL runtime requirement into compiled Toolkit applications.
The bootstrap installs the exact native package versions and SHA-256-bound
package/signature bytes in `tools/windows-dependencies.json`. MSYS build
utilities can update independently; the qualified native compiler and arithmetic
libraries remain pinned. `packages.txt` records the resulting installation.
An existing qualified installation is checked and reused without updates.
If a pinned download is unavailable, setup stops rather than choosing a newer
version. An existing package cache can be selected with `-PackageCache`.
Archive the package inventory, compiler output, Cargo lockfile, library hashes,
and source revision with research runs.

To check the installed compiler, arithmetic headers, and runtime DLL bytes
without installing, updating, or downloading anything:

```powershell
.\tools\windows-bootstrap.ps1 -VerifyOnly
```

The supported lane uses the MSYS2-packaged UCRT Rust GNU **host** compiler.
Use its matching GCC, headers, GMP/MPFR, and FLINT. The environment script sets
`XC_NATIVE_PREFIX`, compiler/linker paths, pkg-config paths, and DLL discovery
for the current PowerShell process. A different installation can be selected
explicitly:

```powershell
. .\tools\windows-env.ps1 -Prefix C:\msys64\ucrt64
```

The alternate prefix must match the pinned compiler and native-file hashes.
The environment script checks those bytes before selecting its tools or DLLs,
including the compiler implementation libraries and transitive arithmetic DLLs.
Qualify a changed dependency set separately before updating the lock.

The scripts explicitly select `x86_64-pc-windows-gnu`; binaries are written
under `$env:CARGO_TARGET_DIR\x86_64-pc-windows-gnu\release` for release builds.

The current `gmp-mpfr-sys` dependency rejects MSVC for high precision. Selecting
an MSVC compiler and merely adding these DLLs to PATH is insufficient. Do not
mix UCRT and legacy MSVCRT static objects or independently installed GMP/MPFR
copies. FLINT and Rug must use the same native arithmetic libraries.

## Validate locally

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows-test.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\windows-test.ps1 -HighPrecision
```

The runner records workspace tests, strict Clippy, an external consumer,
documentation, and documentation tests. The general regression lane selects
CPU execution; the explicit device tests below qualify CUDA separately.
The runner disables remote cache access and
publication for these checks, preserves logs, and stops on failure. It imposes
no calculation timeout. Numerical identity and exact CUDA qualification are
separate from merely compiling successfully.

## Runtime and artifact contracts

Windows Arb builds execute a native MPFR probe before admitting parallel
work. The probe requires TLS and matching headers/runtime versions. Both C
bridges use the same explicit native prefix. Numerical spectral Arb entry
points also check loaded FLINT/GMP/MPFR versions, object sizes, TLS, and the
calling thread's MPFR exponent context. Prepared evaluations check their actual
worker before sharing native objects. Compatible inputs retain the original
arithmetic and rounding.

Applications need the selected native DLLs at runtime. Keep the coherent
prefix's `bin` directory in the process PATH or deploy an audited set of DLLs
and required notices beside the executable. MSYS2 itself is a development
environment; WSL is unnecessary for application execution. The bootstrap is
not a distributable runtime bundle or a substitute for reviewing dependency
redistribution terms.

CUDA is optional at build time. GPU execution requires compatible native
Windows CUDA 12/13 runtime and cuBLAS DLLs, selected through
`XC_CUDA_LIB_DIR`. Device enumeration is insufficient: retain the Toolkit's
exact self-tests and CPU/GPU equality gates. Strict CUDA requests still fail
explicitly if an operation cannot be performed.

To exercise exact products on a selected idle GPU after building, set the
native CUDA DLL directory and run the device tests explicitly:

```powershell
$env:XC_CUDA_DEVICES = '0'
$env:XC_CUDA_LIB_DIR = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
cargo test -p xc-numerics --features hp,arb device_integer_products_cover_non_tile_aligned_reductions_in_both_modes -- --ignored --test-threads=1
cargo test -p xc-numerics --features hp,arb pipeline_device_matches_cpu_across_tail_batches_reuse_and_rejected_operands -- --ignored --test-threads=1
```

Adjust the directory and ordinal to the installed CUDA version and actual
device. These tests require real CUDA execution and compare with CPU results.

Canonical numerical payloads retain their existing precision, rounding,
source-identity and validation contracts. Archive encoding keeps the original
ZIP/compression dependencies and V1/V2 profiles; native Windows reproduces the
immutable encoder fixtures byte for byte. Build provenance, library versions,
platform paths and timings must still be recorded honestly. Historical packages
remain authenticated through their original digests. Never relabel old producer
evidence or overwrite an existing mathematical artifact to imitate another build.

Source/target-bound execution checkpoints are not portable restart files
between Linux and Windows. The opt-in `HpRuntimePolicy::parallel_gl_roots`
schedule is available in the qualified native x64 GNU/UCRT build with `arb`.
The native C probe must establish UCRT and the MPFR TLS/version contract;
runtime checks also verify the loaded native libraries and MPFR context.
A GNU target triple by itself is insufficient. WSL, macOS, and other Windows
builds remain outside this policy's supported scope.

Root scheduling retains the serial root kernel, final ordering, and full
fresh-rule validation. It changes neither precision nor artifact keys. The
policy remains false by default and is recorded in execution provenance when
enabled. It helps cold, expensive single-table construction; existing compatible
tables are reused normally. See [performance reporting](PERFORMANCE_REPORTING.md).

Native Windows performance relative to WSL is a measurement, not a platform
guarantee. Use the same inputs, precision, arithmetic policy, worker count,
backend, and cold/warm-cache conditions for a comparison.

For editor integration, start VS Code from the PowerShell session configured
by `windows-env.ps1`. A separate editor session that still uses an MSVC host
will not inherit the native high-precision build environment.
