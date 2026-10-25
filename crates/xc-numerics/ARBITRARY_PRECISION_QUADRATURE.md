# Opt-in certified Gauss–Legendre rules

The existing HP APIs and `GaussLegendreGenerator::LegacyNewton` retain the
released arithmetic, semantic keys, payloads, and compatibility-cache paths.
The default does not change when the `arb` feature is enabled.

`GaussLegendreGenerator::ArbCertified` is an explicit alternative:

```rust,ignore
use xc_numerics::quadrature::{
    try_gauss_legendre_nodes_with_generator, CacheMode, GaussLegendreGenerator,
};
let (nodes, weights) = try_gauss_legendre_nodes_with_generator(
    9, 128, CacheMode::Off, GaussLegendreGenerator::ArbCertified,
)?;
```

Build with `cargo build -p xc-numerics --features hp,arb`. The feature requires
a separately installed FLINT 3 shared library and development headers; Linux
with FLINT 3.0.1 is the qualification environment. The narrow C bridge calls
public FLINT/Arb APIs. No FLINT source or library binary
is vendored or redistributed. The optional `cc` build dependency is already
used in the workspace. Existing FLINT, GMP/MPFR, and Rug license and native ABI
compatibility notices apply.

The `arb` feature enables `gmp-mpfr-sys/use-system-libs` for the entire Cargo
dependency graph, including Rug. Install FLINT, GMP, and MPFR development
packages from the same native environment. Rug, the bridge, and FLINT must
resolve to one shared GMP library and one shared MPFR library; bundled static
copies mixed with FLINT's shared dependencies are unsupported. Builds without
`arb` keep Rug's bundled GMP 6.3.0 and MPFR 4.2.2. See the upstream
[system-library requirements](https://docs.rs/gmp-mpfr-sys/1.7.1/gmp_mpfr_sys/#optional-features).

The qualified Arb system combination is FLINT 3.0.1, MPFR 4.2.1, and GMP 6.3.0.
FLINT headers must declare `FLINT_USES_TLS`; builds without thread-local storage
are rejected at compile time. This qualification does not cover other FLINT builds.
The dependency checks native ABI/version compatibility during the build. A
native compile-and-run probe requires MPFR thread-local storage
(`--enable-thread-safe`) before any HP route can use the system stack. Cross
compilation and static musl linkage are refused for this feature. The
bridge additionally requires FLINT 3 headers and matching header/runtime version
strings and thread-local storage before generation or proof replay. It also
checks ABI sizes, shared
MPFR exponent context, and serial FLINT settings. Deployments must preserve the
linked library set, with no custom allocator or symbol interposition overrides.
For a Linux binary, inspect `ldd` and `nm -D --defined-only`: exactly one GMP
and one MPFR shared library should be loaded, and the executable must not define
`__gmp*` or `mpfr_*` library implementations. A matching version string alone
does not establish single-library linkage.

The generator uses indexed root-and-weight enclosures, with guards of 64,
128, 256, and 512 bits above the requested precision. Both values must admit
unique nearest/ties-to-even rounding. Full-table checks require disjoint
ordered interior root balls, positive weights, polynomial/derivative and weight
formula consistency, and a total-weight enclosure containing two. The root API
supplies root existence and index; residual containment alone is not a proof.
Every resulting point table also passes the existing full Toolkit Legendre
root/weight validator. Odd-order center nodes are canonical positive zero.

The rounding check first uses Arb's fast predicate. When its additional MPFR
ternary-result requirement is unresolved, equal nearest images of outward
enclosure endpoints can still certify the point value. Endpoint precision is
bounded at the requested precision plus 576 bits. This accepts exact weights
such as one without claiming a known inexact-direction result or relaxing
unique rounding, finite-value, or signed-zero checks.

These are certified rounded nodes and weights, not an integrand quadrature
error bound. They generally differ from legacy Newton point tables. The new
managed semantic/logical keys and `gl_cache/arb-certified-v1/` compatibility
directory are separate before lookup. Managed new records require reader
version 0.17.0. Callers must bind the generator identity into dependent
integral/matrix keys before looking up those parent artifacts.

Payloads retain only the uniquely rounded points and fixed certification
semantics. Backend-dependent ball strings are not serialized. Warm reuse
reconstructs indexed certified root/weight enclosures and requires exact
agreement with every stored point, including the sign of zero, as well as the
full point validator. Certification metadata by itself never authorizes reuse.
Within one managed resolution, an exactly matching immutable payload and MPFR
exponent context can reuse the validator's decoded result for the final return.

`ensure_available()` and `validate_rule(order, precision_bits)` permit early
admission before dependent lookups. A build without `arb` refuses explicit Arb
selection, including required reuse; it never falls back. The bounded domain
is order 1–16384 and precision 64–16384 bits. Its checked workspace estimate
uses limb-rounded bytes for ball/output storage and conservative native scratch,
without a fixed per-call byte ceiling. A `RunResources` context active on the
calling thread must admit that workspace; unscoped calls do not establish a
hidden process budget. Scheduled GL and integral-batch callbacks propagate the
owner's context explicitly. This is conservative allocation admission, not an OS RSS cap
or a guarantee that every admitted request resolves within the finite ladder.
Unsupported MPFR ABI/exponent contexts and nonserial FLINT settings refuse
without changing global thread settings or the caller's exponent range.

Construction is serial within each table; a caller can schedule independent
tables. A larger Rayon pool does not enable another FLINT thread pool. The
scheduled APIs retain the existing point-validation schedule and do not
silently change the numerical generator.

Primary algorithm/API references: [FLINT Legendre root API](https://flintlib.org/doc/arb_hypgeom.html#c.arb_hypgeom_legendre_p_ui_root),
[certified rounding](https://flintlib.org/doc/arb.html#c.arb_can_round_mpfr), and
[Johansson and Mezzarobba, Fast and rigorous arbitrary-precision computation of Gauss–Legendre quadrature nodes and weights](https://arxiv.org/abs/1802.03948).
