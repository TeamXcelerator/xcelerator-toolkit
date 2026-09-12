# Complete positive movable-root discovery

Enable the `xc-spectral/arb` feature and install FLINT 3 or newer to acquire
all positive movable roots of an exactly even retained CCM point source.
The complete route includes roots above the largest retained pole. The older
pole-span scan remains the default for compatibility and bounded queries;
its boundary is a search policy, not a universal spectral-reach bound.

```rust,ignore
use xc_spectral::ccm::{CcmParams, window::ZeroTarget};
use xc_spectral::ccm::hp::{HighPrecConfig, IndependentRootDiscoveryOptions};
use xc_spectral::ccm::hp::capture_run::RetainedCcmRun;

let params = CcmParams::from_lambda_sq_integer(2500, 50);
let cfg = HighPrecConfig::for_decimal_digits(1000)
    .with_adaptive_root_precision();
let run = RetainedCcmRun::independent(
    &params, &cfg, &ZeroTarget::FirstK { count: 50 },
    IndependentRootDiscoveryOptions::complete_positive(true), &cache,
)?;
```

Here `cache` is the application's managed `ArtifactCacheContext`. No reference
zero dataset participates in discovery. The `allow_incomplete` argument
retains available roots when the requested window is larger than the actual
positive-root count; callers must still check count, ordering and convergence.
With `false`, an insufficient window returns an explicit error.

## Method and scope

The route requires exactly symmetric weights and pole points. It constructs
the rational numerator in `s=t^2`, using the same finite MPFR pole values
consumed by root refinement. It does not replace them with ideal lattice
poles. Zero-residue poles and zero factors introduced by clearing the central
denominator are removed before counting positive movable roots.

A Cauchy coefficient bound covers all numerator roots. FLINT/Arb isolates
the polynomial roots and classifies every root against the positive domain.
Positive square-root intervals supply ordered source-only starting points.
The configured point solver refines these points. A nearest-seed-cell check
rejects ordinal crossings during refinement. Adaptive refinement promotes
the stored source points and independently checks the requested correction
target; it does not rebuild the matrix or assert more accurate source data.

This is a complete positive movable-root acquisition policy for a fixed
point source. It is not an interval certificate for the infinite CCM
operator, a full Fourier/determinant ordinal assignment, or an accuracy
claim against reference zeta zeros. The root artifact retains computed
assurance; full certification remains a separately requested capability.

Repeated roots, unresolved classifications, non-even inputs and point
precision too low to represent distinct starting points return explicit
errors. A source can have fewer positive real roots than its polynomial
degree. Missing roots are not supplied from a reference dataset.

## Retention and migration

Complete-range roots use mathematical semantics `ccm-root-range-v0.15.0-v10`
and a `complete-positive` logical namespace. Their semantic parameters name
the complete domain, exact even point-numerator isolation and stored MPFR
pole geometry. Their payload's completeness marker is
`complete_positive_movable_point_source`; their reader floor is v0.15.0.
A separate correction/verification policy records adaptive precision.

No new artifact kind or repository layout is introduced. Both artifact
catalogs can retain these roots using the existing `ccm_root_discovery_window`
family. Existing pole-span and fixed-guard root artifacts remain preserved.
Compatible matrices, eigenstates and secular sources remain reusable;
root-dependent diagnostics bind to the new root parent.

Applications must select the complete policy explicitly. Paper 2's ordinary
independent spectral scripts select it and build with Arb automatically.
A prior incomplete run is retained as historical evidence; a new invocation
can reuse its source and add the newly acquired root window and diagnostics.
Do not clear a cache to extend a discovery window.
