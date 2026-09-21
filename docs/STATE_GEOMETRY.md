# Retained CCM state geometry

`ccm::state_geometry` derives `ccm_state_geometry_analysis` from the actual
retained `ccm_weil_eigenpair`. It does not solve for another eigenstate, need a
target file, or use zeta ordinates. The child belongs to `ccm-evidence`.

## Mathematical convention and scope

For the full real coefficient vector xi[-N..N], put L = log(lambda_squared),
x = log(u), and

    F(x) = sum_j xi[j] exp(2*pi*i*j*(x/L + 1/2)),  -L/2 <= x <= L/2.
    f(x) = orientation * F(x) / sqrt(L * sum_j xi[j]^2).

Parseval gives unit L2(dx) norm. The orientation makes the raw center positive;
when the center is zero, it makes the largest-magnitude coefficient positive
(the last in stored order breaks an exact tie). Raw center and coefficient norm
retain their original values and sign. No division by the center is needed.

The payload retains:

- Raw coefficient norm and center; unit L2 center and, for an exactly even
  coefficient vector, the signed mass integral. These scalars follow directly
  from Fourier coefficients, without sampled integration.
- Relative coefficient evenness defect ||xi - reverse(xi)|| / ||xi||.
- Two periodic-trapezoid grids: norm check, moments integral x^j |f|^2 dx for
  j = 1, 2, 4, and mass outside |x| >= L/8, L/4, 3L/8. Shell boundaries get
  half weight. Matching periodic endpoint values are counted once; the two
  endpoint values of the odd spatial moment are averaged.
- Sampled minimum and negative-part L1 mass of f for an exactly even vector.
  These concern f itself, not an existing f-minus-target residual artifact.
- Absolute coarse/fine differences for norm, the three moments, and the three
  tails, in that order. Both grids also retain their sign measurements.

Odd and mixed coefficient states still produce norm and energy geometry.
Under this unrotated Fourier convention they can be complex, so the signed
mass/minimum/negative-part fields are null with an explicit applicability
status. No even state is substituted and no complex phase is silently changed.

All values are high-precision **point diagnostics**. Higher working precision
cannot restore missing source digits. The artifact does not enclose source
uncertainty, prove global nonnegativity between nodes, prove ground-state
selection, or establish convergence. Grid agreement is not certification.

## Cost, precision, and identity

The default coarse grid has 8(N+1) intervals; the fine grid has twice as many.
Working precision defaults to the retained source precision. Explicit retained
requests can increase precision and grid resolution. Down-rounding and grids
below the mode-resolution budget are rejected. At most 131,072 coarse
intervals and 1,000,000 working bits are accepted as structural limits, not
promises of low resource cost.

Sampling is parallel with fixed 64-point chunks, fixed merge order, reused
recurrence scratch, and compact per-chunk summaries. Output size is independent
of grid length. Warm validation checks provenance, shape and finite scalars;
it does not redo the grid calculation. Options and exact parent dependencies
are part of the semantic key. Public publication requires a public parent.

## Ultra and historical plans

New shared Ultra plans use `ccm-measurement-capture-plan-v3` and request
`state_geometry`. The retained-run adapter handles this ID independently of
other diagnostics. Lower levels retain their previous request sets. Serialized
v1/v2 plans retain their version and do not acquire new requests; old receipts remain evidence
of their original capture policy.

Applications must execute the shared plan's requested IDs through
`RetainedCcmRun::capture_diagnostic`. Calling `primary_options()` alone never
executes the complete shared plan. Existing pinned applications require a
separate dependency update and integration check before this changes their
runs.

## Retained files and later backfill

The same managed producer is available without any primary computation:

```sh
cargo run --release -p xc-spectral --features hp \
  --example ccm_state_geometry_retained -- request.json new-output.json
```

Example request (paths resolve relative to the request):

```json
{
  "manifest": "source/manifest.json",
  "payload": "source/payload.json",
  "approved_payload_digests": ["REPLACE_WITH_SELECTED_MANIFEST_CONTENT_DIGEST"],
  "cache_root": "derived-cache",
  "options": null
}
```

Supply the verified logical JSON payload, not its ZIP transport bytes. Select
and approve the source digest from the intended capture's authenticated
manifest; the allowlist is an explicit selection, not a new signature scheme.
The reader verifies immutable manifest quality, size, hash, supported schema,
finite scalars, vector length, cutoff and precision. Validated, CrossChecked
and Certified parents are accepted without upgrading the child's assurance.

The example creates/reuses a local managed child and writes a new output packet
containing its manifest and report. It refuses to overwrite the output file.
It neither downloads sources nor publishes artifacts. Its source-bound child
can later participate in a managed publication with the complete parent closure.
The parent payload, historical capture receipt and prior numerical results
remain unchanged. Repository-wide backfill planning/execution is a separate
step; the [batch backfill command](RESEARCH_BACKFILL.md) provides resumable
retained-source acquisition.

See the [payload schema](schemas/ccm-state-geometry-analysis-v1.schema.json)
and [capture integration](CAPTURE_LEVELS.md).
