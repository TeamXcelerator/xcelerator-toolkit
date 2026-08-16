# Performance reporting

Introduced in version 0.13.5, this facility records operational timing evidence
without changing numerical or cache behavior.

The toolkit can write an opt-in process-wide JSON timing sidecar for controlled
before-and-after performance studies. It is disabled unless `XC_PERF_REPORT`
names an output file.

```bash
XC_PERF_REPORT="$PWD/performance-reports/claim1a.performance.json" \
  cargo run --release --features hp -- <ordinary claim arguments>
```

The report aggregates instrumented managed-cache and high-precision CCM stages
in the process. The sidecar is refreshed when the last open top-level stage
finishes, avoiding repeated synchronized writes from nested or concurrent
Gauss--Legendre work. A computation error still leaves timings for stages that
completed before the error. Records
include invocation counts, total/minimum/maximum elapsed nanoseconds, problem
shape, precision, Rayon worker count, HP runtime mode, cache disposition, and
Gauss-Legendre batch scheduling where applicable.

For matrix-construction stages, `retained_hp_entries` records the number of
high-precision entries retained in destination matrices. It is a deterministic
problem-shape measure, not a measurement of peak resident memory.

Performance reporting is diagnostic only. The report path and its contents do
not enter semantic keys, execution fingerprints, artifact payloads, manifests,
cache publication, or output-validation reports. `performance-reports/` and
`*.performance.json` are ignored by Git.

## Controlled comparisons

Use the same release build, claim arguments, precision, runtime policy, worker
count, and cache state for baseline and candidate runs. Record every run and
compare medians rather than choosing the best observation.

For cold Gauss--Legendre measurements, the `scheduling` field distinguishes
`table_parallel_root_serial`, `table_serial_planned_roots`, and
`table_serial_root_serial`. Root-level scheduling is opt-in through the
default-false `HpRuntimePolicy::parallel_gl_roots` field, is recorded in the
execution fingerprint, and must be qualified on native Linux. It is not a
supported WSL mode because concurrent GMP allocation has previously caused
non-deterministic allocator failures there. The numerical runtime rejects this
policy on WSL, Windows, and macOS before numerical work begins.

The default `false` policy value is omitted from serialized runtime policy, so
existing provenance bytes and cache identities retain their established form.
The opt-in `true` value is fingerprint-visible, while mathematical artifact
semantic keys remain unchanged. Use a genuinely cold cache when measuring the
new construction schedule; an existing compatible GL artifact is correctly
reused regardless of the schedule that originally produced it.

A cold managed-cache measurement uses a newly created scratch
`XC_CACHE_ROOT`, `XC_CACHE_REMOTE=none`, `XC_PUBLISH_TARGET=none`, and disabled
publication execution. Do not delete workstation, production, registry,
validation-reference, or published artifacts to manufacture a cold state.

`XC_CACHE_MODE=verify` remains the authority for proving that an optimization
preserves current artifact payload bytes. A faster performance report cannot
override a validation mismatch.
