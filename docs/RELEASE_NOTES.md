# Release notes

## 0.13.5

Version 0.13.5 is a payload-preserving performance and observability release.

- High-precision CCM component construction writes directly into final matrix
  storage. This removes full-size intermediate coordinate and result
  collections without changing the established MPFR operation order.
- `XC_PERF_REPORT` enables a process-wide JSON timing sidecar for controlled
  performance studies. Reporting is lazy when disabled, aggregates nested and
  concurrent work, and remains outside cache identity, payloads, manifests,
  publication, and validation evidence.
- An opt-in Gauss--Legendre root schedule can use idle Rayon capacity when a
  cold precompute batch contains too few independent tables. The owning planner
  selects table-level or root-level parallelism, never both.
- Experimental root-level scheduling fails closed on WSL, Windows, and macOS.
  It remains disabled by default and requires native-Linux qualification before
  production use.

### Compatibility

The release does not alter mathematical semantic keys, artifact schemas,
precision targets, solver selection, convergence rules, or default scheduling.
The new runtime-policy field defaults to `false` and is omitted from serialized
policy in that state. Existing compatible cache artifacts and established
default provenance therefore remain reusable without migration.

Exact reference tests compare the optimized component paths with their frozen
materializing counterparts at multiple precisions and Rayon worker counts.
`XC_CACHE_MODE=verify` remains the release authority for exact artifact-payload
comparison during production qualification.

See [Performance reporting](PERFORMANCE_REPORTING.md) for controlled benchmark
guidance and the scope of recorded diagnostics.
