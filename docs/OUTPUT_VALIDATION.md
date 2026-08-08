# Cache output validation

Cache output validation checks a proposed toolkit change for Category A output
preservation. It recomputes the artifacts exercised by a normal claim run and
compares every newly computed payload byte-for-byte with the corresponding
artifact in the current reference cache.

This mode validates a change before it is accepted. It is not a comparison
between toolkit releases, and it does not reject a reference merely because its
producer toolkit version differs from the running package version. Existing
cache acceptance checks still apply, and producer information is retained as a
report diagnostic.

## Running validation

Set `XC_CACHE_MODE=verify` and run the claim through its normal entry point. For
example, in PowerShell:

```powershell
$env:XC_CACHE_MODE = "verify"
cargo run --release --example <claim-example> -- <normal-claim-arguments>
```

The optional validation settings are:

- `XC_VALIDATION_REFERENCE=private_public` selects private artifacts first and
  public artifacts second. This is the default. `private` and `public` select a
  single reference source.
- `XC_VALIDATION_CACHE_ROOT=<path>` selects the isolated validation cache. By
  default it is a sibling of the ordinary cache whose directory name ends in
  `-validation`.
- `XC_VALIDATION_REPORT_ROOT=<path>` selects the report directory. By default it
  is `<validation-cache>/reports`.

The toolkit repository ignores the conventional local validation directory
names `cache-validation`, `xcelerator-validation`,
`.xcelerator-cache-validation`, and `.xcelerator-validation`. Consumer
repositories should ignore the same names. Keep an arbitrary custom validation
root outside a source tree or beneath an ignored directory.

The ordinary cache root and validation root must be distinct and neither may
contain the other. Verification also rejects publication, staging, replacement,
and remote-mutation settings.

## Execution behavior

Verification downloads and accepts reference artifacts through the same cache
policy used by ordinary execution, but completed reference results are never
returned as the result of a requested computation. Each requested artifact is
recomputed and written through the normal artifact-writing path into
`<validation-cache>/computed`. These are real payload and manifest files, not
synthetic report records.

Reference artifacts may still control the computational route. In particular,
CCM continuation seeds may be reused from the reference cache so a validation
run takes the same seeded route as the baseline run. Route probes and seed
discovery consult the reference cache only; previously computed validation
artifacts cannot influence a later run.

After each computation, the toolkit resolves the same semantic identity in the
reference-only resolver and compares the exact payload bytes. Semantic keys and
public numerical function names are unchanged.

## Result and reports

Validation continues after an individual mismatch or missing reference and
across sequential managed-cache sessions belonging to the same claim. Session
finalizers write cumulative `in_progress` checkpoints. The claim's terminal
finalizer writes a `completed` or `aborted` report, replaces `latest.json`,
prints an ASCII summary, and returns an error when a completed claim has any of
the following conditions:

- a computed payload differs from its reference payload;
- a required reference artifact is absent;
- the same key produces different payloads within the validation run; or
- the run performs zero comparisons.

A successful report requires at least one comparison and requires every
comparison to match. Reports include artifact keys, semantic identities,
payload and manifest digests, dependency information, seed provenance when
available, timing, first differing byte offsets, and first-divergence
classification. The isolated computed cache and reports are never published by
verification mode.
