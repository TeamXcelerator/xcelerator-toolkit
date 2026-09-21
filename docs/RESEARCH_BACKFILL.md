# Additive retained-source backfill

The HP example `ccm_research_backfill` derives new children from explicitly
selected retained sources. It never invokes a primary eigensolve, root solve,
network retrieval, publication or claim rerun. It does not repair or replace
source bytes. Use this as the building block for a one-time historical batch.

## Run a frozen batch

Build locally with the documented [HP prerequisites](../README.md):

```sh
cargo build -p xc-spectral --release --features hp --example ccm_research_backfill --locked
./target/release/examples/ccm_research_backfill batch.json output-directory
```

Paths in the batch are relative to its file, or absolute. Each source has a
`manifest`, a decoded logical `payload`, and an optional
`maximum_payload_bytes` (512 MiB by default). Compressed shard objects must be
verified and decoded through the existing retained-source reader first; do not
pass ZIP transport bytes as the logical JSON payload.

```json
{
  "schema_version": 1,
  "approved_payload_digests": ["REPLACE_WITH_EXACT_SOURCE_SHA256"],
  "cache_root": "derived-cache",
  "jobs": [
    {
      "id": "state-geometry",
      "task": {
        "operation": "state_geometry",
        "state": {
          "manifest": "sources/state.manifest.json",
          "payload": "sources/state.payload.json"
        }
      }
    }
  ]
}
```

The placeholder must be replaced by the actual manifest/payload digest. The
allowlist is not a substitute for hashing: bytes, size, quality, kind and exact
required parent dependencies are checked again. States with unsupported source
schemas cannot be silently reinterpreted.

## Operations

All operations accept the `operation` discriminator shown below. Field shapes
for options, specifications and source bundles are the public types in
`ccm::retained_evidence`, `ccm::extended_research` and `ccm::state_geometry`; unknown fields are rejected.

| Operation | Required fields; optional fields |
|---|---|
| `state_geometry` | `state`; optional `options` |
| `operator_energy` | `state`, `matrix`; optional `parent_manifests` for the exact Tau/factorization/sector ancestry |
| `indexed_transform` | `state`, `roots`, `secular`; optional `options` |
| `indexed_dataset_transform` | `state`, managed `dataset`; optional `options` |
| `root_window` | `state`, `roots`, `secular` |
| `reference_source` | `spec`: definition, cutoff, precision and finite Fourier coefficients |
| `reference_dataset` | `spec`: role, attribution, coordinate, precision and ordered points |
| `reference_projection` | `state`, `inputs`: reference, basis and projection policy |
| `observation_packet` | `observation`: original text, attribution, definition, hypotheses, borrowed inputs and limitations |
| extended_research | diagnostic, state; optional matrix, paired roots/secular, parent_manifests, paired input/input_sha256, options |
| `stabilization` | ordered `states`, `options`: working precision, relative tolerance and consecutive steps |

Tau ancestry manifests must also have admitted payload digests. Their exact
metadata chain is checked without loading factorization payloads. Imported
known-ordinate datasets are references, not newly certified zeta zeros.
`retained_ccm_roots` is reserved for the authenticated root-source route.

## Resume and assess

The output directory is bound to a serialized batch digest. Successful job
packets contain the child manifest and report. Repeating the same batch reuses
compatible children and preserves output bytes. Existing mismatched output is
never overwritten. Each attempt and summary is a new file, including failures;
a later retry cannot erase the earlier failure record. Atomic file creation
keeps interrupted writes from becoming a partial canonical result.

Independent jobs continue after a failure. The command exits nonzero if any job
is incomplete, while preserving successful packets. Summary entries distinguish
retention from numerical outcomes and retain per-row statuses. `RETAINED` means
a report was stored, not that a convergence test or proof passed. Inspect the
report's scope, missing rows, unresolved normalization/pivots/derivatives, and
unbounded tails before using the measurements.

Changing the batch requires a new output directory; the same child cache can
be shared. Sources stay immutable. No source deletion or numerical recomputation
is required solely to add these children. Publication is a separate managed
operation with exact dependency closure and the normal destination policy.
Historical campaign discovery, publication and paper upgrades are separate
steps; this release does not claim that any existing campaign was backfilled.

## Local acceptance test

`tools/test_research_backfill.py BINARY --packets DIRECTORY` creates synthetic
sources and exercises all thirty kinds, warm reuse, failed rows, corrupted input,
independent-job survival, frozen batch enforcement, and output preservation.
It does not contact a remote service or modify scientific source data.

## Extended diagnostic and target-only batches

Use operation extended_research with a diagnostic name from
[extended diagnostics](EXTENDED_RESEARCH.md). The name external_source retains
the numeric input bundle itself. Supply the external input path and raw-file
SHA-256 together. Changing that file invalidates the frozen batch.

For a new target, create a new input file and batch/output directory, reuse
the exact eigenstate, and request only target-dependent diagnostics. Projection
and signed-transform refreshes need no Tau or roots. Arithmetic energy needs
the exact parent Tau; directional response also needs roots and secular source.
Compactness needs only the eigenpair.

A retained missing_input report records an unavailable measurement. Row budget
limits and unresolved denominators remain in the summary. Technical failures
return nonzero while preserving independent successes. Original claim receipts
are not rewritten to imply that the new measurements existed at the time.

## Completion diagnostics and reusable references

The five completion names are `capture_preflight`, `consistency`,
`configuration_comparison`, `band_reconstruction`, and `transform_enclosure`.
They use the same `extended_research` operation and producers as live capture.
Build with `--features hp,arb` for certified finite transform enclosures. An HP-only
build retains a qualified missing-feature outcome for that diagnostic.

The frozen `input` file can also contain a source-independent
[reference preparation](ULTRA_COMPLETENESS.md#external-reference-preparation).
The driver detects the format, prepares it against the selected source and roots,
and still requires the original file hash. A new preparation is a new batch;
existing successful output is preserved. Additive child backfill and local
checkpoint recovery never upgrade an old claim receipt retroactively.
