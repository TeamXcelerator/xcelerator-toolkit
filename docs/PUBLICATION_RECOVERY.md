# Recover publication without repeating a calculation

A validated artifact can be saved in the workstation cache even if its subsequent
publication staging fails. Preserve the cache, original capture receipt and run
journal. A packaging limit is not evidence of a numerical error.

On Windows, publication may place disposable Git object databases in a short
directory under `TEMP` when the journal path is long. The journal lock and an
ownership marker bind that directory to this publication. Retries clean up
stale owned Git data; completed attempts remove it. Artifacts, staging and
publication receipts remain in their original locations. Git commands explicitly
select their bare repository and cannot fall back to a parent checkout.
An unusually long `TEMP` path is rejected with a request to choose a shorter
temporary directory; no global Git setting needs to be changed.

## Resource limits

Managed sessions default to the normal workstation policy: 100 GiB temporary
disk, 100 GiB permanent disk, and a separate 20 GiB encoded-transfer ceiling.
These are software budgets, not measurements of available drive space. Temporary
reader sessions use the temporary-disk budget; their Git objects, retained parts
and reconstructed packages count toward it. Prefetch checks capacity before
downloading. For a larger machine or artifact, set
`XC_RESOURCE_POLICY_FILE` to a JSON file containing a complete
`xc_core::ResourcePolicy`. Limits are bytes or seconds as named; `null` means no
explicit ceiling. Unknown fields, zero limits and files larger than 64 KiB are
rejected. Resource settings do not change numerical precision, mathematical
identities, assurance or publication authorization. They do not reserve hardware
or replace the caller's disk/memory availability checks.

On Linux, use `export XC_RESOURCE_POLICY_FILE=/path/to/resources.json`; on
PowerShell, use `$env:XC_RESOURCE_POLICY_FILE = 'C:/path/to/resources.json'`.
Numerical working-memory settings do not implicitly change cache disk budgets.
Repository shard capacity is a separate publication policy, not this local
reader budget.

Example bounded policy for large artifact transport:

```json
{
  "profile": "external_compute",
  "maximum_memory_bytes": 1073741824,
  "maximum_temporary_disk_bytes": 137438953472,
  "maximum_permanent_disk_bytes": 137438953472,
  "maximum_transfer_bytes": 137438953472,
  "maximum_cpu_seconds": 86400,
  "maximum_wall_seconds": 86400,
  "maximum_threads": 8,
  "allow_out_of_core": true,
  "allow_distributed": false,
  "checkpoint_interval_seconds": 300
}
```

The same policy reaches managed remote reads, publication staging and Git
transport. Existing programmatic callers keep `ManagedArtifactCacheSession::new`;
use `new_with_resources(config, resources)` for explicit limits. Environment-based
callers use `ManagedArtifactCacheSession::from_environment`. A known encoded object
that exceeds its transfer cap is rejected before split-part copying. The complete
local object remains available for a subsequent recovery with sufficient limits.

## Exact retained-object recovery

The `xc-cache` example accepts an exact `DependencyRef` JSON with `key`
(`kind`, `logical_key`, `parameters_digest`), `content_digest` and
`required_quality`. Obtain these from the original local manifest, verify its
source identities against the original run, and use a separate staging directory.

```bash
export XC_CACHE_ROOT=/path/to/existing/cache
export XC_PUBLISH_STAGING_ROOT=/path/to/new/recovery-staging
export XC_RESOURCE_POLICY_FILE=/path/to/resources.json
export XC_PUBLISH_REPLACE=false
cargo run --locked --release -p xc-cache --example recover_publication -- artifact.json
```

This only stages locally. It ignores an inherited execute flag unless `--execute`
is also supplied. To publish, explicitly configure the intended destination and
`XC_PUBLISH_EXECUTE=true`, then repeat with `--execute`. The normal authentication,
repository routing, locking, payload verification and additive publication rules
remain in force. Ensure repository workflows will not run without authorization.

`stage_cached_artifact` resolves only the exact encoded artifact and dependency
closure. It has no numerical compute fallback. It hashes the encoded object and
verifies the canonical logical payload through the regular streaming staging
path, without loading or parsing the entire numerical JSON in memory. Existing
quality/provenance is required; recovery does not independently redo the domain
validation or promote assurance. Legacy unprofiled objects require a different
explicit migration and are rejected by this route.

Hashing, ZIP decoding, split-part creation and network transfer still take time.
Dependency preflight logs its destination, pass, family, reuse count and elapsed
time. Successful exact ancestry proofs are reused across phases of one
publication invocation. Proofs bind the destination, manifest, topology and all
observed repository heads, and expire at scheduled revocations. Each transaction
takes a fresh read snapshot; a changed head requires validation again. Mutation
authorization, leases and compare-and-swap checks always use the live store.
For wide dependency graphs, manifest, index and revocation metadata is fetched
in batches of at most 256 paths, followed by the same per-document validation.
No numerical payload object is requested by this ancestry prefetch. Local blob
sizes and small metadata bytes are read with batched Git commands. Disk growth
still requires reservations and post-fetch measurements; returning immutable
bytes from memory does not enumerate the temporary filesystem.
Small metadata and confirmed absent paths share a bounded invocation memo keyed
by repository, full Git commit and path, including registry and topology JSON.
The memo survives individual Git session cleanup, allowing later phases to reuse
authenticated bytes. It is released at invocation end; changing a store's resource
policy detaches its memo. Every caller still validates each returned document.
Metadata bytes are capped at 32 MiB (or one eighth of the memory allowance),
with at most 4,096 records and 1 MiB per document; proof storage is capped at
8 MiB (or one sixteenth of the allowance). Numerical payloads remain streamed.

When an existing compatible V2 producer satisfies a numerical parent, new
descendants reference that admitted producer's exact manifest. A fresh wrapper
with different provenance therefore does not force the same numerical payload
back into publication. Semantic keys and payload identities are preserved;
legacy V1 dependency semantics retain their exact-identity requirements.
Exact external public producer manifests and their complete admission closure
are authenticated before any family writes. Recovery does not repeat numerical
production. The original failed capture receipt
stays failed: record publication recovery and any retrospective completeness
assessment separately, tied to the original and recovered artifact identities.
