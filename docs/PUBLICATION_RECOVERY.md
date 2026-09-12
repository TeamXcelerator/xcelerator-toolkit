# Recover publication without repeating a calculation

A validated artifact can be saved in the workstation cache even if its subsequent
publication staging fails. Preserve the cache, original capture receipt and run
journal. A packaging limit is not evidence of a numerical error.

## Resource limits

Managed sessions default to the normal workstation policy, including a 20 GiB
encoded-transfer ceiling. For a larger machine or artifact, set
`XC_RESOURCE_POLICY_FILE` to a JSON file containing a complete
`xc_core::ResourcePolicy`. Limits are bytes or seconds as named; `null` means no
explicit ceiling. Unknown fields, zero limits and files larger than 64 KiB are
rejected. Resource settings do not change numerical precision, mathematical
identities, assurance or publication authorization. They do not reserve hardware
or replace the caller's disk/memory availability checks.

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
They do not repeat the numerical producer. The original failed capture receipt
stays failed: record publication recovery and any retrospective completeness
assessment separately, tied to the original and recovered artifact identities.
