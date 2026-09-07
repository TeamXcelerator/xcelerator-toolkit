# Repairing retained CCM response artifacts

The `ccm_response_repair` example repairs schema-2 prime-power and cutoff-flow
root velocities from the exact original eigenpair and retained L2 tangent
vectors. It performs no matrix assembly, eigenstate solve, bordered solve, or
paper-claim run. The output uses the ordinary v3 response semantic identities
introduced in v0.15.0. See [numerical compatibility](NUMERICAL_COMPATIBILITY.md#root-response-normalization)
for the defect and its scope.

The repair changes only root-velocity fields. Eigenvalues, roots, eigenvector
tangents, normalization-scale velocities and isolation evidence are preserved.
Failed root entries remain absent. Original response artifacts and receipts
must be retained; corrections are separate artifacts, never replacements.
Repaired values remain computed evidence. The procedure neither certifies a
source nor recovers accuracy absent from its retained working precision.

Use the current v0.15.0 reader when reusing repaired receipts. It validates their
canonical source graph rather than expecting a shard adapter to contain local
key-based dependencies. The reader correction preserves all repaired receipt
identities and bytes; it does not require another repair or claim run. See
[published research record reuse](NUMERICAL_COMPATIBILITY.md#published-research-record-reuse).

## Offline response repair

Build on a platform supporting the HP prerequisites:

```sh
cargo build --release -p xc-spectral --features hp --example ccm_response_repair
```

Supply a JSON request on disk. Paths below are placeholders for explicitly
selected local shard snapshots, with their descriptors, active indexes,
manifests, transport encodings and immutable parts materialized:

```json
{
  "action": "response",
  "response": "/data/evidence/manifests/00/RESPONSE_DIGEST.json",
  "eigenpair": "/data/states/manifests/00/STATE_DIGEST.json",
  "output": "/data/repair/new-response",
  "read": {
    "scratch_directory": "/data/repair/scratch",
    "maximum_payload_bytes": 2000000000,
    "maximum_package_bytes": 2000000000
  }
}
```

```sh
target/release/examples/ccm_response_repair response-request.json
```

The output directory must not exist. Success writes `payload.json`, an immutable
transport under `parts/`, `draft.json`, and an old-to-new identity record in
`repair.json`. The package is decoded and verified before staging completes.
This command does not use the network or publish anything.

The reader verifies canonical metadata, active disposition, transport parts,
package contents and raw payload hashes. Repair checks the exact eigenpair
dependency, configuration, finite vectors, tangent norms and retained root
positions. Working precision is inherited from the source. Unsupported
semantics, missing inputs and mismatches return a nonzero exit code with a
reason; they do not trigger an automatic claim run. Schema-1 response payloads
are not supported. Already-v3 payloads must pass byte-exact retained replay.

## Repairing embedded capture measurements

A receipt request selects the original private receipt and the corrected
response directories produced above:

```json
{
  "action": "receipt",
  "receipt": "/data/evidence/manifests/00/RECEIPT_DIGEST.json",
  "repairs": [
    {
      "original": "/data/evidence/manifests/00/RESPONSE_DIGEST.json",
      "repaired": "/data/repair/new-response"
    }
  ],
  "output": "/data/repair/new-receipt",
  "read": {
    "scratch_directory": "/data/repair/scratch",
    "maximum_payload_bytes": 2000000000,
    "maximum_package_bytes": 2000000000
  }
}
```

Each embedded old measurement must match its authenticated response source.
The new receipt updates measurement evidence digests and both runtime and
canonical dependency identities. It retains the original plan, requested
diagnostics and all missing, blocked or failed outcomes. It does not turn an
incomplete attempt into a complete one. Receipts without affected measurements
need no repair. Capture receipts remain private.

## Explicit additive publication

Review the staged identity records, then make a separate publication request:

```json
{
  "action": "publish",
  "drafts": [
    "/data/repair/new-response/draft.json",
    "/data/repair/new-receipt/draft.json"
  ],
  "target": "private",
  "owner": "YOUR_GITHUB_ORGANIZATION",
  "journal": "/data/repair/publication-journal"
}
```

Run the same executable with this request file. Publication uses the existing
managed GitHub transaction and dependency verification with semantic replacement
disabled. Authentication and permissions are the same as for normal managed
publication. A batch must select exactly one visibility lane. A repair cannot
publish a private source publicly; public response repairs require a separate
`public` batch. Preserve the transaction journal for retries and audit.

New response identities become available to requests with the same retained
parents and configuration. This does not migrate unrelated historical sector
identities, change paper results, or refresh existing application-side reports.
Retain repair records alongside any analysis that used the original values.
