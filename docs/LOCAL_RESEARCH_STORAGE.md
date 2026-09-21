# Local research storage and retention

Canonical artifacts, publication receipts and historical run evidence are durable
research records. The local stores below are operational aids with separate retention.
Nothing is deleted automatically. Run cleanup only while all jobs using the selected
store are stopped; an active writer must never race an operator's cleanup.

| Store | Retention policy | Effect of deleting old entries |
|---|---|---|
| Managed-cache `research-checkpoints/<identity>/` | Keep during interrupted/restartable work. After successful artifact retention, remove whole obsolete identity directories when space is needed. | Recomputes diagnostic stages; primary artifacts remain available. |
| Family staging `publication-metrics/attempt-*.jsonl` (or configured metrics directory) | Keep active and incomplete attempts. Archive completed attempts with the run journal; 30 days is a suggested local working retention period. | Removes local telemetry only; retain canonical execution reports and receipts. |
| Managed-cache `research-cohorts/*.json` | Keep registrations needed by the intended comparison campaign; archive completed campaign registrations outside the discovery directory. | Changes future automatic comparison candidates; never alters retained comparisons or primary artifacts. |
| Managed-cache `research-summaries/` and derived SQLite databases | Rebuildable navigation; retain useful summaries with shared research handoffs. | Loses local navigation only; canonical measurements remain intact. |

Paths can be overridden by the documented `XC_RESEARCH_*` and publication metrics
settings. Confirm the actual configured path and preserve logs needed for failed jobs.
Per-checkpoint and basis limits bound an operation, not cumulative store size.

Cohort discovery selects at most 512 registration files in deterministic digest-name
order, retaining a notice when additional entries were not scanned. It no longer
fails merely because a long-lived directory reached that count. Use a dedicated
`XC_RESEARCH_COHORT_DIR` containing the desired frozen subset for a reproducible
comparison campaign. Metadata traversal and decoded-comparison bounds still apply.
A bounded discovery result is not a claim to have compared every stored state.

Checkpoints are trusted local execution data with SHA-256 integrity checks, not
cryptographic authentication or portable proof certificates. Their identity includes
the Toolkit version, numerical workspace source fingerprint, build target and Arb
feature mode. Changed source invalidates checkpoints even when a release is amended
without changing its version. Finite source certificates are replayed on each new
enclosure calculation; a saved Boolean cannot substitute for verification.
