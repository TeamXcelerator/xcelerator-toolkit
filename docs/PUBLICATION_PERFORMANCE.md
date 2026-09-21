# Publication performance

Managed publication disables Git delta searches when pushing content-addressed
artifact archives. The archives have already been compressed and verified.
Exact Git object reuse, SHA-256 checks, artifact identities, batch limits,
destination checks and atomic publication remain in place. This setting is
passed only to the publication Git command; it does not change a user's Git
configuration or recompress existing artifact archives.

## Measured scope

A local Windows test with Git 2.50.1 packed eight retained 90 MiB archive parts
(720 MiB total). Two trials reversed the order of the baseline and changed
policy:

| Git packing policy | First trial | Second trial | Pack bytes |
|---|---:|---:|---:|
| Default delta search | 109.567 s | 100.892 s | 755,205,160 |
| Delta search disabled | 13.341 s | 13.455 s | 755,205,160 |

The four packs have the same SHA-256. These measurements establish a 7.5-8.2x
reduction in local packing time for this retained sample. They do not measure
GitHub upload time, network throughput, receiver processing, or complete claim
runtime. Other data and machines can behave differently.

Git documents the [delta-search window and packing options](https://git-scm.com/docs/git-config#Documentation/git-config.txt-packwindow).
This release leaves the Git compression level unchanged.

## Visible progress

The publisher reports destination-metadata preparation, batch number, checked
file count, existing bytes reused, and bytes scheduled for new or updated
files. Scheduled bytes are not a measurement of bytes sent over the network.

Git pushes report start and elapsed completion time. Long pushes, fetches and
reference queries emit a heartbeat every 30 seconds. For pushes, the heartbeat
includes the latest available counting, compression or writing phase, object
counts, and numerical transfer amount/rate. A completed writing phase followed
by a wait is distinguished from ongoing object writing. The heartbeat reports
the last Git progress observation; it is not proof that the network advanced
during that interval. Arbitrary remote output, command arguments and credentials
are not echoed as live progress.

## Retained results and recovery

This is a transport change. Existing numerical results, semantic identities,
archives and historical publication records remain valid under their original
validation rules. No cache flush, artifact repair, matrix rebuild or root
recalculation is required by this amendment.

Keep the run journal, production queue, staging directory and publication
reports if publication is interrupted. The managed publisher checks existing
destination content and publishes the missing pieces when invoked again on
the retained drafts. Its existing staging lock prevents two finalizers from
using the same journal concurrently. A resumed publication does not rewrite
an earlier numerical claim assessment.

A running executable retains its original publication implementation until it
exits. Updating a checkout does not accelerate an already running push.

## Archive import and verification reuse

v0.15.1 imports verified canonical `objects/sha256/*.part` archive pieces with
command-local `core.looseCompression=0`. Metadata retains ordinary compression.
This changes Git's local storage work, not blob identity or archive bytes. It is
separate from the existing no-delta push policy; global Git settings are untouched.
A synthetic 32 MiB compressed archive imported in 0.70-0.72 seconds with the
default policy and 0.25 seconds without loose-object compression in two reversed
Windows trials. All Git object IDs matched. This measures local import only,
not live GitHub throughput or complete campaign runtime.

Verified loose blobs have a bounded process-local SHA-256 cache keyed by exact
session, Git object ID and size. Unchanged file size and modification time are
required for reuse. Changed/missing storage is rechecked; packed objects retain
the streaming path. Session cleanup clears entries. The first read still checks
all bytes, size and resource limits. This is not a persistent trust certificate.

## Durable operational reports

Family publication writes append-only attempts under
`family-batches/<family>/<destination>/publication-metrics/attempt-*.jsonl` in the
journal directory. Each record has a schema version, phase, elapsed seconds and
phase-specific details. Timings are excluded from scientific keys and payloads.

Records cover preparation, lock waiting, metadata, staged verification, Git
import/push, destination verification reuse, batch completion and pending batches.
A resumed invocation creates a new attempt file, preserving previous failures.
Scheduled payload bytes are not actual wire bytes. Git push time includes local
packing and remote acknowledgment; it is not a separate bandwidth measurement.
A missing final-success event means completion must be checked against the
canonical publication report. Operational logging failure does not bypass any
publication check or turn an unsuccessful transaction into success.

## Retained archive measurement

A verified 90 MiB artifact part was imported into fresh local Git repositories
and then packed with the existing no-delta policy. Two trials reversed the order:

| Trial | Default import + pack | No loose compression + pack | Improvement |
|---|---:|---:|---:|
| First | 6.121 s | 2.967 s | 2.06x |
| Reversed | 3.911 s | 2.779 s | 1.41x |

All four resulting packs have identical bytes and SHA-256. Warm filesystem state
and concurrent local work can affect timings. This measurement includes import
and local packing, not network transfer, remote processing or complete claims.
The [raw record](validation/v0.15.1-publication-import.json) retains individual
phase times, byte counts and hashes. `tools/benchmark_publication_import.py`
replays this comparison on a caller-selected verified part without a remote.


Verified loose-blob digest reuse requires a platform change stamp. On Unix this
binds device, inode, length, modification time and nanosecond change time; replacing
a same-length object and restoring its modification time still invalidates reuse.
Platforms without a reliable change stamp through the current adapter perform
full byte verification. This local corruption check is not a security boundary
against an administrator who can alter the process or its trusted storage.
