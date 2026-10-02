# Publication performance

Managed publication disables Git delta searches when pushing content-addressed
artifact archives. The archives have already been compressed and verified.
Exact Git object reuse, SHA-256 checks, artifact identities, batch limits,
destination checks and atomic publication remain in place. This setting is
passed only to the publication Git command; it does not change a user's Git
configuration or recompress existing artifact archives.

## Scope

Delta search is disabled only for compressed, verified artifact archives. The
setting reduces local packing work; it makes no claim about upload time, network
throughput, receiver processing, or complete run time.

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
Git object IDs are unchanged. The change affects local import only, not network
throughput or complete campaign runtime.

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

Verified loose-blob digest reuse requires a platform change stamp. On Unix this
binds device, inode, length, modification time and nanosecond change time; replacing
a same-length object and restoring its modification time still invalidates reuse.
Platforms without a reliable change stamp through the current adapter perform
full byte verification. This local corruption check is not a security boundary
against an administrator who can alter the process or its trusted storage.
