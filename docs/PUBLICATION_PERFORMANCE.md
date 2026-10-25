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

## Family concurrency

`XC_PUBLICATION_FAMILY_WORKERS` controls concurrent destination-family work
within each dependency wave. The default is four. Set it to a positive decimal
integer, such as `1` for a constrained host. Each wave uses no more workers than
its runnable family count. Empty, zero, signed, nondecimal, or overflowing
values are rejected before workspace acquisition or remote I/O.

This setting is independent of Rayon and does not limit the total process
thread count: Git, transport monitoring, and the application can have additional
threads. Choose it alongside host CPU, memory, disk, and network limits.
Launch wrappers that sanitize the environment must explicitly preserve it.
Increasing it does not permit concurrent unguarded writes to one shard.
Dependency ordering, result ordering, destination verification, leases, and
compare-and-swap checks remain unchanged. It does not change artifact identity.

## Archive import and verification reuse

Family preparation references the original staged archive parts instead of
copying their payloads into a second family directory. Destination metadata is
still prepared separately. Before importing a part, the transport checks its
path, size, and SHA-256 digest. Symbolic links and Windows reparse points along
the staged path are refused. The imported Git object is then checked again, so
changing the source between verification and import cannot publish unchecked
bytes.

One managed invocation retains each destination shard's transport session and
immutable-object caches across its dependency waves. Authorization, registry
state, branch heads, destination selection, and leases are still checked for each
transaction; leases are released between transactions. Immutable object reuse
does not substitute for any mutable-state check. The two branch heads needed by
an atomic publication are queried together, and the push still uses exact
expected-head leases.

Retained sessions have separate Git object directories and share an aggregate
temporary-disk reservation ledger. Existing session files and concurrent
reservations count against that budget. Retention can therefore consume more
disk between waves; it does not raise or bypass the configured disk limit.
Normal completion and error cleanup remove these sessions while preserving
the durable staging directory and publication journal.

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
Each dependency-wave attempt also gets its own file. A retained transport closes
the previous attempt before opening the next one, including when the new file
cannot be created. An `attempt_closed` event only marks the telemetry handle
closing; it does not establish successful publication.
Scheduled payload bytes are not actual wire bytes. Git push time includes local
packing and remote acknowledgment; it is not a separate bandwidth measurement.
A missing final-success event means completion must be checked against the
canonical publication report. Operational logging failure does not bypass any
publication check or turn an unsuccessful transaction into success.

After acquiring its workspace, the managed publisher also writes
`journals/publication-operational-summary.json` below the staging directory on
return, including failure. This best-effort snapshot summarizes retained
attempts, phase observations, lock wait, metadata, verification/import, push,
deduplication counters and outcomes. It remains outside canonical inventories,
publication journals, receipts, semantic keys and payloads.

`latest_invocation.wall_seconds` comes from one elapsed clock covering the
publisher, including transport cleanup and excluding summary generation and
the caller's subsequent canonical report persistence. Each new family attempt
records `attempt_outcome` with its own whole-attempt clock and explicit result,
including early preparation failure and an all-present no-op. Legacy
`attempt_finished` can supply its recorded result, but its elapsed time covers
only the lease-holding portion. `attempt_closed` alone leaves outcome unknown.
Phase totals include nested operations and concurrent families; adding them
does not produce wall time. An observed executor success does not replace the
canonical durable publication record.

Byte counters retain their original layers: batch selection, local Git import
and destination verification can observe the same bytes repeatedly. Reused
bytes and scheduled bytes therefore remain separate counters. Neither is a
wire-byte measurement. Truncated, malformed, unreadable or oversized attempts
have unknown outcome. The scan is bounded to 1,024 attempts, 16,384 directory
entries, 16 MiB per attempt, 1 MiB per record and 64 MiB total input. A bounded
or incomplete scan is labeled. The raw JSONL files remain unchanged; summary
failure does not change the publication result.

## Repeated cumulative inventories

One finalization takes one cumulative draft snapshot. Inventory construction
validates and hashes each descriptor once per inventory pass, sharing those
results across its requested destinations. Every subsequent pass still fully
validates the descriptor; it retains no trust marker or validation memo.
Empty pending sets and entirely new sets also avoid an unnecessary
dependency-closure graph walk. Mixed sets still retain exact transitive parents,
including destination-neutral aliases.

These changes reduce local repeated work. They do not skip destination
scans or reuse mutable permissions, routes, heads, leases, capacity ledgers or
compare-and-swap state. Archive bytes retain their pre-import and imported-blob
checks. Same-process completion keys also bind assurance, its requirement and
evidence, so a changed assessment is reconsidered. No new upload or whole-run
speedup is established by these changes.

## Explicit retained-staging retry

Applications can call
`ManagedArtifactCacheSession::retry_retained_publication()` after authenticating
their original retained run. The session must explicitly enable author-mode
remote mutation, select a publication target, provide nonempty managed staging
and disable output-validation mode. It uses the ordinary publisher and durable
recovery journals without invoking a numerical producer or domain validator.
Every retained draft is reconsidered, including drafts marked complete earlier
in the same process. Normal finalization still refuses an apparent success when
the current process observed no artifacts. Neither path promotes assurance,
acquisition completeness or scientific coverage.

## Staged source routing and missing paths

Managed family publication routes each declared archive part directly to its
original staging root. Generated metadata is read from the family metadata
root. Duplicate archive paths must declare the same digest and size; conflicting
declarations are rejected. Routing does not authenticate bytes: the selected
path still receives the existing ordinary-file, no-link, containment, size and
SHA-256 checks, followed by verification of the imported Git blob. A damaged
selected copy fails instead of substituting an unrelated staging root.

Before checking a publication batch, the transport lists its paths in bounded
groups. Each Git listing has at most 32 paths and 24 KiB of path arguments;
outer prefetch groups have at most 256 paths. Present files retain their
ordinary digest checks. A successfully observed missing path can be reused
only for the exact repository, immutable commit and path. Its absence says
nothing about a later commit or the current branch head. At most 16,384 cache
entries admit missing-path observations; further misses are checked normally.
Session cleanup removes these observations together with other immutable
object metadata. Failed prefetch falls back to ordinary individual checks.

The completion registry remains process-local. Separate application processes
using one cumulative staging queue still reconsider its earlier descriptors
and destination state. A campaign can retain drafts and defer publication to
one final authenticated retained-staging invocation, provided it reports the
intermediate publication state as pending. This choice changes publication
timing; it does not change the final complete inventory or permit stale
permissions, routes, heads, dependency facts, leases or compare-and-swap state.

Verified loose-blob digest reuse requires a platform change stamp. On Unix this
binds device, inode, length, modification time and nanosecond change time; replacing
a same-length object and restoring its modification time still invalidates reuse.
Platforms without a reliable change stamp through the current adapter perform
full byte verification. This local corruption check is not a security boundary
against an administrator who can alter the process or its trusted storage.
