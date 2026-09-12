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
