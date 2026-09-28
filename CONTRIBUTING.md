# Contributing to Xcelerator Toolkit

The repository is source-available under the license in `LICENSE`. The project owner controls authorization to modify and redistribute the source. Do not assume an open-source contribution grant.

## Authorization before work

1. Open an issue describing the mathematical and software change.
2. Obtain written owner authorization for the exact scope before creating or distributing a modified fork or submitting code.
3. Record the authorization reference in the pull request. Authorization to discuss an idea is not authorization to redistribute a modified source tree.

## Authorship and provenance

Every contribution must identify its human authors and disclose generated-code assistance. The contributor must state whether each material implementation is original, independently implemented from a published algorithm, generated, adapted, or copied. Algorithm references, datasets, fixtures, and derived formulas must name their source and revision where available.

Record every new dependency, imported code fragment, external tool, native library, published algorithm, or external dataset through the owner-managed process summarized in [Third-Party Review](docs/THIRD_PARTY_REVIEW.md). Copied or adapted code requires exact source-file provenance, applicable copyright and license text, compatibility analysis against the project's source-available license, and explicit owner approval. A citation alone is not redistribution permission.

By submitting an authorized contribution, each identified author represents that they have the right to submit the material under the owner-approved contribution terms and that the authorship/provenance declaration is complete. The owner may require a separate contributor agreement before acceptance.

## Engineering and review

- Keep changes focused and identify the public behavior they implement or change.
- Include normal, boundary, failure, and inconclusive tests appropriate to every changed public capability.
- Attach numerical provenance and trusted-reference or certification evidence where applicable.
- Do not weaken HP, determinism, cache validation, resource enforcement, or assurance guarantees for performance.
- Do not submit secrets, private cache locations, access tokens, signed URLs, or unpublished payloads.
- Run formatting, locked workspace tests, and warnings-as-errors Clippy. Run the corresponding HP checks on Linux/WSL when the change affects high-precision code; maintainers run the private release audit before acceptance.

At least one owner-authorized reviewer must examine mathematical semantics, tests, provenance, third-party/license review, assurance impact, and public/private data handling. The author may not self-approve the change. Review approval does not replace the owner's publication or redistribution authority.

## Build storage

The checkout's `.cargo/config.toml` disables incremental compilation and full
debug information for `dev` and `test` builds. It keeps the normal debug assertions
and overflow checks; release/research profiles retain their existing settings.
This trades faster incremental rebuilds and rich debugger information for lower
disk use. When debugging requires it, set `CARGO_PROFILE_DEV_DEBUG=1` (or `2` for
full information) for that invocation; use `CARGO_PROFILE_TEST_DEBUG` for tests.

Run Cargo from this checkout or a directory below it. The toolkit and nested
consumer workspaces then share `target/`, avoiding a second dependency build tree.
`CARGO_TARGET_DIR` or `--target-dir` can override this; keep one reusable directory
per platform instead of creating a new directory for every session. Cargo discovers
configuration from the working directory, not a separately supplied manifest path.

Stable Cargo does not automatically evict old build output or enforce a build-cache
size cap. Use `cargo check` for routine editing, build only the needed package/target,
and periodically use `cargo clean --profile dev` when builds are idle to clear ordinary
development/test output. Avoid an unqualified `cargo clean` in this checkout: the
historical `target/` tree also contains retained research and Git transport caches.
The profile-specific command does not clear custom research/release profiles.
See [Cargo profiles](https://doc.rust-lang.org/cargo/reference/profiles.html) and
[Cargo cache behavior](https://doc.rust-lang.org/cargo/reference/config.html#global-caches).

Before and after a large audit or publication session, measure checkout storage
with `powershell -NoProfile -ExecutionPolicy RemoteSigned -File tools/manage_disk_space.ps1`. Reuse build
directories, and include disk growth and cleanup in the session's completion
record. A session is not operationally complete while its disposable transport
state is left behind without an explicit reason.

Managed publication already removes temporary Git transport repositories on
completion or failure and recovers leftovers when the same journal is reused.
A terminated process can leave an abandoned journal that is never reused.
On Windows, remove explicitly selected, idle journals' transport directories with:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File tools/manage_disk_space.ps1 -CleanTransport -JournalRoot target/old-run
```

Stop publishers using those journals, including any WSL publisher, first. The
maintenance command defaults to reporting only. Cleanup requires explicit paths
inside this checkout's `target/`, at least seven days without modifications,
the publication lock, no links or tracked files, and recognizable temporary bare
repositories without local branches or tags. It rechecks the inventory before
removal, retains sibling publication records and artifact staging, and writes a
receipt with removed bytes and drive free space under `target/disk-maintenance/`.
It does not clean compiler output, research evidence, or the toolkit's `.git/`.
Run `python -m unittest discover -s tools -p test_manage_disk_space.py` after
changing the maintenance command.

## Acceptance record

The pull request must retain the authorization reference, author list, provenance declaration, algorithm/data references, third-party review updates, validation results, reviewer identity, and final owner decision. Public release validation scans tracked files for credential material; the owner separately retains internal review and release evidence. No hosted workflow is required.

The owner may reject or request removal of a contribution even after technical review when authorization, licensing, privacy, authorship, or research-integrity evidence is incomplete.
