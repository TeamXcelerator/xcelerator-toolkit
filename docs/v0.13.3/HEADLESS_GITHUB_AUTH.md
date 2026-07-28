# Headless GitHub authentication (v0.13.3)

The toolkit supports GitHub personal access tokens on ephemeral Linux systems, including Vast.ai instances. Git Credential Manager is not required. Both the permission probe and direct no-checkout Git transport obtain credentials through Git's standard credential-provider protocol.

## Interactive setup

Configure Git's in-memory credential helper, then read the token without terminal echo:

```bash
git config --global credential.helper 'cache --timeout=604800'
read -rsp "GitHub PAT: " XC_PAT
echo
printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' \
  'YOUR_GITHUB_USERNAME' "$XC_PAT" | git credential approve
unset XC_PAT
```

The seven-day timeout allows a long computation to retain publication authority through its final phase. Select a shorter timeout when appropriate for the job.

## Secret-file or platform-secret setup

Read a protected platform secret without placing it in a command argument:

```bash
git config --global credential.helper 'cache --timeout=604800'
IFS= read -r XC_PAT < /run/secrets/github_pat
printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n\n' \
  'YOUR_GITHUB_USERNAME' "$XC_PAT" | git credential approve
unset XC_PAT
```

The secret file should be readable only by the job user. Delete or unmount the platform secret when job policy requires it. Do not export the PAT in shell tracing, embed it in a Git URL, commit it to a configuration file, or include it in a run log.

## Required access

Use a fine-grained PAT scoped to the `TeamXcelerator` organization and only the registry and shard repositories required by the run. Publishing needs repository contents read/write access; routing and permission preflight also need repository metadata read access. Private publication requires access to the private registry and selected private shards. The toolkit automatically creates and updates the private shard's isolated `xcelerator-coordination` branch with the same contents permission; no separate locking repository, workflow permission, or additional secret is required.

Authenticated private cache reads use the same credential provider. Set
`XC_CACHE_REMOTE=private` for a private-only lookup ordered as workstation
cache then authenticated private registry/shard; public fallback is not
consulted. `XC_CACHE_REMOTE=private_public` selects workstation, private, then
public. The default remains unauthenticated public retrieval, and `none`
disables remote reads.

Credentials alone never enable publication. The author must still select an explicit target and enable execution. For example, a computed dual-target run uses:

```bash
export XC_RUN_PROFILE=author
unset XC_ASSURANCE
export XC_PUBLISH_TARGET=both
export XC_PUBLISH_EXECUTE=true
export XC_CACHE_REPOSITORY_OWNER=TeamXcelerator
```

### Forced replacement

An author can deliberately bypass every cache overlay, recompute the complete artifact dependency graph, and replace the current public, private, or dual-target result:

```bash
export XC_RUN_PROFILE=author
export XC_CACHE_MODE=refresh
export XC_CACHE_REMOTE=none
export XC_PUBLISH_REPLACE=true
export XC_PUBLISH_TARGET=both        # public, private, or both
export XC_PUBLISH_EXECUTE=true
export XC_CACHE_REPOSITORY_OWNER=TeamXcelerator
```

`XC_CACHE_MODE=refresh` prevents both workstation and remote reuse. `XC_PUBLISH_REPLACE=true` removes prior entries for the same semantic identities from the current shard indexes, makes the fresh manifests uniquely discoverable, audits the exact resulting shard revision, and removes unreferenced manifests, encodings, and payload objects from the current branch. Shared objects and audit receipts are retained. Git history is not rewritten, so removed historical bytes continue to count toward repository capacity. Replacement is rejected unless author mode, refresh mode, an explicit target, and remote execution are all enabled.

Before mutation, the toolkit resolves `/user`, verifies effective write permission for every target, applies assurance and public-sanitization policy, checks capacity, and records only redacted principal and permission evidence. The PAT is not written to cache artifacts, staging metadata, journals, receipts, reports, or published repositories.

To remove the in-memory credential before the instance is destroyed:

```bash
git credential-cache exit
```
