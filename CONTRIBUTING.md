# Contributing to Xcelerator Toolkit

The repository is source-available under the license in `LICENSE`. The project owner controls authorization to modify and redistribute the source. Do not assume an open-source contribution grant.

## Authorization before work

1. Open an issue describing the mathematical and software change.
2. Obtain written owner authorization for the exact scope before creating or distributing a modified fork or submitting code.
3. Record the authorization reference in the pull request. Authorization to discuss an idea is not authorization to redistribute a modified source tree.

## Authorship and provenance

Every contribution must identify its human authors and disclose generated-code assistance. The contributor must state whether each material implementation is original, independently implemented from a published algorithm, generated, adapted, or copied. Algorithm references, datasets, fixtures, and derived formulas must name their source and revision where available.

Record every new dependency, imported code fragment, external tool, native library, published algorithm, or external dataset through the owner-managed process summarized in [Third-Party Review](docs/v0.13.0/THIRD_PARTY_REVIEW.md). Copied or adapted code requires exact source-file provenance, applicable copyright and license text, compatibility analysis against the project's source-available license, and explicit owner approval. A citation alone is not redistribution permission.

By submitting an authorized contribution, each identified author represents that they have the right to submit the material under the owner-approved contribution terms and that the authorship/provenance declaration is complete. The owner may require a separate contributor agreement before acceptance.

## Engineering and review

- Keep changes focused and identify the public behavior they implement or change.
- Include normal, boundary, failure, and inconclusive tests appropriate to every changed public capability.
- Attach numerical provenance and trusted-reference or certification evidence where applicable.
- Do not weaken HP, determinism, cache validation, resource enforcement, or assurance guarantees for performance.
- Do not submit secrets, private cache locations, access tokens, signed URLs, or unpublished payloads.
- Run formatting, locked workspace tests, and warnings-as-errors Clippy. Run the corresponding HP checks on Linux/WSL when the change affects high-precision code; maintainers run the private release audit before acceptance.

At least one owner-authorized reviewer must examine mathematical semantics, tests, provenance, third-party/license review, assurance impact, and public/private data handling. The author may not self-approve the change. Review approval does not replace the owner's publication or redistribution authority.

## Acceptance record

The pull request must retain the authorization reference, author list, provenance declaration, algorithm/data references, third-party review updates, validation results, reviewer identity, and final owner decision. Public release validation scans tracked files for credential material; the owner separately retains internal review and release evidence. No hosted workflow is required.

The owner may reject or request removal of a contribution even after technical review when authorization, licensing, privacy, authorship, or research-integrity evidence is incomplete.
