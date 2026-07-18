# Security and Private Research Data

Report suspected credential exposure privately to the project owner at the contact address in `README.md`.

- Never commit personal access tokens, private repository credentials, signed URLs, or cloud keys.
- Cache manifests and certificate bundles must not contain secrets.
- Private cache repositories remain private until an explicit reviewed promotion.
- Publication tooling must default to refusing unknown visibility or quality states.
- Generated archives should be scanned before release.

Release validation scans every tracked file for private-key and provider-token
signatures. Canonical cache records, saved-result provenance, persisted research
results, and certificate bundles also pass through the shared secret-free JSON
validator. Rejections report only the field path or marker class and never echo
the suspected credential value.
