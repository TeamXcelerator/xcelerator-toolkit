# Compact analytic sector-gap certificates

New optional analytic sector-gap certificates can retain their exact results
and reconstruction recipe without permanently storing the dense interval
matrices or LDLT pivot traces. This is an explicit API in
`xc_spectral::ccm::sector_gap_certificate::compact`; it requires `hp,arb` for
production and independent verification. Certification remains optional.

## Stored information

The compact record retains integer C, N, assembly and proof precisions,
geometric truncation, arithmetic backend, assembly/inertia/parity semantics,
producer source/build identity, guide artifact identities, raw/component and
canonical-matrix hashes, three exact eigenvalue brackets, the six boundary
counts, full-matrix inertia counts, exact gap bounds, finite conclusions and
the explicit analytic parity premise. All numerical endpoints are exact
rationals. There is no matrix or pivot array in its serialized schema.

Storage grows with the lengths of this fixed set of scalars, rather than with
the number of matrix entries. Computation still needs the analytic matrices;
this change is a retention saving, not a claim of lower peak memory or faster
initial certification. Guide dependency artifacts remain referenced normally.

## APIs and verification

- `CompactCcmSectorGapCertificate::from_portable` projects the exact assertions
  from a structurally valid schema-4 full certificate. Conversion alone does
  not independently verify its mathematics.
- `compact::cache::resolve_or_compute` is the managed production API. It builds
  and verifies a full proof temporarily, then stores and stages only the
  compact record. It takes the retained even/odd/gap guide manifests and exact
  producer source identity explicitly.
- `verify_compact_ccm_sector_gap_certificate` always freshly reconstructs the
  analytic interval matrix, checks the recorded backend and source hashes,
  and independently proves all the full and selected inertia counts. It checks
  conclusions against exact arithmetic on the verified bounds. An unresolved
  count, unsupported semantics or a hash mismatch fails verification.
- `reconstruct_verified_compact_sector_gap` performs that same verification
  and returns an immutable handle to the compact record and reconstructed
  canonical matrix, for later directed-residual consumers.

The verifier tests the recorded bracket boundaries directly, avoiding a repeat
of guide discovery and bracket search. It still performs interval assembly and
the required dense inertia calculations. It can be substantially more
expensive than reading a research result.

The managed API distinguishes ordinary computed reuse from independent proof
replay: authenticated computed reuse checks the compact record's structure and
request identity; certified or reference-verification reuse performs fresh
mathematical replay. A fresh production has already passed its analytic proof.

## Source reconstruction and identity

`ReconstructionSourceIdentity::current(revision_or_snapshot)` records the
compiled source/build closure digest alongside a caller-supplied recoverable
revision or snapshot identity. The caller must retain that source, lockfile and
native dependency environment. For a dirty checkout, a base Git revision alone
is insufficient; retain the patch and new files or a complete source snapshot.
The build digest covers source bytes, Cargo.lock and target; it is not a digest
of every external native library or an attestation of an executable.

Source identity is producer provenance. Mathematical verification independently
reconstructs the analytic matrix and matches its hashes; it does not trust a
producer's source label or saved success flag. A compatible implementation can
verify the record if it reconstructs the exact recorded interval data. A changed
backend producing different valid enclosures does not silently replace them.
The producer's source identity is outside the mathematical cache key: a
compatible build reuses the exact original record and its historical provenance.
Explicit regeneration under another source snapshot can change provenance bytes;
it does not rewrite an existing record. Mathematical changes require updated
algorithm semantics.

The managed representation uses the registered `ccm_sector_gap_certificate`
family with a separate logical key, explicit `compact_analytic_replay_v1`
representation and mathematical semantics. Its payload identity is its own;
it is not the identity of the original full certificate. The old full format
and readers remain available. Minimum reader version is 0.17.1; exact builds
must also contain this new API. No archive bytes are promised to match legacy
full-certificate archives.

## Optional claim capture and proof consumers

`CcmResearchCaptureOptions::with_compact_sector_gap_certification(options, source)`
explicitly selects compact retention. The existing full builder keeps its full
behavior, and neither `maximum` nor a storage-format value alone enables a
certificate. A retained run fixes its certificate options, representation and
provenance on its first certificate request and rejects conflicting later requests.

Capture receipts bind the actual compact payload and manifest. The additive
`SectorGapCertificateRef` APIs admit either representation; compact source-error,
transform and finite-root-budget consumers reconstruct and verify the
analytic matrix before running the shared directed residual arithmetic. Within
one retained run they share an immutable verified handle, bound to the entire
exact compact payload (including its matrix hashes and proof/assembly semantics)
and the current MPFR exponent range. The run retains at most one such matrix,
charged to its available memory budget; ordinary memory-pressure eviction can
release it. Every returned handle, including a public or uncached handle, owns
its resident matrix reservation for the borrower's lifetime. Reconstruction
admits that capacity-based resident allowance before releasing its proof
workspace; insufficient transition headroom returns an error. No success
flag is serialized or trusted on a later run. Public compact verification and
reconstruction, and independent compact root-budget report verification, always
replay freshly, even inside this scope. Full entry points and full request
identities remain available. Compact root-budget
reports and capture requests distinguish their representation explicitly. A
compact manifest cannot be paired with a full certificate object.

Finite-root-budget admission uses checked dimension/index arithmetic and explicit
run-owned memory reservations instead of a fixed N=512 exclusion. Root vectors,
directed row temporaries and exact report scalars have a separate allowance from
analytic assembly, parity projection, record conversion and dense inertia work.
The proof allowance also includes sieve and geometric-correction storage and
uses the maximum recorded proof precision. A standalone call uses the capture
working-budget policy; an enclosing run's budget is authoritative. Overflow,
insufficient budget, cancellation, unresolved proofs and identity mismatches
fail closed. These are conservative allocation allowances, not measured peak RSS
or an OS memory limit. Admission never changes summation order, precision,
certificate bytes or report identities. The existing 256-window and precision
limits remain; a partial census at larger N cannot establish global ordinals.

Paper 1's optional runtime selector is `--sector-gap-certificate compact|full`.
It remains off when omitted, including Ultra; the legacy boolean requests full
storage. Compact support requires the corresponding Paper 1 build feature and
a Toolkit build containing these APIs. The locally qualified integration uses
a retained source snapshot; the published Paper 1 dependency pin is unchanged.

The compact result proves only the recorded finite analytic claims under their
stated parity premise. It supplies no continuum gap, omitted-tail bound,
convergence theorem or RH conclusion. Existing claim artifacts are unchanged.

## Local qualification

The optional claim-consumer integration passed 27 distinct focused native
Windows Toolkit tests, two Paper 1 selector/default tests, and scoped strict
Rust Clippy. A real Paper 1 C=13, N=4, 197-bit executable run completed with
certification off, compact, and full. All primary mathematical bytes matched
after normalizing only the top-level elapsed-time field. The certificate
payloads measured 4,579/83,890 raw bytes and 2,013/9,231 encoded bytes for
compact/full. These are small-fixture measurements, not large-N predictions.
The source snapshot, exact build and qualification records are retained with
the local integration. Its release/pin update is not yet published.

Native Windows qualification passed all 24 focused certificate tests and scoped
strict library Clippy with `hp,arb`. Tests include fresh analytic reconstruction,
exact full-versus-compact matrix/assertion agreement, tampered assertion rejection,
compact-only managed storage, certified replay, and exact-byte cache reuse across
compatible builds. The final C=5, N=2, P=256-bit fixture measured 44,365 versus
4,752 raw JSON bytes and 6,169 versus 2,039 bytes in identical-option Deflate ZIPs.
The compressed reduction is 66.95% for this small fixture; this is not a large-N
storage estimate, peak-memory measurement, or whole-claim timing result.
