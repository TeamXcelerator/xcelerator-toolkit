# Large atom tables, finite cutoff studies, and research queries

Ultra requests the weighted-tail, band-reconstruction and tail-model diagnostics.
Their optional atom policy adds the measurements below when the corresponding
inputs are supplied. It does not generate zeta ordinates or assume an infinite
tail. External campaign evaluator implementations are supplied independently.

## Prepare authenticated input chunks

Use JSONL with one atom per line. Numerical values are decimal **strings**.
A weighted atom has `ordinal`, `coordinate`, `weight`, `family` (`zero` or
`lattice`) and `partition`. A band atom has `coordinate`, `signed_weight` and
`family`. Coordinates and weights must follow the declared model convention.

```sh
python3 tools/prepare_atom_chunks.py atoms.jsonl --kind band --output atom-inputs
```

The command creates a new directory with ordered JSONL chunks and an
`atom-analysis.json` policy containing every chunk's SHA-256, bytes and row count.
It refuses an existing output directory. Files are data-only; no expressions run.

Copy that policy into `completion.atom_analysis` in a reference-preparation file,
or `run_once.completion.atom_analysis` in a state-bound external-input file.
Keep the enclosing input JSON in the same directory as its chunks. Chunk paths
must be relative descendants, with no `..`, absolute paths or escaping symlinks.
For band chunks, also supply `completion.band` with an empty `atoms` array and
the desired degree, coordinate, definition digest, coverage and hypotheses.
For weighted chunks, keep the inline `atoms`/`weighted_atoms` array empty.
Do not combine inline and chunked atoms for the same table.

The declared `maximum_atoms` and `maximum_input_bytes` replace the old count
ceiling only for this explicit policy. Each chunk is authenticated before parsing
and its decoded-byte stream is checked again to detect concurrent changes. Rows
are decoded incrementally; the typed atom arrays remain resident. Input memory,
recurrence memory, basis disk, row and output budgets still apply. Larger declared
limits are permission to consume those resources, not a promise they exist.

For two tables, combine both chunk lists under one policy, choose a row cap large
enough for either table, and set the byte cap to cover both. Preserve chunk order.
The retained external-source artifact contains expanded numerical inputs and the
chunk descriptors, so its meaning does not depend on later access to local files.

## New measurements in existing artifact kinds

**Band coverage:** largest supplied zero coordinate, largest model band root,
number of supplied zero atoms beyond that root, and a finite extent comparison.
These fields require atoms explicitly labeled `zero` in the same coordinate.
They do not certify the missing tail, even when the supplied extent covers the band.

**Fixed cutoff ladders:** `atom_analysis.cutoffs` is a strictly increasing list
of at most 32 decimal strings, frozen in the external input identity. Each cutoff
retains zero-family atoms at or below that coordinate and all supplied lattice/
other-family atoms. Band rows retain first/last model roots, inverse moments and
recurrence positivity margins. The degree and any borrowed energy stay fixed.
Insufficient support, failed positivity and unresolved rows remain in the report.

For model-energy ladders, additionally supply `atom_analysis.tail_recipe` with
`basis_polynomials`, `tail_correction` and `hypotheses`, using the same format as
the reference-preparation tail recipe. The fixed basis, lattice and optional tail
correction are reused at each cutoff. A missing tail correction means a finite-only
model with an unknown omitted remainder. Energy, residual and energy-split fields
are retained for each successful finite solve, along with signed and relative
changes between adjacent successful cutoffs. If an explicit primary tail form is
also supplied, its basis is not assumed identical to the separate ladder recipe. These solves have additional cost;
they do not rebuild the primary CCM matrix or eigenstate.

**Signed per-ordinal atom kernels:** `atom_analysis.evaluations` contains
`ordinal`, `coordinate`, `label` and optional `exclude` keys. Each exclusion is
an exact `(family, partition, ordinal)` atom key. For each evaluation and atom
partition, the report retains

```
K_m(z) = sum over included atoms of w / (x-z)^m,  m = 1, 2, 3
```

It also records absolute sums, cancellation digits when defined, included/excluded
counts and the nearest included atom. An absent exclusion key or a coincident/
precision-limited included atom withholds the affected sum. Exclusions are never
inferred from proximity. Evaluation ordinals are caller declarations; they do not
identify retained roots with zeta zeros. These kernels support a separately
specified displacement identity, but do not themselves assert that identity or
its normalization, one-sided/two-sided factor, completeness or RH premises.

## Recurrence resources and restart

`XC_RESEARCH_WORKING_BYTES` bounds estimated resident numerical work. Band basis
vectors are saved in independently sealed 4,096-entry blocks, with a bounded
memory cache. `XC_RESEARCH_BASIS_BYTES` defaults to 8 GiB and limits estimated
basis disk use. `XC_RESEARCH_CHECKPOINT_BYTES` remains a per-checkpoint limit.
Use `XC_RESEARCH_CHECKPOINT_DIR` (or the managed-cache default) for durable restart;
without one, a temporary basis workspace is removed when the operation finishes.

A completed recurrence degree saves its coefficients, row measurements and the
next basis vector. Restart validates the saved state and basis blocks before
continuing. Corrupted or incompatible data causes recomputation. Signed
contractions use fixed 512-entry blocks and ordered reduction, independent of
worker count. Arithmetic scratch is reused. Checkpoint decoding hashes buffered
reads in one pass; validation does not replay numerical contractions.

Increase disk or memory limits only after checking available resources. Basis
storage and cutoff solves can still be substantial at large degree and precision.

## Small searchable exports

Extended capture automatically writes compact scalar summaries under the managed
cache's `research-summaries` directory; `XC_RESEARCH_SUMMARY_DIR` selects another
directory. Full arrays remain in canonical artifacts. Summaries retain exact
scalar strings, per-row outcomes, conventions and source-manifest identities.
Summary failure does not discard the canonical result. These derived exports can
be copied with a research handoff without transferring large matrix payloads.

```sh
python3 tools/research_query.py build /path/to/research-summaries --output research.sqlite
python3 tools/research_query.py query research.sqlite --C 250 --N 750 --ordinal 19
python3 tools/research_query.py query research.sqlite --observable model_energy
python3 tools/research_query.py query research.sqlite --inventory
```

`C` means `lambda_squared`; `P` means source precision in bits. Ordinal scope,
outcome and normalization remain in each result. Units absent from the source
are labeled undeclared, not invented. Malformed, oversized and unrecognized files
remain in the inventory. An index is navigation, not numerical verification.

```sh
python3 tools/publication_summary.py /path/to/publication-journal --output timings.json
```

The publication summary separates attempts and phases, retains failures and
truncated logs, and reports the latest pending-work counters. Timings overlap;
summing phases does not give wall time. Byte counters are not wire throughput.
The canonical publication report remains authoritative for completion.

## Compatibility and backfill

The thirty managed kinds and Ultra v6 remain. Weighted-tail, band and tail-model
producers now use extended semantics v4; resolution diagnostics retain v3.
Historical payloads and receipts are preserved. Backfill can add these children
from retained primary sources and authenticated external inputs. Old receipts
are not relabeled, and missing atoms are not synthesized.
