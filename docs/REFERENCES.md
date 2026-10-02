# References

- Xcelerator Toolkit public repository and source-available license.
- Connes, Consani, Moscovici, *Zeta Spectral Triples*, arXiv:2511.22755.
- Sliwinski, *High-Performance Computation of M_k*.
- Groskin, *A finite Guinand-Weil dictionary and archimedean tail order for the truncated Weil quadratic form*, arXiv:2607.02828.
- Groskin, *High-Precision Approximation of Riemann Zeros via the Truncated Weil Form*, arXiv:2605.20224.
- Suzuki, *Weil's Quadratic Form via the Screw Function*, arXiv:2606.09096.
- GitHub documentation: repository limits, large files, Git data APIs, authentication, and push behavior.
- Git documentation: partial fetch filters, object creation, tree and commit construction, and fast-forward push rules.
- The Cargo Book: manifest format and publishing on crates.io.
- Zenodo documentation: GitHub software integration, `CITATION.cff`, `.zenodo.json`, and DOI records.
- Team Xcelerator cache operating constraints: every Git-managed file remains strictly below 100 MB; deterministic byte-split ZIP/ZIP64 parts default to 90 MiB; each commit/push batch introduces no more than 1,000,000,000 payload bytes; and a shard rolls over before projected reachable repository payload exceeds 100,000,000,000 bytes, with history and reserve tracked separately.
