# Independent PARI eigenvalue reference

Generated locally with PARI/GP 2.15.4 at 2,000 decimal digits from the exact rational matrices in `generate_eigen_reference.gp`. The original generator computes each exact characteristic polynomial and asks PARI for its real roots. It does not call the toolkit or MPFR eigensolver.

Regenerate from this directory with `gp -q generate_eigen_reference.gp > eigen_reference.json`, then run `cargo test -p xc-numerics --features hp --test eigen_reference -- --nocapture` from the workspace root. The test requires the committed fixture and checks at least 500 matching decimal digits at 3,338-bit toolkit precision. Four matrices supply 16 eigenvalues; these finite cases are not a universal numerical guarantee.

The reference fixture SHA-256 for this generation is `491cd23a2f9045eeb462d0c275dfdd883439684be1a39afd84fee3b4dd936797`. PARI documents the relative-accuracy contract and its polynomial real-root algorithm in [polrootsreal](https://pari.math.u-bordeaux.fr/dochtml/html/Polynomials_and_power_series.html#polrootsreal). The matrices and generator are original AI-assisted validation work; no PARI implementation source or external research dataset is copied. PARI is a separately installed development oracle, not a new toolkit runtime dependency.
