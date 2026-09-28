# Prolate numerical model

The ordinary `compute_k_lambda_f64`, `hp::compute_k_lambda`, and `hp::compute_k_lambda_via_cache` routes approximate the bounded prolate eigenfunctions at the singular endpoints. They use an even Legendre Galerkin expansion, not a Dirichlet condition at the endpoints.

With `x = lambda*t` and `c = 2*pi*lambda^2`, the operator is

`-d/dt((1-t^2) d/dt) + c^2*t^2`, for `-1 <= t <= 1`.

For the orthonormal basis `phi_j(t) = sqrt((2j+1)/2)*P_j(t)`, multiplication by `t` has coefficients `a_j = (j+1)/sqrt((2j+1)(2j+3))`. Its square gives the symmetric even block:

- diagonal: `j(j+1) + c^2*(a_j^2 + a_(j-1)^2)`, with `a_-1 = 0`;
- coupling from degree `j` to `j+2`: `c^2*a_j*a_(j+1)`.

The selected even-block indices are 0 and 2, corresponding to full prolate modes `h_0` and `h_4`. The polynomial is evaluated directly; no spatial interpolation or zero endpoint value is imposed on these modes. See [DLMF 30.8](https://dlmf.nist.gov/30.8) for the spheroidal Legendre expansion. The symmetric coefficients above follow independently by applying the Legendre multiplication recurrence twice.

The coefficient vectors have unit Euclidean norm, so dividing their Legendre expansions by `sqrt(lambda)` gives unit integral squared norm on `[-lambda,lambda]`. The sign convention is positive at the origin. Only the degree-zero Legendre coefficient contributes to the integral. Therefore `c_0 = -v_4[0]/v_0[0]`, with `c_4 = 1`; the combined polynomial has its degree-zero coefficient explicitly zeroed to retain the defining zero integral.

## Resolution and accuracy

The existing `n_grid` argument is retained as a **maximum even-Legendre coefficient count** for ordinary candidate APIs. It is no longer a spatial-grid count on those routes. The resource policy limits combined retained outputs and numerical workspace to 1 GiB. It also requires `n_sample * (ceil(lambda^2)+1) * 2*basis_dimension * precision_words <= 2^32`, with `precision_words = 1` for native arithmetic and `ceil((precision_bits+64)/64)` for HP. These checks run before each candidate solve or output allocation; they are conservative work limits, not an accuracy claim. Ordinary defaults and a 512-coefficient, 256-sample, 3386-bit request at `lambda^2 <= 13` pass these preflights.

The selected dimension starts at at most 24 and increases until the computed operator residual resolves the requested working precision, subject to the maximum budget. An insufficient budget returns an error. There is no fallback to the historical Dirichlet model.

The residual includes both the finite Jacobi-matrix residual and the single omitted coupling from the last retained coefficient to the next degree. It is divided by `1+abs(eigenvalue)` and maximized over modes 0 and 4. Native arithmetic requires this diagnostic to be at most `2^-39`; HP arithmetic requires at most `2^(-(precision_bits-12))` and uses 64 internal guard bits. These are computed residual requirements, **not interval certificates of continuum eigenvalue, eigenfunction, or sampled-distance accuracy**. Matrix-entry rounding, point evaluation, and conditioning remain relevant. Repeated truncation agreement alone is not the acceptance test.

Results retain the requested resolution budget, actual retained basis dimension, discretization, and computed residual diagnostic. `CcmProlateDistanceHp` also retains `n_grid`, `n_sample`, working precision, and the candidate discretization/basis count. The sampled distance remains a finite, unweighted grid measurement.

The Eisenstein sum retains **open support**: terms at `x = lambda` are excluded. The underlying bounded prolate mode generally has a nonzero endpoint value. The last logarithmic sample is pinned to `u = lambda` and therefore has zero sum. Interior evaluation uses the bounded polynomial. At a computed sample, a product within `2^(-(p-8))*lambda` of the support boundary is treated as equality and excluded (`p=53` for binary64). This rounding band makes exact lower-endpoint and interior coincidences obey the same open-support convention across precisions; cutoffs inside that band cannot be distinguished from the boundary by this sampler.

## Historical finite Dirichlet model

The following explicit APIs preserve the old numerical object:

- `compute_k_lambda_finite_dirichlet_f64`;
- `hp::compute_k_lambda_finite_dirichlet`;
- `hp::compute_k_lambda_finite_dirichlet_via_cache`;
- the existing `build_pw_matrix*` functions and finite-grid trial-subspace forms.

For those APIs, `n_grid` still means interior finite-difference nodes. Existing finite-grid oracle tests continue testing this model. Its small-cutoff endpoint error can converge logarithmically; increasing MPFR precision does not remove that discretization error. Historical results must retain their finite-Dirichlet interpretation.

## Artifact identities and trial subspaces

The new spectrum identity is `prolate-bounded-legendre-even-spectrum-v1`; its retained dimension is an even Legendre basis count, and its cache precision includes internal guard bits. The candidate reuse-plan identity is `prolate-bounded-legendre-candidate-v6`. Returned candidate discretization is `prolate-bounded-legendre-even-v2`. The old spectrum identity `prolate-fd-working-precision-spectrum-v0.15.1-v2` remains exclusive to the explicit historical model. The identities do not alias. Every cached new-model eigenvalue is checked against its ordered index in the current source tridiagonal before use. The historical candidate also checks its two selected source-matrix indices before using a cached spectrum.

Finite-grid trial bases now pass a precision-scaled reorthogonalization check before Gram formation. The public solver independently requires the stored Gram form minus `max_diagonal*2^(-p/2)*I` to be positive definite by directed inertia, after exact power-of-two scaling. Unresolved bases fail closed. Returned Ritz reports must also satisfy a `2^(-p/2)` scaled-backward-error gate. This qualifies the rounded finite pencil; it does not certify an arbitrary original basis or continuum approximation.
