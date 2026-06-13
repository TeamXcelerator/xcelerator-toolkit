// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Root-finding helpers (f64).
//!
//! Bisection on a sign-change bracket. The HP equivalents (Newton on
//! HP polynomials, bisection of HP-evaluated sign changes) are inlined
//! at their call sites in domain-specific code where the function
//! signature varies.

/// Bisect `f` on `[a, b]` assuming a sign change in the interval.
/// Returns the root or `None` if no sign change is detected.
///
/// If `f(a)` or `f(b)` evaluates to a value with magnitude below `tol`,
/// that endpoint is returned directly — including the exact-zero case.
/// Past iterations track the sign at the left endpoint via a cached
/// boolean so the inner loop never multiplies function values (avoids
/// over-/underflow when `f` returns very large or very small magnitudes,
/// which is common at HP boundaries even after `to_f64`).
pub fn bisect_f64<F: Fn(f64) -> f64>(
    f: &F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let fa = f(a);
    let fb = f(b);

    // Endpoint hits: either exactly zero or within tol → return that
    // endpoint as the root. Subsumes the |f(a) * f(b)| == 0 case so we
    // don't depend on float sign-of-zero behavior in the bracket check.
    if fa.abs() < tol { return Some(a); }
    if fb.abs() < tol { return Some(b); }

    // No sign change → no root we can guarantee. Use signum to avoid
    // over-/underflow in fa * fb when either side has extreme magnitude.
    if fa.signum() == fb.signum() { return None; }

    // Track the sign at the left endpoint instead of recomputing f(a)
    // each iteration. Equivalent to the textbook formulation, but
    // immune to multiplication over-/underflow on extreme inputs.
    let mut fa_sign = fa.signum();

    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if (b - a).abs() < tol || fm.abs() < tol { return Some(m); }

        if fm.signum() != fa_sign {
            // Sign change in [a, m] → root is there; shrink b.
            b = m;
        } else {
            // Sign change in [m, b] → root is there; shrink a, update sign.
            a = m;
            fa_sign = fm.signum();
        }
    }
    Some(0.5 * (a + b))
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Bisect should find the root of x² - 2 (i.e., √2) on [1, 2].
    #[test]
    fn bisect_finds_sqrt2() {
        let root = bisect_f64(&|x| x * x - 2.0, 1.0, 2.0, 1e-15, 200).unwrap();
        let expected = std::f64::consts::SQRT_2;
        let err = (root - expected).abs();
        assert!(err < 1e-14, "bisect_f64 √2: got {}, expected {}, err {:.2e}", root, expected, err);
    }

    /// Bisect should find the root of sin(x) near π on [3, 4].
    #[test]
    fn bisect_finds_pi() {
        let root = bisect_f64(&|x| x.sin(), 3.0, 4.0, 1e-15, 200).unwrap();
        let expected = std::f64::consts::PI;
        let err = (root - expected).abs();
        assert!(err < 1e-14, "bisect_f64 π: got {}, expected {}, err {:.2e}", root, expected, err);
    }

    /// Bisect should return None when there's no sign change.
    #[test]
    fn bisect_no_sign_change() {
        let result = bisect_f64(&|x| x * x + 1.0, -1.0, 1.0, 1e-15, 200);
        assert!(result.is_none());
    }

    /// Bisect should honor the tolerance `tol`: if the function evaluates
    /// to a value smaller than `tol` at the midpoint, return that midpoint.
    /// Conversely, the returned root should be within `tol` of the true root
    /// (or have a residual below `tol`).
    #[test]
    fn bisect_tolerance_honored() {
        // A loose tolerance (1e-3) on a function with a wide bracket
        // should produce a root within ~tol of the true value (√2).
        let tol = 1e-3;
        let root = bisect_f64(&|x| x * x - 2.0, 1.0, 2.0, tol, 200).unwrap();
        let expected = std::f64::consts::SQRT_2;
        // Either |root - expected| < tol, or |f(root)| < tol — bisection
        // can return on either condition.
        let root_err = (root - expected).abs();
        let resid = (root * root - 2.0).abs();
        assert!(root_err < tol || resid < tol,
            "bisect_f64 with tol={}: root={}, expected={}, |root-expected|={}, |f(root)|={}",
            tol, root, expected, root_err, resid);

        // A tighter tolerance (1e-12) should produce a root within ~tol.
        let tol_tight = 1e-12;
        let root2 = bisect_f64(&|x| x * x - 2.0, 1.0, 2.0, tol_tight, 200).unwrap();
        let root2_err = (root2 - expected).abs();
        let resid2 = (root2 * root2 - 2.0).abs();
        assert!(root2_err < tol_tight * 10.0 || resid2 < tol_tight,
            "bisect_f64 with tol={}: |root-expected|={}, |f(root)|={}",
            tol_tight, root2_err, resid2);
    }

    /// Bisect should respect `max_iter`: with a tiny iteration cap on a
    /// problem requiring many bisections, the returned approximation
    /// is the midpoint of the final bracket — which is `(a + b) / 2`
    /// after `max_iter` halvings of the original interval.
    #[test]
    fn bisect_max_iter_honored() {
        // Original bracket [1, 2] has width 1. After k bisections the
        // bracket has width 2^-k. With max_iter = 3 the final bracket
        // is 1/8 = 0.125 wide; the midpoint is within 1/16 = 0.0625 of
        // the true root.
        let tol = 0.0;  // disable tolerance-based exit so max_iter actually bites
        let root = bisect_f64(&|x| x * x - 2.0, 1.0, 2.0, tol, 3).unwrap();
        let expected = std::f64::consts::SQRT_2;
        let err = (root - expected).abs();
        // After 3 bisections, error ≤ 2^-4 = 0.0625.
        assert!(err < 0.07, "bisect with max_iter=3: err {} should be < 0.07", err);
        // But should not have converged to true precision.
        assert!(err > 1e-6, "bisect with max_iter=3 unexpectedly converged: err {}", err);
    }

    /// Narrow bracket: bisect on [√2 - 1e-3, √2 + 1e-3] should converge
    /// to √2 in a handful of iterations.
    #[test]
    fn bisect_narrow_bracket() {
        let expected = std::f64::consts::SQRT_2;
        let a = expected - 1e-3;
        let b = expected + 1e-3;
        let root = bisect_f64(&|x| x * x - 2.0, a, b, 1e-15, 200).unwrap();
        let err = (root - expected).abs();
        assert!(err < 1e-14, "narrow-bracket bisect: err {:.2e} should be < 1e-14", err);
    }

    /// Edge case: function value exactly zero at one endpoint. Bisect
    /// detects this via fa * fb == 0 (which is not > 0), so a root is
    /// returned. With f(x) = x at bracket [0, 1], the midpoint heads
    /// toward 0; the test verifies the returned root has small |f(x)|.
    #[test]
    fn bisect_zero_at_endpoint() {
        let result = bisect_f64(&|x| x, 0.0, 1.0, 1e-15, 200);
        assert!(result.is_some(),
            "bisect_f64 should accept zero-at-endpoint as a sign change");
        let root = result.unwrap();
        let resid = root.abs();
        assert!(resid < 1e-14,
            "bisect with f(0)=0: residual {} should be tiny", resid);
    }

    /// Edge case: max_iter = 0. The loop body never executes so bisect
    /// falls through to the final `Some(0.5*(a+b))` return. The result
    /// is the midpoint of the bracket — not converged, but not panicked.
    #[test]
    fn bisect_max_iter_zero_returns_midpoint() {
        // f(x) = x² - 2, bracket [1, 2]. max_iter=0 → no iterations.
        // Note: endpoints are checked BEFORE the loop, so if either
        // endpoint happens to be near zero it would return early.
        // Use tol=0 to disable that path; then result must be midpoint 1.5.
        let result = bisect_f64(&|x| x * x - 2.0, 1.0, 2.0, 0.0, 0);
        assert!(result.is_some(), "max_iter=0 should return Some (midpoint)");
        let root = result.unwrap();
        assert!((root - 1.5).abs() < 1e-15,
            "max_iter=0 should return midpoint 1.5, got {}", root);
    }

    /// Edge case: a == b (degenerate bracket). f(a) == f(b) so there's
    /// no sign change; bisect returns None.
    #[test]
    fn bisect_degenerate_bracket_a_eq_b() {
        let result = bisect_f64(&|x| x * x - 2.0, 1.5, 1.5, 1e-15, 200);
        // f(1.5) = 0.25 ≠ 0 and same sign on both sides → None.
        assert!(result.is_none(),
            "degenerate bracket a==b with no root at endpoint should return None");
    }
}
