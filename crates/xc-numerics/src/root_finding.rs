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
pub fn bisect_f64<F: Fn(f64) -> f64>(
    f: &F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> Option<f64> {
    let fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 { return None; }
    for _ in 0..max_iter {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if (b - a).abs() < tol || fm.abs() < tol { return Some(m); }
        if fm * f(a) < 0.0 { b = m; } else { a = m; }
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
}
