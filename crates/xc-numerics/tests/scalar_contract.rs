use xc_numerics::{
    primes::{prime_count, try_sieve_primes},
    root_finding::bisect_f64,
};
#[test]
fn sieve_matches_independent_trial_division_at_every_small_cutoff() {
    for bound in (0..=512).chain([997, 1000, 4096]) {
        let expected: Vec<u64> = (2..=bound)
            .filter(|n| {
                let mut divisor = 2;
                while divisor <= n / divisor {
                    if n % divisor == 0 {
                        return false;
                    }
                    divisor += 1;
                }
                true
            })
            .collect();
        assert_eq!(try_sieve_primes(bound).unwrap(), expected);
        assert_eq!(prime_count(bound), expected.len());
    }
    assert!(try_sieve_primes(u64::MAX).is_err());
}
#[test]
fn bisection_recovers_exact_dyadic_roots_independently_of_residual_scale() {
    for j in -90..=90 {
        let root = (j as f64) / 32.;
        for exponent in [-900, 0, 900] {
            let scale = 2f64.powi(exponent);
            assert_eq!(
                bisect_f64(&|x| (x - root) * scale, -4., 4., 0., 16),
                Some(root)
            );
        }
    }
}
