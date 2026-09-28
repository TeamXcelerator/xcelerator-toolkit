use xc_numerics::primes::try_sieve_primes;
#[test]
fn exact_prime_lists_match_independent_trial_division() {
    for bound in 0..=1000_u64 {
        let expected: Vec<_> = (2..=bound)
            .filter(|&n| (2..n).all(|d| n % d != 0))
            .collect();
        assert_eq!(try_sieve_primes(bound).unwrap(), expected);
    }
    assert!(try_sieve_primes(u64::MAX).is_err());
}
