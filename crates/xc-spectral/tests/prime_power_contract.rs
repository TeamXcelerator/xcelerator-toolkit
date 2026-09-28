use xc_spectral::ccm::{prime_powers_up_to, try_prime_powers_up_to};

#[test]
fn prime_powers_agree_with_independent_integer_factorization() {
    for bound in 0..=512u64 {
        let expected: Vec<_> = (2..=bound)
            .filter_map(|n| {
                let p = (2..=n).find(|p| n % p == 0).unwrap();
                let mut quotient = n;
                let mut exponent = 0;
                while quotient % p == 0 {
                    quotient /= p;
                    exponent += 1;
                }
                (quotient == 1).then_some((n, p, exponent))
            })
            .collect();
        assert_eq!(try_prime_powers_up_to(bound).unwrap(), expected);
        assert_eq!(prime_powers_up_to(bound), expected);
    }
}

#[test]
fn unrepresentable_prime_sieve_bounds_return_errors() {
    assert!(try_prime_powers_up_to(u64::MAX).is_err());
    assert!(xc_numerics::primes::try_sieve_primes(u64::MAX).is_err());
}
