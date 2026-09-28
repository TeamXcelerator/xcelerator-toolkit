// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Prime number utilities: sieve of Eratosthenes, prime enumeration.

/// Sieve of Eratosthenes through the inclusive bound, in ascending order.
/// Panics when the platform cannot index or allocate the sieve. Use
/// [`try_sieve_primes`] to receive an explicit error for those domains.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    try_sieve_primes(bound).expect("representable and allocatable prime sieve required")
}

/// Checked exact integer sieve. Time and storage grow with the inclusive bound.
pub fn try_sieve_primes(bound: u64) -> anyhow::Result<Vec<u64>> {
    if bound < 2 {
        return Ok(Vec::new());
    }
    let n = usize::try_from(bound)
        .map_err(|_| anyhow::anyhow!("prime bound exceeds platform indices"))?;
    let length = n
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("prime sieve length overflow"))?;
    let mut is_prime = Vec::new();
    is_prime.try_reserve_exact(length)?;
    is_prime.resize(length, true);
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2usize;
    while p <= n / p {
        if is_prime[p] {
            let mut q = p * p; // p <= n/p proves this product fits.
            loop {
                is_prime[q] = false;
                if q > n - p {
                    break;
                }
                q += p;
            }
        }
        p += 1;
    }
    let count = is_prime[2..].iter().filter(|&&prime| prime).count();
    let mut primes = Vec::new();
    primes.try_reserve_exact(count)?;
    primes.extend((2..=n).filter(|&i| is_prime[i]).map(|i| i as u64));
    Ok(primes)
}

/// Count primes through the inclusive bound; allocates the sieve and prime list.
/// Returns zero for bounds below two. Has the same resource contract as
/// [`sieve_primes`].
pub fn prime_count(bound: u64) -> usize {
    sieve_primes(bound).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sieve_small() {
        assert_eq!(sieve_primes(1), Vec::<u64>::new());
        assert_eq!(sieve_primes(2), vec![2u64]);
        assert_eq!(sieve_primes(10), vec![2u64, 3, 5, 7]);
        assert_eq!(sieve_primes(13), vec![2u64, 3, 5, 7, 11, 13]);
    }

    #[test]
    fn prime_count_100() {
        assert_eq!(prime_count(100), 25); // π(100) = 25
    }

    #[test]
    fn prime_count_1000() {
        assert_eq!(prime_count(1000), 168); // π(1000) = 168
    }

    /// Boundary: bound = 0 → no primes.
    #[test]
    fn sieve_zero_bound() {
        assert_eq!(sieve_primes(0), Vec::<u64>::new());
        assert_eq!(prime_count(0), 0);
    }

    /// The primes themselves should be returned exactly.
    #[test]
    fn sieve_exact_prime_boundary() {
        // Bound equal to a prime: 7 is prime → should be included.
        assert!(sieve_primes(7).contains(&7));
        // Bound one below a prime: 6 → should NOT include 7.
        assert!(!sieve_primes(6).contains(&7));
    }

    /// Consecutive outputs of sieve_primes should be strictly ascending
    /// and every element should be prime (no composites).
    #[test]
    fn sieve_output_is_sorted_and_prime() {
        let primes = sieve_primes(50);
        for w in primes.windows(2) {
            assert!(w[0] < w[1], "sieve output should be strictly ascending");
        }
        // Quick primality check: none of the returned values should be
        // divisible by any earlier prime.
        for &p in &primes {
            for &q in primes.iter().take_while(|&&q| q * q <= p) {
                assert_ne!(
                    p % q,
                    0,
                    "{} is divisible by {} but was returned as prime",
                    p,
                    q
                );
            }
        }
    }
}
