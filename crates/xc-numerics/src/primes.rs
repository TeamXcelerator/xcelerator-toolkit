// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Prime number utilities: sieve of Eratosthenes, prime enumeration.

/// Sieve of Eratosthenes up to `bound` (inclusive). Returns a sorted
/// vector of all primes ≤ bound.
pub fn sieve_primes(bound: u64) -> Vec<u64> {
    if bound < 2 { return Vec::new(); }
    let n = bound as usize;
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2usize;
    while p * p <= n {
        if is_prime[p] {
            let mut q = p * p;
            while q <= n { is_prime[q] = false; q += p; }
        }
        p += 1;
    }
    (2..=n).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}

/// Count primes up to `bound` (π(x) function).
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
}
