use xc_spectral::lfunction::LFunctionSpec;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[test]
fn generated_subgroup_validator_matches_exhaustive_multiplication_tables() {
    let mut candidates = 0;
    let mut valid = 0;
    for q in 1..=16 {
        let units: Vec<_> = (0..q).filter(|&x| gcd(x, q) == 1).collect();
        for signs in 0usize..(1usize << units.len()) {
            let mut values = vec![0i8; q];
            for (bit, &unit) in units.iter().enumerate() {
                values[unit] = if signs & (1 << bit) == 0 { 1 } else { -1 };
            }
            let parity = u8::from(values[q - 1] == -1);
            let expected = values[1 % q] == 1
                && (0..q).all(|a| (0..q).all(|b| values[(a * b) % q] == values[a] * values[b]));
            let got = LFunctionSpec::new(q as u64, values.clone(), parity, "fresh_audit".into());
            assert_eq!(got.is_ok(), expected, "q={q}; values={values:?}");
            candidates += 1;
            if let Ok(character) = got {
                valid += 1;
                for base in 0..q {
                    let mut residue = 1 % q;
                    for exponent in 0..=8 {
                        assert_eq!(
                            character.chi_at_prime_power(base as u64, exponent),
                            values[residue]
                        );
                        residue = (residue * base) % q;
                    }
                }
            }
        }
    }
    eprintln!("independent exhaustive character oracle: {candidates} tables; {valid} accepted");
}
