#!/usr/bin/env python3
"""One-shot, source-hash-guarded migration for the owner-authorized CCM audit.

The qualification job materializes these exact edits, runs Rust tests, and
commits the resulting source files. This script is removed by that commit.
No caches, credentials, network services, or research payloads are read here.
AI-generated implementation assistance; see docs/CCM_HARDENING.md.
"""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
changes = {}

def source(path, expected):
    data = (ROOT / path).read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if digest != expected:
        raise RuntimeError(f"source revision changed: {path}: {digest}")
    return data.decode("utf-8")

def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one edit anchor: {old[:100]!r}")
    return text.replace(old, new, 1)

path = "crates/xc-spectral/src/ccm/cutoff_free.rs"
s = source(path, "479942e6ca47f16b3705946e90386fd78b6656ed")
s = once(s, "            geometric_terms: 32,", "            geometric_terms: recommended_geometric_terms(integer_cutoff_c, modes, precision_bits),")
s = once(s, "        Ok(())\n    }\n\n    pub fn dimension", """        let dimension = self.modes.checked_mul(2).and_then(|n| n.checked_add(1));
        if dimension.and_then(|n| n.checked_mul(n)).is_none()
            || self.modes > (i64::MAX as usize) / 4
            || self.geometric_terms > ((i64::MAX - 1) / 4) as usize
        {
            bail!(\"cutoff-free CCM dimensions or series indices overflow\");
        }
        Ok(())
    }

    pub fn dimension""")
anchor = "#[derive(Clone, Debug, Eq, PartialEq)]\npub struct CutoffFreeConfig"
s = once(s, anchor, """/// Corrected finite-endpoint and aggregate-prime assembly identity.
/// Old inertia records may remain readable as records, but are not evidence
/// for this assembly. Sector certificates independently version their schema.
pub const ASSEMBLY_SEMANTICS: &str =
    \"ccm-cutoff-free-zero-endpoint-aggregate-primes-v0.14.4-v1\";

/// Deterministic conservative analytic-tail budget, with no floating-point
/// estimate of log(c). For c >= 2 and b=floor(log2(c)), the common special-value
/// tail is at most 6*2^(-2*M*b). The additional dimension allowance covers the
/// O(N) frequency factor and O(d) row-sum propagation in the archimedean form.
/// This controls the analytic series tail, NOT total assembly roundoff or a
/// spectral gap. Exact interval verification still decides certificate success.
pub fn recommended_geometric_terms(c: u64, modes: usize, precision_bits: u32) -> usize {
    let b = u64::from(63_u32.saturating_sub(c.leading_zeros())).max(1);
    let d = modes.saturating_mul(2).saturating_add(1);
    let dimension_bits = u64::from(usize::BITS - d.leading_zeros());
    let required = u64::from(precision_bits) + 2 * dimension_bits + 16;
    usize::try_from(required.div_ceil(2 * b)).unwrap_or(usize::MAX).max(1)
}

""" + anchor)
s = once(s, "        let component_evidence = (\n", "        let component_evidence = (\n            ASSEMBLY_SEMANTICS,\n")
s = once(s, "            std::collections::BTreeMap::from([\n", "            std::collections::BTreeMap::from([\n                (\"assembly_semantics\".to_owned(), ASSEMBLY_SEMANTICS.to_owned()),\n")
start = s.index("    if n == 0 {", s.index("fn special_values("))
end = s.index("    let n_interval =", start)
s = s[:start] + """    // n=0 needs the SAME finite-endpoint correction as every other mode.
    // psi_1(1/4)/4 alone is the integral over [0,infinity), not [0,L].
""" + s[end:]
s = once(s, "    Ok(SpecialValues { s, cc, xc })", """    // The sine and (cos-1) integrals vanish identically at zero frequency;
    // preserve that identity instead of subtracting two interval evaluations.
    if n == 0 {
        let zero = MpfrInterval::from_i64(0, p);
        Ok(SpecialValues { s: zero.clone(), cc: zero, xc })
    } else {
        Ok(SpecialValues { s, cc, xc })
    }""")
start = s.index("    let dimension = config.dimension();", s.index("pub fn assemble("))
s = s[:start] + """    // Sum the prime-power generators once per mode. Off-diagonal entries
    // are divided differences of these generators. Outward rounding remains
    // in force, and the changed enclosure arithmetic has a NEW identity.
    let mut sine_moments = Vec::with_capacity(config.modes + 1);
    let mut diagonal_moments = Vec::with_capacity(config.modes + 1);
    for n in 0..=config.modes {
        let nf = MpfrInterval::from_u64(n as u64, p);
        let mut sine = zero.clone();
        let mut diagonal = zero.clone();
        for (log_power, log_prime, sqrt_power) in &prime_data {
            let phase = pi.mul(&two).mul(&nf).mul(log_power).div(&l)?;
            let weight = log_prime.div(sqrt_power)?;
            if n != 0 {
                sine = sine.add(&phase.sin().mul(&weight));
            }
            diagonal = diagonal.add(
                &one.sub(&log_power.div(&l)?).mul(&two).mul(&phase.cos()).mul(&weight)
            );
        }
        sine_moments.push(sine);
        diagonal_moments.push(diagonal);
    }
    let signed_moment = |n: i64| {
        let value = sine_moments[n.unsigned_abs() as usize].clone();
        if n < 0 { value.neg() } else { value }
    };

""" + s[start:]
start = s.index("            let mut wp_cell = zero.clone();", s.index("pub fn assemble("))
end = s.index("            let tau_cell =", start)
s = s[:start] + """            let wp_cell = if n == m {
                diagonal_moments[n.unsigned_abs() as usize].clone()
            } else {
                signed_moment(m).sub(&signed_moment(n))
                    .div(&pi.mul(&MpfrInterval::from_i64(n - m, p)))?
            };
""" + s[end:]
# Regression fixtures are decimal point checks independently obtained by
# integrating CCM (4.4); they are NOT asserted to be interval certificates.
s += r'''

#[cfg(test)]
mod endpoint_regression {
    use super::*;

    #[test]
    fn zero_mode_matches_independent_defining_integral() {
        for (c, expected) in [
            (5, "2.25608966855868498643015465180094363462904"),
            (13, "2.88506309771709566996707209569382572588128"),
            (100, "3.67872561806049666284203395968608485646508"),
        ] {
            let matrix = assemble(&CutoffFreeConfig::new(c, 0, 192)).unwrap();
            let actual = Float::with_val(192, matrix.wr[0].midpoint());
            let expected = Float::with_val(192, Float::parse(expected).unwrap());
            let error = Float::with_val(192, actual - expected).abs();
            let tolerance = Float::with_val(192, Float::parse("1e-37").unwrap());
            assert!(error < tolerance, "zero-mode finite-endpoint regression at c={c}: {error}");
        }
    }

    #[test]
    fn analytic_tail_budget_grows_with_precision() {
        let low = CutoffFreeConfig::new(13, 4, 256);
        let high = CutoffFreeConfig::new(13, 4, 2048);
        assert!(high.geometric_terms > low.geometric_terms);
        assert!(recommended_geometric_terms(100, 4, 256) < low.geometric_terms);
        let b = 3_u64;
        let d_bits = u64::from(usize::BITS - low.dimension().leading_zeros());
        assert!(2 * b * low.geometric_terms as u64 >= 256 + 2 * d_bits + 16);
    }

    #[test]
    fn zero_frequency_keeps_exact_vanishing_integrals() {
        let p = 192;
        let l = MpfrInterval::from_u64(13, p).ln().unwrap();
        let zero = MpfrInterval::from_i64(0, p);
        let (psi, _) = complex_digamma(&q(1, 4, p), &zero).unwrap();
        let values = special_values(0, &l, &MpfrInterval::pi(p), &psi, 64).unwrap();
        assert_eq!(values.s.to_rational_interval(), zero.to_rational_interval());
        assert_eq!(values.cc.to_rational_interval(), zero.to_rational_interval());
        let (infinite, _) = complex_trigamma(&q(1, 4, p), &zero).unwrap();
        assert!(values.xc.upper() < infinite.div(&MpfrInterval::from_i64(4, p)).unwrap().lower());
    }

    #[test]
    fn aggregate_prime_entries_agree_with_direct_interval_sum() {
        let cfg = CutoffFreeConfig::new(13, 3, 192);
        let matrix = assemble(&cfg).unwrap();
        let p = cfg.precision_bits;
        let zero = MpfrInterval::from_i64(0, p);
        let one = MpfrInterval::from_i64(1, p);
        let two = MpfrInterval::from_i64(2, p);
        let pi = MpfrInterval::pi(p);
        let l = MpfrInterval::from_u64(13, p).ln().unwrap();
        for n in -3_i64..=3 {
            for m in -3_i64..=3 {
                let mut direct = zero.clone();
                for (power, prime, _) in prime_powers_up_to(13) {
                    let x = MpfrInterval::from_u64(power, p).ln().unwrap();
                    let phase = |mode: i64| pi.mul(&two)
                        .mul(&MpfrInterval::from_i64(mode, p)).mul(&x).div(&l).unwrap();
                    let kernel = if n == m {
                        one.sub(&x.div(&l).unwrap()).mul(&two).mul(&phase(n).cos())
                    } else {
                        phase(m).sin().sub(&phase(n).sin())
                            .div(&pi.mul(&MpfrInterval::from_i64(n-m, p))).unwrap()
                    };
                    direct = direct.add(&kernel
                        .mul(&MpfrInterval::from_u64(prime, p).ln().unwrap())
                        .div(&MpfrInterval::from_u64(power, p).sqrt().unwrap()).unwrap());
                }
                let index = (n+3) as usize * 7 + (m+3) as usize;
                assert!(matrix.wp[index].intersection(&direct.to_rational_interval()).is_some());
            }
        }
    }
}
'''
changes[path] = s

path = "crates/xc-spectral/src/ccm/sector_gap_certificate.rs"
s = source(path, "961cbe6c0d9927fbadefe76f4b18e3c467e12ede")
s = once(s, "const SCHEMA_VERSION: u32 = 2;", "const SCHEMA_VERSION: u32 = 3;")
s = once(s, "raw_full_cutoff_free_tau_certified_by_interval_ldlt_then_reflection_orbits_intersected_for_conditional_parity_projection", "raw_full_corrected_endpoint_aggregate_prime_tau_certified_by_interval_ldlt_then_reflection_orbits_intersected_for_conditional_parity_projection_v3")
s = once(s, "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2", "ccm-cutoff-free-sector-gap-certificate-v0.14.4-v3")
s = once(s, "cutoff_free_interval_assembly_full_matrix_ldlt_then_conditional_symmetry_intersection_parity_projection_exact_shifted_inertia_v2", "corrected_endpoint_aggregate_prime_interval_assembly_full_matrix_ldlt_then_conditional_symmetry_intersection_parity_projection_exact_shifted_inertia_v3")
s = once(s, '    let lambda_squared = params.lambda_sq_int().to_string();', '''    let geometric_terms = super::cutoff_free::recommended_geometric_terms(
        params.lambda_sq_int(), params.n_modes, precision_bits,
    );
    let lambda_squared = params.lambda_sq_int().to_string();''')
s = once(s, '            "geometric_terms": 32,', '            "geometric_terms": geometric_terms,\n            "assembly_semantics": super::cutoff_free::ASSEMBLY_SEMANTICS,')
s = once(s, "                || artifact.geometric_terms != 32", "                || artifact.geometric_terms != geometric_terms")
s = once(s, 'minimum_reader_version: ToolkitVersion::parse("0.14.1")?', 'minimum_reader_version: ToolkitVersion::parse("0.14.4")?')
changes[path] = s

path = "crates/xc-spectral/src/ccm/hp.rs"
s = source(path, "70cf1150163e3cf8d8d9f4cdcd656bd2c3f9b610")
s = once(s, "use rug::{ops::Pow, Float};", "use rug::{ops::Pow, Assign, Float};")
start = s.index("fn compute_prime_component_matrix(")
end = s.index("// Frozen allocating implementation", start)
part = s[start:end]
part = once(part, "            for (column, matrix_cell) in matrix_row.iter_mut().enumerate() {", """            // One scratch allocation per row, not per prime-power/cell.
            // Operation ordering is unchanged: this route must remain
            // bit-identical to compute_prime_component_matrix_reference.
            let mut sum = Float::with_val(prec, 0);
            let mut kernel = Float::with_val(prec, 0);
            let mut term = Float::with_val(prec, 0);
            for (column, matrix_cell) in matrix_row.iter_mut().enumerate() {""")
part = once(part, "                let mut sum = Float::with_val(prec, 0);", "                sum.assign(0);")
old = """                    let kernel = if n == m {
                        let mut factor = data.diagonal_factor.clone();
                        factor *= &data.cosines[n_index];
                        factor
                    } else {
                        let mut difference = data.sines[m_index].clone();
                        difference -= &data.sines[n_index];
                        difference /=
                            &difference_denominators[(n - m + 2 * n_modes as i64) as usize];
                        difference
                    };
                    let mut term = kernel;
                    term *= &data.log_prime;
                    term /= &data.sqrt_power;
                    sum += term;
                }
                *matrix_cell = sum;"""
new = """                    if n == m {
                        kernel.assign(&data.diagonal_factor);
                        kernel *= &data.cosines[n_index];
                    } else {
                        kernel.assign(&data.sines[m_index]);
                        kernel -= &data.sines[n_index];
                        kernel /=
                            &difference_denominators[(n - m + 2 * n_modes as i64) as usize];
                    }
                    term.assign(&kernel);
                    term *= &data.log_prime;
                    term /= &data.sqrt_power;
                    sum += &term;
                }
                matrix_cell.assign(&sum);"""
part = once(part, old, new)
s = s[:start] + part + s[end:]
changes[path] = s

path = "crates/xc-numerics/src/linalg.rs"
s = source(path, "d1f246d1caa3882d30c2f38ee6bd6629a08dfcc3")
s = once(s, "use rug::{ops::Pow, Float};", "use rug::{ops::Pow, Assign, Float};")
s = once(s, """            row[k] = factor.clone();
            for (j_off, j) in ((k + 1)..dim).enumerate() {
                let mut product = pivot_row[j_off].clone();
                product *= &factor;
                row[j] -= &product;
            }""", """            row[k].assign(&factor);
            let mut product = hp_zero(pivot.prec());
            for (j_off, j) in ((k + 1)..dim).enumerate() {
                product.assign(&pivot_row[j_off]);
                product *= &factor;
                row[j] -= &product;
            }""")
changes[path] = s

path = "crates/xc-spectral/src/ccm/mod.rs"
s = source(path, "32d8f9069f19648f84c5ffe6c8e182a7810981ad")
s = once(s, "pub mod convergence;", "pub mod convergence;\n\n#[cfg(feature = \"hp\")]\npub mod research;")
changes[path] = s

# Advance the development candidate without changing external dependencies.
for path in ["Cargo.toml", "tests/external-consumer/Cargo.toml"]:
    s = (ROOT / path).read_text()
    changes[path] = s.replace('"0.14.3"', '"0.14.4"')
for path in ["Cargo.lock", "tests/external-consumer/Cargo.lock"]:
    s = (ROOT / path).read_text()
    changes[path] = re.sub(r'(name = "xc-[^"]+"\nversion = ")0\.14\.3(")', r'\g<1>0.14.4\2', s)

# Stage every edit in memory before changing anything. A changed upstream
# revision or unmatched anchor leaves the checkout untouched.
for path, text in changes.items():
    target = ROOT / path
    temporary = target.with_name(target.name + ".ccm-migration-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    print(f"materialized {path}")
