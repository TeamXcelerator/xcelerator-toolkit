#!/usr/bin/env python3
"""One-shot extension applied after apply_ccm_hardening.py; removed after tests.
No private data, caches, credentials, or network access.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"ambiguous source edit: {old[:100]!r}")
    return text.replace(old, new, 1)

path = ROOT / "crates/xc-spectral/src/ccm/hp.rs"
s = path.read_text()
start = s.index("fn compute_archimedean_integrals_tracked(")
end = s.index("fn assemble_pole_and_archimedean_components(", start)
part = s[start:end]
part = once(part, "fn compute_archimedean_integrals_tracked(", "fn compute_archimedean_integrals_tracked_with_bucket(")
part = once(part, "    fabric_cache: Option<&ArtifactCacheContext<'_>>,\n)", "    fabric_cache: Option<&ArtifactCacheContext<'_>>,\n    bucket: usize,\n)")
part = once(part, """    let pts_for_n: Vec<usize> = (0..=n_modes)
        .map(|n| base_pts.max(3 * n + prec_extra))
        .collect();""", """    let pts_for_n: Vec<usize> = if bucket == 1 {
        // Preserve the original production sequence and exact GL identities.
        (0..=n_modes).map(|n| base_pts.max(3 * n + prec_extra)).collect()
    } else {
        super::research::quadrature_orders(n_modes, base_pts, prec, bucket)?
    };""")
wrapper = """fn compute_archimedean_integrals_tracked(
    n_modes: usize,
    l: &Float,
    cfg: &HighPrecConfig,
    fabric_cache: Option<&ArtifactCacheContext<'_>>,
) -> Result<(ComputedArchimedeanIntegrals, Vec<ArtifactManifest>)> {
    compute_archimedean_integrals_tracked_with_bucket(n_modes, l, cfg, fabric_cache, 1)
}

"""
s = s[:start] + wrapper + part + s[end:]
s += r'''

/// Opt-in exact-rational-input matrix assembly for mechanism experiments.
/// This never substitutes a research matrix under an ordinary Tau cache key.
/// Only quadrature rules are reused by their existing order/precision identity.
/// The returned identity records the actual order list and prime arithmetic.
pub fn assemble_research_matrix_hp(
    cutoff: &super::research::ExactCutoff,
    n_modes: usize,
    cfg: &HighPrecConfig,
    options: &super::research::ResearchAssemblyOptions,
) -> Result<super::research::ResearchMatrixHp> {
    use super::research::{
        aggregate_prime_component_hp, quadrature_orders, PrimeAssemblyRoute,
        ResearchAssemblyIdentity, ResearchMatrixHp, RESEARCH_ASSEMBLY_SEMANTICS,
    };
    let dimension = options.validate(cutoff, n_modes)?;
    let precision_bits = cfg.precision_bits;
    let length = cutoff.log_length(precision_bits)?;
    let orders = quadrature_orders(n_modes, cfg.quad_points, precision_bits, options.quadrature_order_bucket)?;
    let (integrals, _) = compute_archimedean_integrals_tracked_with_bucket(
        n_modes, &length, cfg, None, options.quadrature_order_bucket,
    )?;
    let (pole, archimedean) = assemble_pole_and_archimedean_components(n_modes, &length, precision_bits, &integrals);
    let prime = match options.prime_route {
        PrimeAssemblyRoute::CanonicalCellSum => compute_prime_component_matrix(n_modes, cutoff.prime_cutoff(), &length, precision_bits),
        PrimeAssemblyRoute::AggregateGenerators => aggregate_prime_component_hp(cutoff, n_modes, precision_bits, options)?,
    };
    let mut entries = assemble_tau_components(&ComputedCcmMatrixComponents { pole, archimedean, prime }, precision_bits);
    force_symmetric(&mut entries, dimension);
    Ok(ResearchMatrixHp {
        identity: ResearchAssemblyIdentity {
            semantics: RESEARCH_ASSEMBLY_SEMANTICS.to_owned(), exact_cutoff: cutoff.canonical(),
            prime_cutoff: cutoff.prime_cutoff(), n_modes, precision_bits,
            prime_route: options.prime_route, quadrature_orders: orders,
            assurance: "computed_point_matrix_not_certified".to_owned(),
        }, entries,
    })
}

#[cfg(test)]
mod audit_research_tests {
    use super::*;
    use super::super::research::*;

    #[test]
    fn audit_aggregate_prime_matches_canonical_at_multiple_precisions() {
        for precision_bits in [128, 256] {
            for c in [5, 13, 100] {
                let cutoff = ExactCutoff::parse(&c.to_string()).unwrap();
                let length = cutoff.log_length(precision_bits).unwrap();
                let options = ResearchAssemblyOptions::default();
                let expected = compute_prime_component_matrix(6, c, &length, precision_bits);
                let actual = aggregate_prime_component_hp(&cutoff, 6, precision_bits, &options).unwrap();
                let tolerance = Float::with_val(precision_bits, 2).pow(-((precision_bits - 32) as i32));
                for (a,b) in actual.iter().zip(&expected) {
                    let difference = Float::with_val(precision_bits, a-b).abs();
                    assert!(difference < tolerance, "prime generator disagreement at c={c}");
                }
            }
        }
    }

    #[test]
    fn audit_research_route_retains_its_identity_and_never_floors_from_f64() {
        let cutoff = ExactCutoff::parse("12.99999999999999999999999999999999999999").unwrap();
        let mut cfg = HighPrecConfig::for_decimal_digits(40);
        cfg.precision_bits = 192;
        cfg.quad_points = 128;
        let options = ResearchAssemblyOptions {
            prime_route: PrimeAssemblyRoute::AggregateGenerators,
            quadrature_order_bucket: 32,
            ..ResearchAssemblyOptions::default()
        };
        let matrix = assemble_research_matrix_hp(&cutoff, 2, &cfg, &options).unwrap();
        assert_eq!(matrix.identity.prime_cutoff, 12);
        assert_eq!(matrix.identity.prime_route, PrimeAssemblyRoute::AggregateGenerators);
        assert_eq!(matrix.identity.quadrature_orders, vec![128;3]);
        assert!(matrix.content_digest().unwrap().validate());
        assert_eq!(matrix.entries.len(), 25);
    }

    #[test]
    #[ignore = "explicit release-mode performance measurement, no speed assertion"]
    fn audit_prime_generator_benchmark() {
        let p = 256;
        let cutoff = ExactCutoff::parse("500").unwrap();
        let length = cutoff.log_length(p).unwrap();
        let options = ResearchAssemblyOptions::default();
        for n in [32, 64, 128] {
            let mut canonical = Vec::new();
            let mut aggregate = Vec::new();
            for _ in 0..3 {
                let start = std::time::Instant::now();
                let reference = compute_prime_component_matrix(n, 500, &length, p);
                canonical.push(start.elapsed().as_nanos());
                let start = std::time::Instant::now();
                let candidate = aggregate_prime_component_hp(&cutoff,n,p,&options).unwrap();
                aggregate.push(start.elapsed().as_nanos());
                let tolerance = Float::with_val(p,2).pow(-((p-32) as i32));
                assert!(reference.iter().zip(&candidate).all(|(a,b)| Float::with_val(p,a-b).abs() < tolerance));
            }
            canonical.sort_unstable(); aggregate.sort_unstable();
            println!("CCM_BENCH {{\"cutoff\":500,\"n_modes\":{n},\"precision_bits\":{p},\"samples\":3,\"canonical_median_ns\":{},\"aggregate_median_ns\":{},\"peak_rss_bytes\":{:?}}}", canonical[1], aggregate[1], peak_resident_memory_bytes());
        }
    }
}
'''
path.write_text(s)

# Preserve the original cloned operand's precision even for public LU callers
# supplying mixed-precision Float entries. Uniform HP matrices incur no reset.
path = ROOT / "crates/xc-numerics/src/linalg.rs"
s = path.read_text()
s = once(s, "                product.assign(&pivot_row[j_off]);", """                if product.prec() != pivot_row[j_off].prec() {
                    product.set_prec(pivot_row[j_off].prec());
                }
                product.assign(&pivot_row[j_off]);""")
path.write_text(s)

path = ROOT / "crates/xc-spectral/src/ccm/research.rs"
s = path.read_text()
s = once(s, "pub fn analyze_nested_gram_schur_hp(", "#[allow(clippy::too_many_arguments)]\npub fn analyze_nested_gram_schur_hp(")
path.write_text(s)
print("integrated exact-input research assembly, canonical allocation reuse, and release benchmark")
