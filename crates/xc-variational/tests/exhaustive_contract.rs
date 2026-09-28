#![cfg(feature = "hp")]
use xc_core::DecimalLiteral;
use xc_variational::maynard::*;

fn options() -> MkScaleAcceptanceOptions {
    MkScaleAcceptanceOptions {
        historical_dense_degree_limit: 3,
        target_degree: 4,
        precision_bits: 192,
        minimum_exact_lower_bound: xc_certify::ExactRationalRecord {
            numerator: "2".into(),
            denominator: "1".into(),
        },
        quotient_agreement_tolerance: DecimalLiteral::new("1e-45").unwrap(),
    }
}
#[test]
fn scale_acceptance_rejects_fabricated_streamed_storage() {
    let o = options();
    let mut r = run_mk_scale_acceptance(&o).unwrap();
    r.streamed_working_vector_bytes = 0;
    assert!(verify_mk_scale_acceptance(&r, &o).is_err());
}
#[test]
fn scale_acceptance_rejects_fabricated_dense_storage() {
    let o = options();
    let mut r = run_mk_scale_acceptance(&o).unwrap();
    r.equivalent_dense_forms_bytes += 1;
    assert!(verify_mk_scale_acceptance(&r, &o).is_err());
}
#[test]
fn partition_length_cap_does_not_require_allocating_k_entries() {
    let result = std::panic::catch_unwind(|| enumerate_integer_partitions(usize::MAX, 1));
    assert!(
        matches!(&result,Ok(Ok(p)) if p==&vec![IntegerPartition(vec![]),IntegerPartition(vec![1])]),
        "{result:?}"
    );
}
#[test]
fn impossible_multi_index_storage_returns_an_error() {
    let result = std::panic::catch_unwind(|| enumerate_multi_indices(usize::MAX, 0));
    assert!(matches!(result, Ok(Err(_))), "{result:?}");
}
#[test]
fn multi_index_enumeration_matches_combinatorial_dimension() {
    for k in 1..=7 {
        for d in 0..=5 {
            let indices = enumerate_multi_indices(k, d).unwrap();
            let expected = (1..=d).fold(1usize, |n, j| n * (k + j) / j);
            assert_eq!(indices.len(), expected);
            assert!(indices
                .iter()
                .all(|x| x.dimension() == k && x.total_degree() <= d));
            assert!(indices
                .windows(2)
                .all(|p| (p[0].total_degree(), &p[0]) < (p[1].total_degree(), &p[1])));
        }
    }
}
#[test]
fn constant_multi_index_space_does_not_recurse_through_variables() {
    let indices = enumerate_multi_indices(100_000, 0).unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].0, vec![0; 100_000]);
}
