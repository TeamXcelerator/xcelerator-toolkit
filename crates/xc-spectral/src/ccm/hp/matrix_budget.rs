use super::*;

#[test]
fn point_workspace_covers_both_dyadic_integer_parts() {
    let p = 3386;
    let point = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
    let points = vec![point; 3000];
    let actual_integer_bits = points
        .iter()
        .map(|value| {
            let exact = value.to_rational().unwrap();
            u128::from(exact.numer().significant_bits())
                + u128::from(exact.denom().significant_bits())
        })
        .sum::<u128>();
    let estimate =
        crate::ccm::certified_roots::boundary::rational_point_vector_workspace(points.iter())
            .unwrap();
    assert!(estimate >= actual_integer_bits.div_ceil(8));
    assert!(estimate < (8u128 << 30));
}

#[test]
fn dyadic_matrix_stages_use_linear_storage_budgets() {
    let p = 3386;
    let length = Float::with_val(p, 2);
    for (modes, count, active) in [(550, 3 * 551, 1), (300, 4 * 601, 2)] {
        let (dimension, base) = dimensions(modes, &length, p).unwrap();
        let points = vec![Float::with_val(p, 1); count];
        rational_stage_budget(dimension, base, points.iter(), active).unwrap();
        let extreme = Float::with_val(p, 1) >> 1_000_000_000u32;
        assert!(rational_stage_budget(dimension, base, std::iter::once(&extreme), active).is_err());
    }
    let (dimension, base) = dimensions(550, &length, p).unwrap();
    let moderate_points = vec![Float::with_val(p, 1) >> 60_000_000u32; 256];
    let rational_bytes = crate::ccm::certified_roots::boundary::rational_point_vector_workspace(
        moderate_points.iter(),
    )
    .unwrap();
    assert!(matrix_workspace_bytes(dimension, base) + rational_bytes < (8u128 << 30));
    rational_stage_budget(dimension, base, moderate_points.iter(), 1).unwrap();
    // Larger count exceeds the combined budget without allocating rational
    // numerators/denominators: these remain compact MPFR dyadic points.
    let larger_points = vec![Float::with_val(p, 1) >> 60_000_000u32; 1024];
    let rational_bytes = crate::ccm::certified_roots::boundary::rational_point_vector_workspace(
        larger_points.iter(),
    )
    .unwrap();
    assert!(matrix_workspace_bytes(dimension, base) + rational_bytes > (8u128 << 30));
    assert!(rational_stage_budget(dimension, base, larger_points.iter(), 1).is_err());
}
