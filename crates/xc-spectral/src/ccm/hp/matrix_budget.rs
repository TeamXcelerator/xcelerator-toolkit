use super::*;

#[test]
fn matrix_admission_covers_campaign_shape_and_worker_count() {
    let p = 3386;
    let length = Float::with_val(p, 100).ln();
    let (dimension, base) = dimensions(500, &length, p).unwrap();
    assert_eq!(dimension, 1001);
    // Full-significand points near one cover the ordinary dyadic storage;
    // actual exponent spans remain checked when real input points arrive.
    let point = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
    let points = vec![point; 3 * 500 + p as usize / 2];
    rational_stage_budget(dimension, base, points.iter(), 122).unwrap();
}

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
    let budget = 8u128 << 30;
    let p = 3386;
    let length = Float::with_val(p, 2);
    for (modes, count, active) in [(550, 3 * 551, 1), (300, 4 * 601, 2)] {
        let (dimension, base) = dimensions_with_budget(modes, &length, p, budget).unwrap();
        let points = vec![Float::with_val(p, 1); count];
        rational_stage_budget_with_budget(dimension, base, points.iter(), active, budget).unwrap();
        let extreme = Float::with_val(p, 1) >> 1_000_000_000u32;
        assert!(rational_stage_budget_with_budget(
            dimension,
            base,
            std::iter::once(&extreme),
            active,
            budget
        )
        .is_err());
    }
    let (dimension, base) = dimensions_with_budget(550, &length, p, budget).unwrap();
    let moderate_points = vec![Float::with_val(p, 1) >> 60_000_000u32; 256];
    let rational_bytes = crate::ccm::certified_roots::boundary::rational_point_vector_workspace(
        moderate_points.iter(),
    )
    .unwrap();
    assert!(matrix_workspace_bytes(dimension, base) + rational_bytes < (8u128 << 30));
    rational_stage_budget_with_budget(dimension, base, moderate_points.iter(), 1, budget).unwrap();
    // Larger count exceeds the combined budget without allocating rational
    // numerators/denominators: these remain compact MPFR dyadic points.
    let larger_points = vec![Float::with_val(p, 1) >> 60_000_000u32; 1024];
    let rational_bytes = crate::ccm::certified_roots::boundary::rational_point_vector_workspace(
        larger_points.iter(),
    )
    .unwrap();
    assert!(matrix_workspace_bytes(dimension, base) + rational_bytes > (8u128 << 30));
    assert!(
        rational_stage_budget_with_budget(dimension, base, larger_points.iter(), 1, budget)
            .is_err()
    );
    rational_stage_budget_with_budget(dimension, base, larger_points.iter(), 1, 96u128 << 30)
        .unwrap();
}

#[test]
fn matrix_admission_covers_above_floor_campaign_with_declared_budget() {
    // Always exercise the campaign policy, independent of the invoking shell.
    let budget = 96u128 << 30;
    let p = 6708;
    let length = Float::with_val(p, 1200).ln();
    for modes in [800, 890, 970] {
        assert!(dimensions_with_budget(modes, &length, p, 8u128 << 30).is_err());
        let (dimension, base) = dimensions_with_budget(modes, &length, p, budget).unwrap();
        let point = Float::with_val(p, 1) + (Float::with_val(p, 1) >> (p - 1));
        let points = vec![point; 3 * modes + p as usize / 2];
        rational_stage_budget_with_budget(dimension, base, points.iter(), 122, budget).unwrap();
    }
}
