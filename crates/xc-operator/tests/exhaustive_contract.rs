use xc_core::{CancellationToken, ConfigDigest, ResourcePolicy};
use xc_operator::checkpoint::{assemble_symmetric_operator_f64, RestartableSymmetricAssemblerF64};
use xc_operator::vector_storage::{
    dot_stored_f64, SegmentedVectorF64, StoredDiagonalF64, StoredLinearOperatorF64, VectorReadF64,
    VectorWriteF64,
};
use xc_operator::OperatorError;
struct HugeAssembler {
    digest: ConfigDigest,
}
impl RestartableSymmetricAssemblerF64 for HugeAssembler {
    fn builder_id(&self) -> &str {
        "overflow_fixture"
    }
    fn builder_version(&self) -> u32 {
        1
    }
    fn operator_identity(&self) -> &str {
        "identity"
    }
    fn configuration_digest(&self) -> &ConfigDigest {
        &self.digest
    }
    fn dimension(&self) -> usize {
        usize::MAX
    }
    fn entry(&self, _: usize, _: usize) -> Result<f64, OperatorError> {
        panic!("impossible dimensions must be rejected before entry evaluation")
    }
}
#[test]
fn impossible_dense_construction_returns_error_without_panicking() {
    let result = std::panic::catch_unwind(|| {
        assemble_symmetric_operator_f64(
            &HugeAssembler {
                digest: ConfigDigest("a".repeat(64)),
            },
            1,
            1,
            None,
            &ResourcePolicy {
                maximum_memory_bytes: Some(u64::MAX),
                ..ResourcePolicy::default()
            },
            &CancellationToken::new(),
        )
    });
    assert!(
        result.is_ok(),
        "oversized descriptor caused a capacity panic"
    );
    assert!(result.unwrap().is_err());
}
#[test]
fn generous_workspace_does_not_allocate_beyond_stored_diagonal_dimension() {
    let input = SegmentedVectorF64::zeros(1, usize::MAX).unwrap();
    let output = SegmentedVectorF64::zeros(1, usize::MAX).unwrap();
    input.write_chunk(0, &[3.]).unwrap();
    StoredDiagonalF64::new(vec![2.])
        .unwrap()
        .apply_stored(&input, &output, usize::MAX)
        .unwrap();
    let mut actual = [0.];
    output.read_chunk(0, &mut actual).unwrap();
    assert_eq!(actual, [6.]);
}
#[test]
fn generous_workspace_does_not_allocate_beyond_stored_dot_dimension() {
    let input = SegmentedVectorF64::zeros(1, usize::MAX).unwrap();
    input.write_chunk(0, &[3.]).unwrap();
    assert_eq!(dot_stored_f64(&input, &input, usize::MAX).unwrap(), 9.);
}

#[test]
fn structured_and_composed_actions_match_exact_integer_oracles() {
    use xc_operator::{
        DenseSymmetricF64, LinearOperator, NegatedF64, PackedSymmetricF64, RankOneUpdateF64,
        ShiftedF64, SymmetricBandedF64, SymmetricOperator, TridiagonalF64,
    };
    for n in 1..=9 {
        let entry = |i: usize, j: usize| ((i.min(j) * 7 + i.max(j) * 3) % 11) as i64 - 5;
        let data: Vec<_> = (0..n)
            .flat_map(|i| (0..n).map(move |j| entry(i, j) as f64))
            .collect();
        let packed: Vec<_> = (0..n)
            .flat_map(|i| (0..=i).map(move |j| entry(i, j) as f64))
            .collect();
        let bands: Vec<_> = (0..n)
            .map(|d| (0..n - d).map(|i| entry(i, i + d) as f64).collect())
            .collect();
        let dense = DenseSymmetricF64::new("dense", n, data, 0.).unwrap();
        let packed = PackedSymmetricF64::new("packed", n, packed).unwrap();
        let banded = SymmetricBandedF64::new("banded", bands).unwrap();
        let x: Vec<_> = (0..n).map(|i| i as f64 - 4.).collect();
        let expected: Vec<_> = (0..n)
            .map(|i| (0..n).map(|j| entry(i, j) * (j as i64 - 4)).sum::<i64>() as f64)
            .collect();
        for op in [&dense as &dyn SymmetricOperator<f64>, &packed, &banded] {
            let mut actual = vec![0.; n];
            op.apply(&x, &mut actual).unwrap();
            assert_eq!(actual, expected);
            let exact_row_bound = (0..n)
                .map(|i| (0..n).map(|j| entry(i, j).abs()).sum::<i64>())
                .max()
                .unwrap() as f64;
            assert!(op.norm_bound().unwrap() >= exact_row_bound);
        }
        let mut actual = vec![0.; n];
        ShiftedF64::new(&dense, 3.)
            .unwrap()
            .apply(&x, &mut actual)
            .unwrap();
        assert_eq!(
            actual,
            expected
                .iter()
                .zip(&x)
                .map(|(y, x)| y - 3. * x)
                .collect::<Vec<_>>()
        );
        NegatedF64::new(&dense).apply(&x, &mut actual).unwrap();
        assert_eq!(actual, expected.iter().map(|y| -y).collect::<Vec<_>>());
        let v: Vec<_> = (0..n).map(|i| if i % 2 == 0 { 1. } else { -1. }).collect();
        let dot: f64 = v.iter().zip(&x).map(|(a, b)| a * b).sum();
        RankOneUpdateF64::new(&dense, 2., v.clone())
            .unwrap()
            .apply(&x, &mut actual)
            .unwrap();
        assert_eq!(
            actual,
            expected
                .iter()
                .zip(v)
                .map(|(y, v)| y + 2. * v * dot)
                .collect::<Vec<_>>()
        );
        let tridiag = TridiagonalF64::new(
            "tri",
            (0..n).map(|i| entry(i, i) as f64).collect(),
            (0..n - 1).map(|i| entry(i, i + 1) as f64).collect(),
        )
        .unwrap();
        tridiag.apply(&x, &mut actual).unwrap();
        let tri_expected: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| i.abs_diff(j) <= 1)
                    .map(|j| entry(i, j) * (j as i64 - 4))
                    .sum::<i64>() as f64
            })
            .collect();
        assert_eq!(actual, tri_expected);
    }
}
