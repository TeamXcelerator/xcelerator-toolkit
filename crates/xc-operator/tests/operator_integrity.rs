use xc_core::{CancellationToken, ConfigDigest, ResourcePolicy};
use xc_operator::checkpoint::{
    assemble_symmetric_operator_f64, OperatorConstructionOutcomeF64,
    RestartableSymmetricAssemblerF64,
};
use xc_operator::{
    DenseSymmetricF64, LinearOperator, OperatorError, PackedSymmetricF64, SymmetricBandedF64,
};

#[test]
fn independent_integer_actions_match_all_symmetric_storage_forms() {
    for n in 1..=9 {
        let entry = |i: usize, j: usize| -> i64 { ((i + j) % 5) as i64 - 2 };
        let x = (0..n).map(|j| (j as i64 - 3) as f64).collect::<Vec<_>>();
        let expected = (0..n)
            .map(|i| (0..n).map(|j| entry(i, j) * (j as i64 - 3)).sum::<i64>() as f64)
            .collect::<Vec<_>>();
        let dense = DenseSymmetricF64::new(
            "integer",
            n,
            (0..n)
                .flat_map(|i| (0..n).map(move |j| entry(i, j) as f64))
                .collect(),
            0.0,
        )
        .unwrap();
        let packed = PackedSymmetricF64::new(
            "integer",
            n,
            (0..n)
                .flat_map(|i| (0..=i).map(move |j| entry(i, j) as f64))
                .collect(),
        )
        .unwrap();
        let banded = SymmetricBandedF64::new(
            "integer",
            (0..n)
                .map(|d| (0..n - d).map(|i| entry(i, i + d) as f64).collect())
                .collect(),
        )
        .unwrap();
        for operator in [&dense as &dyn LinearOperator<f64>, &packed, &banded] {
            let mut y = vec![0.0; n];
            operator.apply(&x, &mut y).unwrap();
            assert_eq!(y, expected);
        }
    }
}
#[test]
fn induced_norm_bound_does_not_round_away_a_positive_row_term() {
    let epsilon = 2.0_f64.powi(-54);
    let op = DenseSymmetricF64::new("rounded-row-sum", 2, vec![1.0, epsilon, epsilon, 0.0], 0.0)
        .unwrap();
    assert_eq!(op.norm_bound(), Some(1.0_f64.next_up()));
}
struct ExactDiagonal {
    digest: ConfigDigest,
}
impl RestartableSymmetricAssemblerF64 for ExactDiagonal {
    fn builder_id(&self) -> &str {
        "audit_exact_diagonal"
    }
    fn builder_version(&self) -> u32 {
        1
    }
    fn operator_identity(&self) -> &str {
        "exact-diag-2-3"
    }
    fn configuration_digest(&self) -> &ConfigDigest {
        &self.digest
    }
    fn dimension(&self) -> usize {
        2
    }
    fn entry(&self, row: usize, column: usize) -> Result<f64, OperatorError> {
        Ok(if row == column { 2.0 + row as f64 } else { 0.0 })
    }
}
#[test]
fn checkpoint_validation_replays_source_and_rejects_retained_row_corruption() {
    // The historical witness is retained in the checkpoint commit and audit logs.
    let a = ExactDiagonal {
        digest: ConfigDigest("a".repeat(64)),
    };
    let resources = ResourcePolicy::default();
    let cancel = CancellationToken::new();
    let first = assemble_symmetric_operator_f64(&a, 1, 1, None, &resources, &cancel).unwrap();
    let OperatorConstructionOutcomeF64::Checkpointed { mut checkpoint } = first else {
        panic!("checkpoint expected")
    };
    checkpoint.assembled_rows[0] = 999.0;
    assert!(checkpoint.validate(&a, 1).is_err());
    assert!(
        assemble_symmetric_operator_f64(&a, 1, 1, Some(&checkpoint), &resources, &cancel).is_err()
    );
}
