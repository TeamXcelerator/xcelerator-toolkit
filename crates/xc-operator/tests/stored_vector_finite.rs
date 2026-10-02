use xc_operator::vector_storage::{
    dot_stored_f64, FileBackedVectorF64, SegmentedVectorF64, StoredDiagonalF64,
    StoredLinearOperatorF64, VectorReadF64, VectorWriteF64,
};

#[test]
fn dot_rejects_nonfinite_inputs_products_and_partial_sums() {
    for (a, b) in [
        (vec![f64::MAX], vec![2.0]),
        (vec![f64::MAX; 2], vec![1.0; 2]),
        (vec![f64::NAN], vec![0.0]),
        (vec![0.0], vec![f64::INFINITY]),
        (vec![f64::NEG_INFINITY], vec![1.0]),
        (vec![f64::MAX; 2], vec![2.0, -2.0]),
    ] {
        let left = SegmentedVectorF64::zeros(a.len(), 1).unwrap();
        let right = SegmentedVectorF64::zeros(b.len(), 2).unwrap();
        left.write_chunk(0, &a).unwrap();
        right.write_chunk(0, &b).unwrap();
        for workspace in [1, 2, 7] {
            assert!(dot_stored_f64(&left, &right, workspace).is_err());
        }
    }
}

#[test]
fn diagonal_rejects_invalid_input_and_output_before_writing_the_chunk() {
    for (diagonal, source) in [(f64::MAX, 2.0), (0.0, f64::NAN), (1.0, f64::INFINITY)] {
        let input = SegmentedVectorF64::zeros(1, 1).unwrap();
        let output = SegmentedVectorF64::zeros(1, 1).unwrap();
        input.write_chunk(0, &[source]).unwrap();
        output.write_chunk(0, &[17.0]).unwrap();
        assert!(StoredDiagonalF64::new(vec![diagonal])
            .unwrap()
            .apply_stored(&input, &output, 1)
            .is_err());
        let mut result = [0.0];
        output.read_chunk(0, &mut result).unwrap();
        assert_eq!(result, [17.0]);
    }
}

#[test]
fn finite_stored_arithmetic_matches_exact_integer_reference_for_every_chunking() {
    for n in [1, 2, 7, 17] {
        for segment in [1, 2, 5] {
            for workspace in [1, 2, 3, 8] {
                let a = (0..n).map(|i| i as i64 - 8).collect::<Vec<_>>();
                let b = (0..n).map(|i| 3 - i as i64).collect::<Vec<_>>();
                let expected = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| i128::from(*x) * i128::from(*y))
                    .sum::<i128>();
                let left = SegmentedVectorF64::zeros(n, segment).unwrap();
                let right = SegmentedVectorF64::zeros(n, segment + 1).unwrap();
                let output = SegmentedVectorF64::zeros(n, segment + 2).unwrap();
                left.write_chunk(0, &a.iter().map(|&v| v as f64).collect::<Vec<_>>())
                    .unwrap();
                right
                    .write_chunk(0, &b.iter().map(|&v| v as f64).collect::<Vec<_>>())
                    .unwrap();
                assert_eq!(
                    dot_stored_f64(&left, &right, workspace).unwrap(),
                    expected as f64
                );
                StoredDiagonalF64::new(b.iter().map(|&v| v as f64).collect())
                    .unwrap()
                    .apply_stored(&left, &output, workspace)
                    .unwrap();
                let mut actual = vec![0.0; n];
                output.read_chunk(0, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    a.iter()
                        .zip(&b)
                        .map(|(x, y)| (x * y) as f64)
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn corrupted_file_values_cannot_escape_the_arithmetic_boundary() {
    let dir = xc_core::test_support::TestDir::new("stored-finite");
    let path = dir.join("nan.bin");
    // Raw retained bytes can contain nonfinite values even though write_chunk
    // checks its own inputs. Arithmetic must validate the decoded source.
    std::fs::write(&path, f64::NAN.to_le_bytes()).unwrap();
    let input = FileBackedVectorF64::open_existing(&path, 1, 1).unwrap();
    let output = SegmentedVectorF64::zeros(1, 1).unwrap();
    assert!(dot_stored_f64(&input, &output, 1).is_err());
    assert!(StoredDiagonalF64::new(vec![0.0])
        .unwrap()
        .apply_stored(&input, &output, 1)
        .is_err());
}
