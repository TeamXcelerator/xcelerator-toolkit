use xc_operator::{DenseSymmetricF64, LinearOperator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let operator = DenseSymmetricF64::new("two-by-two", 2, vec![2.0, -1.0, -1.0, 2.0], 0.0)?;
    let mut output = vec![0.0; 2];
    operator.apply(&[1.0, 3.0], &mut output)?;
    println!("A*x={output:?}");
    Ok(())
}
