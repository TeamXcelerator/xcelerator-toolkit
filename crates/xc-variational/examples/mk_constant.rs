#[cfg(feature = "hp")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rug::Rational;
    use xc_variational::maynard::MkMonomialReference;

    let engine = MkMonomialReference::new(2, 0)?;
    let certificate = engine.certificate(&[Rational::from((1, 1))])?;
    println!("{}", serde_json::to_string_pretty(&certificate)?);
    Ok(())
}

#[cfg(not(feature = "hp"))]
fn main() {
    eprintln!("run with: cargo run -p xc-variational --example mk_constant --features hp");
}
