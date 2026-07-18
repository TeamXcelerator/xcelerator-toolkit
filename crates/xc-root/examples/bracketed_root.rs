use xc_root::{
    safeguarded_newton_f64, RealFunctionF64, RootBracketF64, RootError, RootStoppingF64,
};

struct CosMinusX;

impl RealFunctionF64 for CosMinusX {
    fn evaluate(&self, x: f64) -> Result<f64, RootError> {
        Ok(x.cos() - x)
    }

    fn derivative(&self, x: f64) -> Result<f64, RootError> {
        Ok(-x.sin() - 1.0)
    }
}

fn main() {
    let root = safeguarded_newton_f64(
        &CosMinusX,
        RootBracketF64 {
            lower: 0.0,
            upper: 1.0,
        },
        0.5,
        &RootStoppingF64::default(),
    )
    .unwrap();
    println!(
        "root = {:.17e}, residual = {:.3e}",
        root.midpoint, root.residual
    );
}
