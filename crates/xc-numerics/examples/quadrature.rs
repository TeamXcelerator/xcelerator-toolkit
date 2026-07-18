use xc_numerics::quadrature::gauss_legendre_npt_f64;

fn main() {
    let integral = gauss_legendre_npt_f64(|x| x * x, 0.0, 1.0, 16);
    println!("integral={integral:.16e} expected={:.16e}", 1.0 / 3.0);
}
