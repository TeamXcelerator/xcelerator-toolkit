use xc_spectral::yakaboylu::try_v_r_matrix_element_f64;

#[test]
fn off_axis_equal_components_keep_the_small_real_part() {
    // Exact algebra for a=b=1/2: Re = e^4/(e^4+1/4), Im=(e^2/2)/(e^4+1/4).
    let epsilon = 1e-10_f64;
    let (real, imaginary) = try_v_r_matrix_element_f64(0.75, 0.0, 0.75, 0.5, epsilon).unwrap();
    let e2 = epsilon * epsilon;
    let denominator = e2 * e2 + 0.25;
    assert!((real / (e2 * e2 / denominator) - 1.0).abs() < 2e-15);
    assert!((imaginary / (0.5 * e2 / denominator) - 1.0).abs() < 2e-15);
}

#[test]
fn character_wire_type_rejects_unused_mathematical_fields() {
    let value = serde_json::json!({"modulus":1,"chi":[1],"parity":0,"label":"zeta","conductor":99});
    assert!(serde_json::from_value::<xc_spectral::lfunction::LFunctionSpec>(value).is_err());
}

#[cfg(feature = "hp")]
#[test]
fn off_axis_hp_small_component_matches_exact_rational_identity() {
    use rug::{Float, Rational};
    let p = 200;
    let epsilon = Float::with_val(p, Float::parse("1e-40").unwrap());
    let (real, imaginary) = xc_spectral::yakaboylu::try_v_r_matrix_element_hp(
        &Float::with_val(p, 0.75),
        &Float::with_val(p, 0),
        &Float::with_val(p, 0.75),
        &Float::with_val(p, 0.5),
        &epsilon,
        p,
    )
    .unwrap();
    let e2 = epsilon.to_rational().unwrap().square();
    let e4 = e2.clone().square();
    let denominator = e4.clone() + Rational::from((1, 4));
    let expected = Float::with_val(p + 64, e4 / denominator.clone());
    let expected_imaginary = Float::with_val(p + 64, (e2 / 2) / denominator);
    let tolerance = Float::with_val(p, 1) >> 190;
    assert!(Float::with_val(p + 64, real / expected - 1).abs() < tolerance);
    assert!(Float::with_val(p + 64, imaginary / expected_imaginary - 1).abs() < tolerance);
}

#[test]
fn external_provider_honest_replies_pass_and_prefetched_replies_fail() {
    use std::{fs, process::Command};
    use xc_spectral::target::{TargetEvaluatorF64, TargetProfileSpec};
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).unwrap();
    let root = std::env::temp_dir().join(format!(
        "xc-protocol-regression-{}-{}",
        std::process::id(),
        u64::from_ne_bytes(random)
    ));
    fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let source = root.join("provider.rs");
    fs::write(&source, r#"
use std::io::{self,BufRead,Write};
fn field<'a>(line:&'a str,name:&str)->&'a str {
 let marker=format!("\"{}\":",name);
 line.split_once(&marker).unwrap().1.split([',','}']).next().unwrap()
}
fn main(){
 for line in io::stdin().lock().lines(){
  let line=line.unwrap();
  let id=field(&line,"request_id");let nonce=field(&line,"request_nonce");let operation=field(&line,"operation");let bits=field(&line,"precision_bits");
  println!("{{\"protocol_version\":2,\"request_id\":{id},\"request_nonce\":{nonce},\"operation\":{operation},\"precision_bits\":{bits},\"ready\":true,\"value\":\"0.5\"}}");
  if operation=="\"initialize\"" && std::env::var_os("XC_TEST_PROVIDER_PREFETCH").is_some(){
   println!("{{\"protocol_version\":2,\"request_id\":2,\"request_nonce\":{nonce},\"operation\":\"evaluate\",\"precision_bits\":{bits},\"value\":\"999\"}}");
  }
  io::stdout().flush().unwrap();
 }
}
"#).unwrap();
    let executable = root.join(if cfg!(windows) {
        "provider.exe"
    } else {
        "provider"
    });
    let output = Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let digest = xc_cache::ContentDigest::sha256(&fs::read(&executable).unwrap()).0;
    let previous = std::env::var_os("XC_TARGET_PROVIDER_EXECUTABLE");
    let previous_prefetch = std::env::var_os("XC_TEST_PROVIDER_PREFETCH");
    std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", &executable);
    std::env::remove_var("XC_TEST_PROVIDER_PREFETCH");
    let spec = TargetProfileSpec::from_json(&serde_json::to_vec(&serde_json::json!({
        "schema_version":3,"profile_id":"protocol-regression","external_profile":{
            "lambda_squared":"4","evaluation_precision_bits":256,"provider_sha256":digest,"input":{}
        }
    })).unwrap()).unwrap();
    {
        let honest = TargetEvaluatorF64::from_spec(&spec).unwrap();
        assert_eq!(honest.try_value(1.25).unwrap(), 1.0);
    }
    std::env::set_var("XC_TEST_PROVIDER_PREFETCH", "1");
    let forged = TargetEvaluatorF64::from_spec(&spec);
    match previous {
        Some(value) => std::env::set_var("XC_TARGET_PROVIDER_EXECUTABLE", value),
        None => std::env::remove_var("XC_TARGET_PROVIDER_EXECUTABLE"),
    }
    match previous_prefetch {
        Some(value) => std::env::set_var("XC_TEST_PROVIDER_PREFETCH", value),
        None => std::env::remove_var("XC_TEST_PROVIDER_PREFETCH"),
    }
    assert!(forged.is_err());
}
