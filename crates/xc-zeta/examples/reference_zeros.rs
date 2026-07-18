use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use xc_zeta::zeros::first_n_strings;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!("xc-zeta-example-{nonce}.json"));
    fs::write(
        &path,
        r#"["14.134725141734693790", "21.022039638771554993"]"#,
    )?;
    let zeros = first_n_strings(&path, 2);
    let _ = fs::remove_file(&path);
    println!("reference_zeros={:?}", zeros?);
    Ok(())
}
