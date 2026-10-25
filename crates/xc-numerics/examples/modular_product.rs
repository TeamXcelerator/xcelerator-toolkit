// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Bounded exact-product driver. Input integers are hexadecimal strings; all
//! paths and inputs are supplied by the caller. No cache or solver is invoked.

#[cfg(feature = "modular-product")]
fn main() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use num_bigint::BigInt;
    use xc_numerics::modular_product::{multiply, ProductBackend, ProductLimits};
    let arguments: Vec<_> = std::env::args().collect();
    if arguments.len() != 4 {
        bail!("usage: modular_product INPUT_JSON cpu|cuda OUTPUT_JSON");
    }
    let path = std::path::Path::new(&arguments[1]);
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((128 << 20) + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 128 << 20 {
        bail!("input exceeds driver size limit");
    }
    let input: serde_json::Value = serde_json::from_slice(&bytes)?;
    let dimension = |key: &str| -> anyhow::Result<usize> {
        usize::try_from(input[key].as_u64().context("missing dimension")?)
            .context("dimension overflow")
    };
    let integers = |key: &str| -> anyhow::Result<Vec<BigInt>> {
        input[key]
            .as_array()
            .context("missing integer array")?
            .iter()
            .map(|v| {
                BigInt::parse_bytes(
                    v.as_str()
                        .context("integer must be a hexadecimal string")?
                        .as_bytes(),
                    16,
                )
                .context("invalid hexadecimal integer")
            })
            .collect()
    };
    let backend = match arguments[2].as_str() {
        "cpu" => ProductBackend::Cpu,
        #[cfg(feature = "cuda-product")]
        "cuda" => ProductBackend::Cuda,
        _ => bail!("backend is unavailable in this build"),
    };
    let a = integers("left")?;
    let b = integers("right")?;
    let started = std::time::Instant::now();
    let result = multiply(
        &a,
        &b,
        dimension("rows")?,
        dimension("inner")?,
        dimension("columns")?,
        backend,
        ProductLimits {
            max_workspace_bytes: 2 << 30,
            max_input_bits: 16_000,
            max_modular_scalar_products: 100_000_000_000,
            workers: 8,
        },
    )?;
    let total = started.elapsed().as_secs_f64();
    let report = serde_json::json!({
        "rows":result.rows,"columns":result.columns,"backend":arguments[2],
        "moduli":result.moduli,"reconstruction_bits":result.reconstruction_bits,
        "required_bound_bits":result.required_bound_bits,"estimated_workspace_bytes":result.estimated_workspace_bytes,
        "seconds":{"total_product":total,"planning":result.timings.planning_seconds,
            "conversion":result.timings.conversion_seconds,"products":result.timings.products_seconds,
            "reconstruction":result.timings.reconstruction_seconds},
        "entries":result.entries.iter().map(|v|v.to_str_radix(16)).collect::<Vec<_>>()
    });
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&arguments[3])?;
    serde_json::to_writer(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!("{}", report["seconds"]);
    Ok(())
}

#[cfg(not(feature = "modular-product"))]
fn main() {
    eprintln!("enable the modular-product feature to run this example");
    std::process::exit(2);
}
