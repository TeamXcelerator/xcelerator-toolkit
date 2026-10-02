#![cfg(feature = "hp")]
use xc_numerics::quadrature::{verify_gl_cache_dir, CacheFileStatus};
#[test]
fn plain_json_and_stale_zip_are_not_reported_runtime_usable() {
    use std::io::Write;
    let root = xc_core::test_support::TestDir::new("cache-admission");
    let data=serde_json::json!({"schema_version":1,"toolkit_version":"0.0.0","n_pts":1,"precision_bits":128,"nodes":["0"],"weights":["2"]}).to_string();
    let plain = root.join("prec128_npts1.json");
    std::fs::write(&plain, &data).unwrap();
    let zip_path = root.join("prec128_npts1.json.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
    zip.start_file(
        "prec128_npts1.json",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    zip.write_all(data.as_bytes()).unwrap();
    zip.finish().unwrap();
    let report = verify_gl_cache_dir(&root).unwrap();
    assert_eq!(report.ok_count(), 0);
    assert_eq!(report.failure_count(), 1);
    assert!(report.statuses.iter().any(|s|matches!(s,CacheFileStatus::Skipped{path,reason}if path==&plain&&reason.contains("runtime"))));
    assert!(report.statuses.iter().any(|s|matches!(s,CacheFileStatus::Stale{path,found_version,..}if path==&zip_path&&found_version=="0.0.0")));
    assert_eq!(std::fs::read_to_string(&plain).unwrap(), data);
}
