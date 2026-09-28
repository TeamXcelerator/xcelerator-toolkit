#![cfg(feature = "hp")]
use rug::Float;
use xc_core::{DecimalLiteral, LogBase};
use xc_numerics::hypothesis::convert_log_units;
#[test]
fn log_unit_conversion_cannot_export_invalid_enclosure_endpoints() {
    let huge = Float::with_val(128, 1) << (rug::float::exp_max() - 1) as u32;
    let source = DecimalLiteral::new(huge.to_string_radix(10, Some(48))).unwrap();
    assert!(convert_log_units(
        &source,
        LogBase::Decimal,
        false,
        LogBase::Natural,
        false,
        128
    )
    .is_err());
}
