//! Identity and resource boundaries for legacy local CCM JSON/ZIP caches.
use crate::ccm::LambdaSq;
use serde_json::{Map, Value};
use std::{io::Read, path::Path};

pub(super) const RESIDUAL_ARITHMETIC: &str = "directed_scaled_stored_infinity_norm_v2";

pub(super) fn shape(n: usize, p: u32, matrix: bool) -> Option<(usize, u64)> {
    if !(64..=1_000_000).contains(&p) {
        return None;
    }
    let dimension = n.checked_mul(2)?.checked_add(1)?;
    let count = if matrix {
        dimension.checked_mul(dimension)?
    } else {
        dimension
    };
    let count64 = u64::try_from(count).ok()?;
    let decimal = (u64::from(p) * 30103).div_ceil(100000) + 128;
    let budget = 8u64 << 30;
    if count64
        .checked_mul(4 * decimal + u64::from(p).div_ceil(8) + 128)?
        .checked_add(1 << 20)?
        > budget
    {
        return None;
    }
    Some((count, count64.checked_mul(decimal)?.checked_add(1 << 20)?))
}

/// Fingerprint the exact stored MPFR source, including shape and working precision.
/// Hexadecimal round-trip strings retain every significand bit without creating
/// an unbounded rational integer when an exponent is near the MPFR limits.
pub(super) fn tau_point_digest(
    tau: &[rug::Float],
    n: usize,
    p: u32,
) -> Option<xc_cache::ContentDigest> {
    use sha2::{Digest, Sha256};
    let (count, _) = shape(n, p, true)?;
    if tau.len() != count || tau.iter().any(|x| !x.is_finite() || x.prec() != p) {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(b"xc-standalone-tau-stored-mpfr-v1\0");
    hash.update(u64::try_from(n).ok()?.to_le_bytes());
    hash.update(p.to_le_bytes());
    for x in tau {
        let value = x.to_string_radix(16, None);
        hash.update(u64::try_from(value.len()).ok()?.to_le_bytes());
        hash.update(value.as_bytes());
    }
    Some(xc_cache::ContentDigest(format!("{:x}", hash.finalize())))
}

pub(super) fn identity_matches(
    object: &Map<String, Value>,
    lambda: LambdaSq,
    n: usize,
    p: u32,
    schema: u64,
) -> bool {
    lambda.value_f64.is_finite()
        && lambda.value_f64 > 1.0
        && lambda.value_u64 > 0
        && object.get("schema_version").and_then(Value::as_u64) == Some(schema)
        && object.get("n_modes").and_then(Value::as_u64) == u64::try_from(n).ok()
        && object.get("precision_bits").and_then(Value::as_u64) == Some(u64::from(p))
        && object
            .get("lambda_sq")
            .and_then(Value::as_f64)
            .map(f64::to_bits)
            == Some(lambda.value_f64.to_bits())
        && object.get("lambda_sq_mode").and_then(Value::as_str) == Some(lambda.mode_str())
        && object.get("lambda_sq_key").and_then(Value::as_str)
            == Some(lambda.filename_str().as_str())
        && object.get("prime_cutoff").and_then(Value::as_u64) == Some(lambda.value_u64)
}

pub(super) fn text(reader: impl Read, limit: u64) -> Option<String> {
    let mut value = String::new();
    reader
        .take(limit.checked_add(1)?)
        .read_to_string(&mut value)
        .ok()?;
    (value.len() as u64 <= limit).then_some(value)
}

pub(super) fn bytes(path: &Path, limit: u64) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    if file.metadata().ok()?.len() > limit {
        return None;
    }
    let mut value = Vec::new();
    file.take(limit.checked_add(1)?)
        .read_to_end(&mut value)
        .ok()?;
    (value.len() as u64 <= limit).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_standalone_source_identity_exact_points_and_domain() {
        use rug::{float::Special, Float};
        let source = (0..9).map(|x| Float::with_val(128, x)).collect::<Vec<_>>();
        let digest = tau_point_digest(&source, 1, 128).unwrap();
        assert!(digest.validate());
        assert_eq!(digest, tau_point_digest(&source.clone(), 1, 128).unwrap());
        let mut changed = source.clone();
        changed[1].next_up();
        assert_ne!(digest, tau_point_digest(&changed, 1, 128).unwrap());
        changed = source.clone();
        changed.swap(1, 2);
        assert_ne!(digest, tau_point_digest(&changed, 1, 128).unwrap());
        let wider = source
            .iter()
            .map(|x| Float::with_val(256, x))
            .collect::<Vec<_>>();
        assert_ne!(digest, tau_point_digest(&wider, 1, 256).unwrap());
        assert!(tau_point_digest(&source, 2, 128).is_none());
        assert!(tau_point_digest(&source, usize::MAX, 128).is_none());
        assert!(tau_point_digest(&source, 1, 0).is_none());
        assert!(tau_point_digest(&source, 1, 256).is_none());
        for value in [
            Float::with_val(128, Special::Nan),
            Float::with_val(128, Special::Infinity),
            Float::with_val(64, 1),
        ] {
            changed = source.clone();
            changed[0] = value;
            assert!(tau_point_digest(&changed, 1, 128).is_none());
        }
        for exponent in [-1_000_000_000, 1_000_000_000] {
            changed = source.clone();
            changed[0] = Float::with_val(128, 1) << exponent;
            let wide = tau_point_digest(&changed, 1, 128).unwrap();
            changed[0].next_up();
            assert_ne!(wide, tau_point_digest(&changed, 1, 128).unwrap());
        }
    }
    #[test]
    fn limits_precede_allocation_and_reads_obey_actual_byte_count() {
        assert!(shape(usize::MAX, 128, true).is_none());
        assert!(shape(1, 0, false).is_none());
        assert!(shape(1, u32::MAX, false).is_none());
        assert!(shape(1_000_000, 1_000_000, true).is_none());
        assert_eq!(
            text(std::io::Cursor::new(b"12345678"), 8).as_deref(),
            Some("12345678")
        );
        assert!(text(std::io::Cursor::new(b"123456789"), 8).is_none());
        assert!(text(std::io::Cursor::new([255u8]), 8).is_none());
    }
    #[test]
    fn integer_identity_does_not_collapse_above_f64_exact_range() {
        let lambda = LambdaSq::integer((1u64 << 53) + 1);
        let value = serde_json::json!({"schema_version":1,"n_modes":1,"precision_bits":128,
            "lambda_sq":lambda.value_f64,"lambda_sq_mode":"integer","lambda_sq_key":lambda.filename_str(),"prime_cutoff":lambda.value_u64});
        assert!(identity_matches(
            value.as_object().unwrap(),
            lambda,
            1,
            128,
            1
        ));
        assert!(!identity_matches(
            value.as_object().unwrap(),
            LambdaSq::integer(1u64 << 53),
            1,
            128,
            1
        ));
        assert!(!identity_matches(
            value.as_object().unwrap(),
            LambdaSq::fractional(lambda.value_f64),
            1,
            128,
            1
        ));
    }
}
