//! Operational admission for private target documents. These limits do not
//! change target coefficients, evaluation arithmetic, or semantic identities.
use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;

pub const TARGET_MAXIMUM_BYTES_ENV: &str = "XC_TARGET_MAXIMUM_BYTES";
const DEFAULT_MAXIMUM_BYTES: u64 = 64 << 20;
const HARD_MAXIMUM_BYTES: u64 = 4 << 30;
const HARD_WORKING_BYTES: u64 = i64::MAX as u64;
const PROTOCOL_ENVELOPE_BYTES: u64 = 4096;

fn positive_setting(value: &str, name: &str, maximum: u64) -> Result<u64> {
    anyhow::ensure!(
        !value.is_empty()
            && value.len() <= 20
            && value.as_bytes()[0] != b'0'
            && value.bytes().all(|byte| byte.is_ascii_digit()),
        "{name} must be a canonical positive decimal byte count"
    );
    let count = value
        .parse::<u64>()
        .with_context(|| format!("{name} byte count overflows"))?;
    anyhow::ensure!(count <= maximum, "{name} exceeds its byte limit");
    Ok(count)
}

fn maximum_from_settings(target: Option<&str>, working: Option<&str>) -> Result<u64> {
    let mut maximum = target
        .map(|value| positive_setting(value, TARGET_MAXIMUM_BYTES_ENV, HARD_MAXIMUM_BYTES))
        .transpose()?
        .unwrap_or(DEFAULT_MAXIMUM_BYTES);
    if let Some(working) = working {
        // Parsing, JSON values, request frames, and the provider can coexist.
        // The document is admitted within one eighth of the supplied budget.
        maximum = maximum
            .min(positive_setting(working, "XC_RESEARCH_WORKING_BYTES", HARD_WORKING_BYTES)? / 8);
    }
    anyhow::ensure!(maximum > 0, "target document has no working-memory budget");
    usize::try_from(maximum).context("target byte limit exceeds this platform's address space")?;
    Ok(maximum)
}

/// Maximum complete target-document size admitted by this process. The legacy
/// default is 64 MiB. An operator may explicitly raise it to at most 4 GiB;
/// an explicit research working-memory budget also caps it at one eighth.
pub fn target_maximum_bytes() -> Result<u64> {
    let target =
        std::env::var(TARGET_MAXIMUM_BYTES_ENV)
            .map(Some)
            .or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                error => Err(error),
            })?;
    let working = std::env::var("XC_RESEARCH_WORKING_BYTES")
        .map(Some)
        .or_else(|error| match error {
            std::env::VarError::NotPresent => Ok(None),
            error => Err(error),
        })?;
    maximum_from_settings(target.as_deref(), working.as_deref())
}

pub(super) fn target_request_maximum_bytes() -> Result<u64> {
    target_maximum_bytes()?
        .checked_add(PROTOCOL_ENVELOPE_BYTES)
        .context("target request byte limit overflows")
}

fn read_bounded(mut input: impl Read, maximum: u64, reserve: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reserve)
        .context("cannot reserve target document memory")?;
    let mut chunk = [0u8; 64 << 10];
    loop {
        let count = input.read(&mut chunk)?;
        if count == 0 {
            return Ok(bytes);
        }
        let length = bytes
            .len()
            .checked_add(count)
            .context("target document length overflows")?;
        anyhow::ensure!(
            u64::try_from(length)? <= maximum,
            "target specification exceeds its configured byte limit"
        );
        bytes
            .try_reserve(count)
            .context("cannot reserve target document memory")?;
        bytes.extend_from_slice(&chunk[..count]);
    }
}

/// Read every byte of an admitted target document with checked allocation.
/// A size change after metadata inspection is checked while reading as well.
pub fn read_target_specification(path: &Path) -> Result<Vec<u8>> {
    let maximum = target_maximum_bytes()?;
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed reading target specification {}", path.display()))?;
    let length = file.metadata()?.len();
    anyhow::ensure!(
        length <= maximum,
        "target specification exceeds its configured byte limit"
    );
    read_bounded(file, maximum, usize::try_from(length)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_document_limits_preserve_defaults_and_require_explicit_larger_admission() {
        assert_eq!(maximum_from_settings(None, None).unwrap(), 64 << 20);
        assert_eq!(
            maximum_from_settings(Some("1073741824"), None).unwrap(),
            1 << 30
        );
        assert_eq!(
            maximum_from_settings(Some("4294967296"), Some("8589934592")).unwrap(),
            1 << 30
        );
        assert_eq!(maximum_from_settings(None, Some("800")).unwrap(), 100);
        for invalid in ["", "0", "01", "+1", " 1", "1 ", "1.0", "4294967297"] {
            assert!(maximum_from_settings(Some(invalid), None).is_err());
        }
        for invalid in ["0", "01", "+8", "7", "9223372036854775808"] {
            assert!(maximum_from_settings(None, Some(invalid)).is_err());
        }
    }

    #[test]
    fn target_reader_keeps_every_byte_and_rejects_growth_beyond_admission() {
        let bytes = b"complete target bytes";
        assert_eq!(
            read_bounded(&bytes[..], bytes.len() as u64, 0).unwrap(),
            bytes
        );
        assert!(read_bounded(&bytes[..], bytes.len() as u64 - 1, 0).is_err());
        assert!(read_bounded(&bytes[..], 1, 1).is_err());
        assert!(read_bounded(&bytes[..], u64::MAX, usize::MAX).is_err());
    }
}
