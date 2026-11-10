// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Canonical lowercase hexadecimal encoding for byte-oriented identities.

/// Encode every byte as exactly two lowercase hexadecimal digits.
///
/// Leading zero bytes are retained. This encoding is independent of the
/// concrete digest-output container and preserves existing content identities.
#[must_use]
pub fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for &byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// SHA-256 of ordinary serde_json output, streamed without a payload-sized buffer.
///
/// This uses the same serialization as `serde_json::to_vec`, including its
/// handling of nonfinite floats. Use [`crate::finite_json::digest`] when the
/// caller's contract requires rejecting nonfinite values.
pub fn sha256_json<T: serde::Serialize + ?Sized>(value: &T) -> Result<[u8; 32], serde_json::Error> {
    use sha2::{Digest, Sha256};
    struct Writer(Sha256);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Writer(Sha256::new());
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.0.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::{hex_lower, sha256_json};
    use sha2::{Digest, Sha256};

    #[test]
    fn lowercase_hex_preserves_leading_zeroes_and_every_byte() {
        assert_eq!(hex_lower(&[]), "");
        assert_eq!(hex_lower(&[0, 1, 15, 16, 128, 255]), "00010f1080ff");
        let bytes: Vec<_> = (0..=u8::MAX).collect();
        let expected: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(hex_lower(&bytes), expected);
    }

    #[test]
    fn sha256_known_vectors_keep_the_canonical_64_digit_encoding() {
        for (input, expected) in [
            (
                b"".as_slice(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc".as_slice(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
        ] {
            let encoded = hex_lower(&Sha256::digest(input));
            assert_eq!(encoded, expected);
            assert_eq!(encoded.len(), 64);
        }
    }

    #[test]
    fn streaming_json_keeps_the_exact_existing_serialization() {
        let value = serde_json::json!({
            "escaped": "line\nquote\"",
            "unicode": "\u{03c0}",
            "negative_zero": -0.0,
            "payload": "x".repeat(65537),
            "values": [null, true, 17, -23, 1.25e-20]
        });
        let expected: [u8; 32] = Sha256::digest(serde_json::to_vec(&value).unwrap()).into();
        assert_eq!(sha256_json(&value).unwrap(), expected);
        assert_eq!(
            sha256_json(&f64::NAN).unwrap(),
            <[u8; 32]>::from(Sha256::digest(b"null"))
        );
    }
}
