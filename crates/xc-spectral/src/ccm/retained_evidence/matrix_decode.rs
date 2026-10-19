// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
// All rights reserved. See LICENSE in the repository root.

//! Decode retained Tau entries without keeping their decimal strings alive.
//!
//! Both passes authenticate the complete logical stream. The
//! first checks the original payload shape and counts strings without collecting
//! them, so entries may precede precision or dimension. Only after all metadata
//! and workspace checks does the second pass allocate the final Float vector.
use super::{retained_matrix_shape, scalar};
use anyhow::{bail, Result};
use rug::Float;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use xc_cache::ContentDigest;
use xc_core::{MemoryReservation, RunResources};

#[derive(Deserialize)]
struct Metadata {
    schema_version: u32,
    lambda_squared: String,
    n_modes: usize,
    precision_bits: u32,
    #[serde(deserialize_with = "count_entries")]
    entries: usize,
}

pub(super) struct DecodedMatrix {
    pub cutoff: String,
    pub modes: usize,
    pub precision: u32,
    pub dimension: usize,
    pub entries: Vec<Float>,
    pub reservation: MemoryReservation,
}

const READ_BUFFER: usize = 16 * 1024;

// Charge before serde sees a chunk that can grow its reusable token buffer.
// Unknown fields retain their original acceptance domain: their length is a
// configured-resource decision, not a new fixed JSON-token limit. Keep the
// high-water charge across both passes, including metadata and scalar parsing.
struct ParserBudget<'a> {
    resources: &'a RunResources,
    maximum_token: u64,
    admitted_token: u64,
    reservations: Vec<MemoryReservation>,
}
impl ParserBudget<'_> {
    fn admit(&mut self, token: u64) -> io::Result<()> {
        if token <= self.admitted_token {
            return Ok(());
        }
        let capacity = token
            .checked_next_power_of_two()
            .unwrap_or(self.maximum_token)
            .min(self.maximum_token);
        let extra = capacity
            .checked_sub(self.admitted_token)
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(|| io::Error::other("retained matrix parser workspace overflow"))?;
        let reservation = self
            .resources
            .try_reserve_working(extra)
            .map_err(io::Error::other)?;
        self.reservations.push(reservation);
        self.admitted_token = capacity;
        Ok(())
    }
}

struct CheckedRead<'a, 'resources, R> {
    reader: &'a mut R,
    budget: &'a mut ParserBudget<'resources>,
    expected: u64,
    read: u64,
    hash: Sha256,
    quoted: bool,
    escaped: bool,
    token: u64,
}
impl<R: Read> Read for CheckedRead<'_, '_, R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        // At the declared end, one byte suffices to reject any trailing data.
        let remaining = self.expected.saturating_sub(self.read);
        let limit = output
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX))
            .max(1);
        let count = self.reader.read(&mut output[..limit])?;
        self.read = self
            .read
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("retained matrix length overflow"))?;
        if self.read > self.expected {
            return Err(io::Error::other("retained matrix exceeds manifest length"));
        }
        for &byte in &output[..count] {
            if self.quoted {
                self.token = self
                    .token
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("retained matrix token length overflow"))?;
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.quoted = false;
                    self.token = 0;
                }
            } else if byte == b'"' {
                self.quoted = true;
                self.token = 1;
            } else if byte.is_ascii_whitespace()
                || matches!(byte, b'[' | b']' | b'{' | b'}' | b',' | b':')
            {
                self.token = 0;
            } else {
                self.token = self
                    .token
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("retained matrix token length overflow"))?;
            }
            self.budget.admit(self.token)?;
        }
        self.hash.update(&output[..count]);
        Ok(count)
    }
}
impl<'a, 'resources, R: Read + Seek> CheckedRead<'a, 'resources, R> {
    fn new(
        reader: &'a mut R,
        expected: u64,
        budget: &'a mut ParserBudget<'resources>,
    ) -> Result<Self> {
        reader.seek(SeekFrom::Start(0))?;
        Ok(Self {
            reader,
            budget,
            expected,
            read: 0,
            hash: Sha256::new(),
            quoted: false,
            escaped: false,
            token: 0,
        })
    }
    fn verify(self, digest: &ContentDigest) -> Result<()> {
        if self.read != self.expected || format!("{:x}", self.hash.finalize()) != digest.0 {
            bail!("unapproved or invalid retained research source");
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn decode(bytes: &[u8]) -> Result<DecodedMatrix> {
    decode_reader(
        &mut io::Cursor::new(bytes),
        bytes.len() as u64,
        &ContentDigest::sha256(bytes),
        &super::retained_matrix_resources()?,
    )
}

pub(super) fn decode_reader<R: Read + Seek>(
    reader: &mut R,
    length: u64,
    digest: &ContentDigest,
    resources: &RunResources,
) -> Result<DecodedMatrix> {
    // The live parser holds at most one token scratch, metadata, conversion
    // temporaries, and this read buffer. The token high-water estimate depends
    // on the largest JSON token, not on the number of matrix entries.
    let _scratch = resources.try_reserve_working((4 * READ_BUFFER) as u64)?;
    let mut budget = ParserBudget {
        resources,
        maximum_token: length,
        admitted_token: 0,
        reservations: Vec::new(),
    };
    let checked = CheckedRead::new(reader, length, &mut budget)?;
    let mut buffered = BufReader::with_capacity(READ_BUFFER, checked);
    // Preserve derived duplicate/missing fields, ignored fields and positional
    // sequences. Authentication finishes before allocating any Float entries.
    let metadata: Metadata = serde_json::from_reader(&mut buffered)?;
    buffered.into_inner().verify(digest)?;
    let dimension =
        retained_matrix_shape(metadata.n_modes, metadata.precision_bits, metadata.entries)?;
    super::super::research::ExactCutoff::parse(&metadata.lambda_squared)?;
    if metadata.schema_version != 2 {
        bail!("invalid retained matrix shape");
    }
    let _conversion = resources
        .try_reserve_working(u64::from(metadata.precision_bits).div_ceil(8) * 32 + 65_536)?;
    let reservation = resources.try_reserve_retained(
        super::retained_matrix_bytes(metadata.precision_bits, metadata.entries)?
            .checked_add(metadata.lambda_squared.capacity() as u64)
            .ok_or_else(|| anyhow::anyhow!("retained matrix metadata size overflow"))?,
    )?;
    let checked = CheckedRead::new(reader, length, &mut budget)?;
    let mut buffered = BufReader::with_capacity(READ_BUFFER, checked);
    let mut deserializer = serde_json::Deserializer::from_reader(&mut buffered);
    let entries = MatrixEntries {
        count: metadata.entries,
        precision: metadata.precision_bits,
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    drop(deserializer);
    buffered.into_inner().verify(digest)?;
    Ok(DecodedMatrix {
        cutoff: metadata.lambda_squared,
        modes: metadata.n_modes,
        precision: metadata.precision_bits,
        dimension,
        entries,
        reservation,
    })
}

struct DiscardString;

impl<'de> Deserialize<'de> for DiscardString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StringVisitor;
        impl Visitor<'_> for StringVisitor {
            type Value = DiscardString;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a decimal string")
            }
            fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> {
                Ok(DiscardString)
            }
        }
        // The streaming deserializer uses reusable scratch, never a retained
        // String for each entry. The reader admits that scratch before growth.
        deserializer.deserialize_str(StringVisitor)
    }
}

fn count_entries<'de, D: Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
    struct CountVisitor;
    impl<'de> Visitor<'de> for CountVisitor {
        type Value = usize;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a sequence of decimal strings")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<usize, A::Error> {
            let mut count = 0usize;
            while sequence.next_element::<DiscardString>()?.is_some() {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| de::Error::custom("retained matrix entry count overflow"))?;
            }
            Ok(count)
        }
    }
    deserializer.deserialize_seq(CountVisitor)
}

#[derive(Clone, Copy)]
struct MatrixEntries {
    count: usize,
    precision: u32,
}

impl<'de> DeserializeSeed<'de> for MatrixEntries {
    type Value = Vec<Float>;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const FIELDS: &[&str] = &[
            "schema_version",
            "lambda_squared",
            "n_modes",
            "precision_bits",
            "entries",
        ];
        deserializer.deserialize_struct("Payload", FIELDS, self)
    }
}

impl<'de> Visitor<'de> for MatrixEntries {
    type Value = Vec<Float>;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a retained matrix payload")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier)]
        enum Field {
            #[serde(rename = "entries")]
            Entries,
            #[serde(other)]
            Other,
        }
        let mut entries = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Entries => {
                    if entries.is_some() {
                        return Err(de::Error::duplicate_field("entries"));
                    }
                    entries = Some(map.next_value_seed(EntrySequence(self))?);
                }
                Field::Other => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        entries.ok_or_else(|| de::Error::missing_field("entries"))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        for index in 0..4 {
            if sequence.next_element::<IgnoredAny>()?.is_none() {
                return Err(de::Error::invalid_length(index, &self));
            }
        }
        sequence
            .next_element_seed(EntrySequence(self))?
            .ok_or_else(|| de::Error::invalid_length(4, &self))
    }
}

struct EntrySequence(MatrixEntries);

impl<'de> DeserializeSeed<'de> for EntrySequence {
    type Value = Vec<Float>;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for EntrySequence {
    type Value = Vec<Float>;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the checked number of decimal strings")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        entries.try_reserve_exact(self.0.count).map_err(|error| {
            de::Error::custom(format!("retained matrix allocation failed: {error}"))
        })?;
        for index in 0..self.0.count {
            let entry = sequence
                .next_element_seed(EntryFloat(self.0.precision))?
                .ok_or_else(|| de::Error::invalid_length(index, &self))?;
            entries.push(entry);
        }
        if sequence.next_element::<IgnoredAny>()?.is_some() {
            return Err(de::Error::custom("retained matrix entry count changed"));
        }
        Ok(entries)
    }
}

struct EntryFloat(u32);

impl<'de> DeserializeSeed<'de> for EntryFloat {
    type Value = Float;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Float, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl Visitor<'_> for EntryFloat {
    type Value = Float;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a finite decimal string")
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Float, E> {
        scalar(value, self.0).map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccm::retained_evidence::{admits, RetainedMatrix};
    use serde_json::json;
    use std::collections::BTreeMap;
    use xc_cache::*;

    fn manifest(bytes: &[u8]) -> ArtifactManifest {
        let digest = ContentDigest::sha256(bytes);
        ArtifactManifest {
            schema_version: 1,
            key: ArtifactKey::new("ccm_tau_matrix", "typed-matrix-fixture", b"typed-matrix")
                .unwrap(),
            content_digest: digest.clone(),
            size_bytes: bytes.len() as u64,
            objects: vec![CacheObjectRef {
                content_digest: digest,
                size_bytes: bytes.len() as u64,
            }],
            created_unix_seconds: 0,
            producer_toolkit_version: ToolkitVersion::parse("0.16.0").unwrap(),
            minimum_reader_version: ToolkitVersion::parse("0.16.0").unwrap(),
            maximum_reader_version: None,
            quality: CacheQuality::Validated,
            visibility: CacheVisibility::Local,
            immutable: true,
            dependencies: vec![],
            tags: BTreeMap::new(),
            provenance_digest: None,
        }
    }

    // The pre-streaming reader is intentionally retained as the compatibility
    // oracle, including its derived-deserialization behavior and check order.
    fn reference(
        m: &ArtifactManifest,
        b: &[u8],
        allowed: &[ContentDigest],
    ) -> Result<RetainedMatrix<'static>> {
        admits(m, b, allowed, "ccm_tau_matrix")?;
        #[derive(Deserialize)]
        struct Payload {
            schema_version: u32,
            lambda_squared: String,
            n_modes: usize,
            precision_bits: u32,
            entries: Vec<String>,
        }
        let r: Payload = serde_json::from_slice(b)?;
        let n = retained_matrix_shape(r.n_modes, r.precision_bits, r.entries.len())?;
        crate::ccm::research::ExactCutoff::parse(&r.lambda_squared)?;
        if r.schema_version != 2 {
            bail!("invalid retained matrix shape");
        }
        let entries = r
            .entries
            .iter()
            .map(|v| scalar(v, r.precision_bits))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..n {
            for j in 0..i {
                if entries[i * n + j] != entries[j * n + i] {
                    bail!("retained matrix is not exactly symmetric");
                }
            }
        }
        Ok(RetainedMatrix {
            manifest: m.clone(),
            cutoff: r.lambda_squared,
            modes: r.n_modes,
            precision: r.precision_bits,
            entries: super::super::RetainedMatrixEntries::Owned(std::sync::Arc::new(
                super::super::OwnedRetainedMatrix {
                    _reservation: super::super::retained_matrix_resources()?.try_reserve_retained(
                        super::super::retained_matrix_bytes(r.precision_bits, entries.len())?,
                    )?,
                    entries,
                },
            )),
        })
    }

    fn check_success(bytes: &[u8]) {
        let source = manifest(bytes);
        let allowed = std::slice::from_ref(&source.content_digest);
        let expected = reference(&source, bytes, allowed).unwrap();
        let actual = RetainedMatrix::from_payload(&source, bytes, allowed).unwrap();
        assert_eq!(actual.cutoff, expected.cutoff);
        assert_eq!(actual.modes, expected.modes);
        assert_eq!(actual.precision, expected.precision);
        assert_eq!(
            serde_json::to_value(&actual.manifest).unwrap(),
            serde_json::to_value(&expected.manifest).unwrap()
        );
        assert_eq!(actual.entries.len(), expected.entries.len());
        for (actual, expected) in actual.entries.iter().zip(expected.entries.iter()) {
            assert_eq!(actual, expected);
            assert_eq!(actual.prec(), expected.prec());
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        }
    }

    fn check_failure(bytes: &[u8]) {
        let source = manifest(bytes);
        let allowed = std::slice::from_ref(&source.content_digest);
        assert!(reference(&source, bytes, allowed).is_err());
        assert!(RetainedMatrix::from_payload(&source, bytes, allowed).is_err());
    }

    fn fixture() -> serde_json::Value {
        json!({
            "schema_version": 2,
            "lambda_squared": "9",
            "n_modes": 1,
            "precision_bits": 128,
            "entries": ["-0", "1.25", "-2.5", "1.25", "3", "0", "-2.5", "-0", "4"]
        })
    }

    #[test]
    fn every_field_order_matches_the_reference() {
        fn permute(fields: &mut [&str], start: usize) {
            if start == fields.len() {
                check_success(format!("{{{}}}", fields.join(",")).as_bytes());
                return;
            }
            for index in start..fields.len() {
                fields.swap(start, index);
                permute(fields, start + 1);
                fields.swap(start, index);
            }
        }
        let mut fields = [
            r#""schema_version":2"#,
            r#""lambda_squared":"9""#,
            r#""n_modes":1"#,
            r#""precision_bits":128"#,
            r#""entries":["-0","1.25","-2.5","1.25","3","0","-2.5","-0","4"]"#,
        ];
        permute(&mut fields, 0);
    }

    #[test]
    fn escaped_entries_unknown_fields_and_positional_payloads_match() {
        check_success(
            br#"{
            "entries":["\u002d0","1\u002e25","-2.5","1.25","\u0033","0","-2.5","-0","4"],
            "unknown":{"nested":[null,false,"ignored",{"precision_bits":0}]},
            "unknown":true,
            "precision_bits":128,"lambda_squared":"\u0039","n_modes":1,"schema_version":2
        }"#,
        );
        check_success(br#"[2,"9",1,128,["-0","1.25","-2.5","1.25","3","0","-2.5","-0","4"]]"#);
        check_success(br#"{"schema_version":2,"lambda_squared":"9","n_modes":0,"precision_bits":128,"entri\u0065s":["1"]}"#);
    }

    #[test]
    fn numeric_rounding_precision_and_signed_zeros_match() {
        for precision in [64, 128, 3539, 1_000_000] {
            for value in [
                "0",
                "-0",
                "-0.000e+12",
                "+1.25",
                "-1.25e-30",
                "1.000000000000000000000000000000000000000001",
                "1e1000",
                "1e-1000",
            ] {
                check_success(
                    &serde_json::to_vec(&json!({
                        "schema_version": 2, "lambda_squared": "27/3", "n_modes": 0,
                        "precision_bits": precision, "entries": [value]
                    }))
                    .unwrap(),
                );
            }
        }
        // Symmetry is checked on parsed points, as before, and accepts zero
        // sign differences and decimal spellings that round to the same point.
        let mut value = fixture();
        value["precision_bits"] = json!(64);
        value["entries"][1] = json!("1.250000000000000000000000000000000000000001");
        check_success(&serde_json::to_vec(&value).unwrap());
    }

    #[test]
    fn missing_duplicate_and_invalid_metadata_match() {
        let fields = [
            "schema_version",
            "lambda_squared",
            "n_modes",
            "precision_bits",
            "entries",
        ];
        for field in fields {
            let mut value = fixture();
            value.as_object_mut().unwrap().remove(field);
            check_failure(&serde_json::to_vec(&value).unwrap());
            let value = fixture();
            let text = serde_json::to_string(&value).unwrap();
            check_failure(
                format!("{},\"{field}\":{}}}", &text[..text.len() - 1], value[field]).as_bytes(),
            );
        }
        for (field, replacement) in [
            ("schema_version", json!(1)),
            ("schema_version", json!(null)),
            ("schema_version", json!(-1)),
            ("lambda_squared", json!("NaN")),
            ("lambda_squared", json!("1/0")),
            ("lambda_squared", json!(false)),
            ("n_modes", json!(8193)),
            ("n_modes", json!(usize::MAX)),
            ("n_modes", json!(1.0)),
            ("n_modes", json!(-1)),
            ("precision_bits", json!(63)),
            ("precision_bits", json!(1_000_001)),
            ("precision_bits", json!(u64::MAX)),
            ("precision_bits", json!("128")),
            ("entries", json!(null)),
            ("entries", json!([])),
            ("entries", json!(["1"])),
        ] {
            let mut value = fixture();
            value[field] = replacement;
            check_failure(&serde_json::to_vec(&value).unwrap());
        }
    }

    #[test]
    fn malformed_json_invalid_late_entries_and_asymmetry_match() {
        for value in [
            json!("not-decimal"),
            json!("NaN"),
            json!("1e9999999999"),
            json!("1e-9999999999"),
            json!(1),
            json!(null),
            json!(["1"]),
        ] {
            let mut payload = fixture();
            payload["entries"][8] = value;
            check_failure(&serde_json::to_vec(&payload).unwrap());
        }
        let mut payload = fixture();
        payload["entries"][1] = json!("1.25001");
        check_failure(&serde_json::to_vec(&payload).unwrap());
        let mut payload = fixture();
        payload["entries"][8] = json!(format!("1.{}", "0".repeat(1_048_575)));
        check_failure(&serde_json::to_vec(&payload).unwrap());
        let valid = serde_json::to_string(&fixture()).unwrap();
        check_failure(format!("{valid} null").as_bytes());
        check_failure(&valid.as_bytes()[..valid.len() - 1]);
        for invalid in [
            r#"[2,"9",0,128,["1"],null]"#,
            r#"[2,"9",0,128]"#,
            r#"{"entries":["\q"]}"#,
        ] {
            check_failure(invalid.as_bytes());
        }
    }

    #[test]
    fn domain_checks_precede_float_allocation() {
        // A bad decimal must never be parsed before the checked shape or
        // precision. Entries are deliberately first in these payloads.
        for (metadata, expected) in [
            (r#""n_modes":1,"precision_bits":128"#, "shape or workspace"),
            (
                r#""n_modes":0,"precision_bits":63"#,
                "unsupported research precision",
            ),
        ] {
            let bytes = format!(
                "{{\"entries\":[\"not-decimal\"],\"schema_version\":2,\"lambda_squared\":\"9\",{metadata}}}"
            );
            let error = match decode(bytes.as_bytes()) {
                Ok(_) => panic!("invalid metadata was admitted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(expected), "{error}");
        }
        // Pure shape planning no longer embeds a workstation-sized memory cap.
        // Allocation admission below uses the configured shared run budget.
        assert!(retained_matrix_shape(500, 3386, 1001 * 1001).is_ok());
        assert!(retained_matrix_shape(2500, 3386, 5001 * 5001).is_ok());
        assert!(retained_matrix_shape(8193, 64, 16387 * 16387).is_err());
    }

    #[test]
    fn source_authentication_is_unchanged() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let check = |source: &ArtifactManifest, payload: &[u8], allowed: &[ContentDigest]| {
            assert!(reference(source, payload, allowed).is_err());
            assert!(RetainedMatrix::from_payload(source, payload, allowed).is_err());
        };
        check(&source, &bytes, &[]);
        check(
            &source,
            &bytes,
            &[ContentDigest::sha256(b"different source")],
        );
        let mut tampered = fixture();
        tampered["entries"][8] = json!("5");
        check(
            &source,
            &serde_json::to_vec(&tampered).unwrap(),
            std::slice::from_ref(&source.content_digest),
        );
        for variant in 0..7 {
            let mut altered = source.clone();
            match variant {
                0 => altered.immutable = false,
                1 => altered.key.kind = "ccm_weil_eigenpair".into(),
                2 => altered.size_bytes += 1,
                3 => altered.objects[0].size_bytes += 1,
                4 => altered.quality = CacheQuality::Staged,
                5 => altered.quality = CacheQuality::Published,
                6 => altered.quality = CacheQuality::Deprecated,
                _ => unreachable!(),
            }
            check(
                &altered,
                &bytes,
                std::slice::from_ref(&source.content_digest),
            );
        }
    }

    #[test]
    fn streaming_reader_rewinds_and_preserves_exact_values() {
        struct Chunks(io::Cursor<Vec<u8>>);
        impl Read for Chunks {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                let count = bytes.len().min(7);
                self.0.read(&mut bytes[..count])
            }
        }
        impl Seek for Chunks {
            fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
                self.0.seek(position)
            }
        }
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let allowed = std::slice::from_ref(&source.content_digest);
        let expected = reference(&source, &bytes, allowed).unwrap();
        let mut reader = Chunks(io::Cursor::new(bytes));
        reader.seek(SeekFrom::Start(11)).unwrap();
        let actual = RetainedMatrix::from_reader(&source, &mut reader, allowed).unwrap();
        for (a, b) in actual.entries.iter().zip(expected.entries.iter()) {
            assert_eq!(a, b);
            assert_eq!(a.prec(), b.prec());
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
    }

    #[test]
    fn owned_matrix_clones_share_allocation_and_keep_reservation() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let resources = RunResources::new(4 << 20).unwrap();
        let matrix = {
            let _scope = resources.enter();
            RetainedMatrix::from_payload(
                &source,
                &bytes,
                std::slice::from_ref(&source.content_digest),
            )
            .unwrap()
        };
        let charged = resources.usage().retained_bytes;
        assert!(charged >= super::super::retained_matrix_bytes(128, 9).unwrap());
        assert_eq!(resources.usage().working_bytes, 0);
        let copy = matrix.clone();
        let owned = matrix.to_owned_entries().unwrap();
        assert_eq!(matrix.entries.as_ptr(), copy.entries.as_ptr());
        assert_eq!(matrix.entries.as_ptr(), owned.entries.as_ptr());
        assert_eq!(resources.usage().retained_bytes, charged);
        drop(matrix);
        drop(copy);
        assert_eq!(resources.usage().retained_bytes, charged);
        drop(owned);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn borrowed_matrix_copy_admits_before_allocating_and_releases() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let entries = vec![Float::with_val(128, 1); 9];
        let matrix =
            RetainedMatrix::from_admitted_runtime(source, "9".into(), 1, 128, &entries).unwrap();
        let small = RunResources::new(1).unwrap();
        {
            let _scope = small.enter();
            assert!(matrix.to_owned_entries().is_err());
        }
        assert_eq!(small.usage().retained_bytes, 0);
        let resources = RunResources::new(4 << 20).unwrap();
        let _scope = resources.enter();
        let owned = matrix.to_owned_entries().unwrap();
        assert_ne!(owned.entries.as_ptr(), entries.as_ptr());
        assert!(resources.usage().retained_bytes > 0);
        drop(owned);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn shared_pressure_refuses_without_retaining_a_failure() {
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let resources = RunResources::new(4 << 20).unwrap();
        let occupied = resources.try_reserve_retained((4 << 20) - 1).unwrap();
        assert!(RetainedMatrix::from_reader_with_resources(
            &source,
            &mut io::Cursor::new(&bytes),
            std::slice::from_ref(&source.content_digest),
            &resources
        )
        .is_err());
        assert_eq!(resources.usage().retained_bytes, (4 << 20) - 1);
        assert_eq!(resources.usage().working_bytes, 0);
        drop(occupied);
        let matrix = RetainedMatrix::from_reader_with_resources(
            &source,
            &mut io::Cursor::new(&bytes),
            std::slice::from_ref(&source.content_digest),
            &resources,
        )
        .unwrap();
        drop(matrix);
        assert_eq!(
            resources.usage().retained_bytes + resources.usage().working_bytes,
            0
        );
    }

    #[test]
    fn configured_large_matrix_planning_has_no_eight_gib_ceiling() {
        assert_eq!(
            retained_matrix_shape(2500, 3386, 5001 * 5001).unwrap(),
            5001
        );
        let bytes = super::super::retained_matrix_bytes(3386, 5001 * 5001).unwrap();
        assert!(bytes > 8 << 30);
        let resources = RunResources::new(bytes).unwrap();
        let guard = resources.try_reserve_retained(bytes).unwrap();
        assert_eq!(resources.usage().retained_bytes, bytes);
        drop(guard);
        assert!(RunResources::new(bytes - 1)
            .unwrap()
            .try_reserve_retained(bytes)
            .is_err());
        // Planning and cooperative reservation only: no large allocation.
    }

    #[test]
    fn ignored_large_json_tokens_use_declared_budget_not_a_fixed_cap() {
        let mut value = fixture();
        value["ignored"] = serde_json::Value::String("x".repeat(7 << 20));
        let bytes = serde_json::to_vec(&value).unwrap();
        let source = manifest(&bytes);
        let allowed = std::slice::from_ref(&source.content_digest);
        let small = RunResources::new(1 << 20).unwrap();
        assert!(RetainedMatrix::from_reader_with_resources(
            &source,
            &mut io::Cursor::new(&bytes),
            allowed,
            &small,
        )
        .is_err());
        assert_eq!(
            small.usage().working_bytes + small.usage().retained_bytes,
            0
        );
        let resources = RunResources::new(512 << 20).unwrap();
        let actual = RetainedMatrix::from_reader_with_resources(
            &source,
            &mut io::Cursor::new(&bytes),
            allowed,
            &resources,
        )
        .unwrap();
        assert_eq!(resources.usage().working_bytes, 0);
        let expected = reference(&source, &bytes, allowed).unwrap();
        for (a, b) in actual.entries.iter().zip(expected.entries.iter()) {
            assert_eq!(a, b);
            assert_eq!(a.prec(), b.prec());
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
        drop(actual);
        assert_eq!(resources.usage().retained_bytes, 0);
    }

    #[test]
    fn stream_mutation_and_wrong_lengths_never_return_an_owned_matrix() {
        struct Mutable(io::Cursor<Vec<u8>>, usize);
        impl Read for Mutable {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.0.read(bytes)
            }
        }
        impl Seek for Mutable {
            fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
                if position == SeekFrom::Start(0) {
                    self.1 += 1;
                    if self.1 == 2 {
                        let bytes = self.0.get_mut();
                        let at = bytes.iter().position(|b| *b == b'4').unwrap();
                        bytes[at] = b'5';
                    }
                }
                self.0.seek(position)
            }
        }
        let bytes = serde_json::to_vec(&fixture()).unwrap();
        let source = manifest(&bytes);
        let allowed = std::slice::from_ref(&source.content_digest);
        let resources = RunResources::new(4 << 20).unwrap();
        let mut changed = Mutable(io::Cursor::new(bytes.clone()), 0);
        assert!(RetainedMatrix::from_reader_with_resources(
            &source,
            &mut changed,
            allowed,
            &resources
        )
        .is_err());
        for invalid in [
            bytes[..bytes.len() - 1].to_vec(),
            [bytes.as_slice(), b" "].concat(),
        ] {
            assert!(RetainedMatrix::from_reader_with_resources(
                &source,
                &mut io::Cursor::new(invalid),
                allowed,
                &resources
            )
            .is_err());
        }
        assert_eq!(
            resources.usage().retained_bytes + resources.usage().working_bytes,
            0
        );
    }
}
