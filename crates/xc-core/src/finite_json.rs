//! JSON encoding that rejects nonfinite floats before serde_json can replace them with null.
use serde::ser::*;
use serde::{Serialize, Serializer};

struct Finite<'a, T: ?Sized>(&'a T);
impl<T: Serialize + ?Sized> Serialize for Finite<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(FiniteSerializer(serializer))
    }
}
struct FiniteSerializer<S>(S);
struct FiniteCompound<S>(S);

/// Serialize JSON while rejecting nonfinite numbers at every visited value.
pub fn to_vec<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&Finite(value))
}

/// SHA-256 and length of exactly the bytes [`to_vec`] would return, streamed
/// so that the encoding is never held in memory.
pub fn digest<T: Serialize + ?Sized>(value: &T) -> Result<([u8; 32], u64), serde_json::Error> {
    let mut writer = DigestWriter::default();
    serde_json::to_writer(&mut writer, &Finite(value))?;
    Ok(writer.finish())
}

/// SHA-256 and length of `bytes`, comparable with [`digest`].
pub fn digest_bytes(bytes: &[u8]) -> ([u8; 32], u64) {
    let mut writer = DigestWriter::default();
    std::io::Write::write_all(&mut writer, bytes).expect("hashing never fails");
    writer.finish()
}

#[derive(Default)]
struct DigestWriter {
    hasher: sha2::Sha256,
    bytes: u64,
}
impl DigestWriter {
    fn finish(self) -> ([u8; 32], u64) {
        use sha2::Digest;
        (self.hasher.finalize().into(), self.bytes)
    }
}
impl std::io::Write for DigestWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        use sha2::Digest;
        self.hasher.update(buffer);
        self.bytes += buffer.len() as u64;
        Ok(buffer.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Convert a typed value without silently replacing nonfinite numbers with null.
pub fn to_value<T: Serialize + ?Sized>(value: &T) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::to_value(Finite(value))
}

impl<S: Serializer> Serializer for FiniteSerializer<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = FiniteCompound<S::SerializeSeq>;
    type SerializeTuple = FiniteCompound<S::SerializeTuple>;
    type SerializeTupleStruct = FiniteCompound<S::SerializeTupleStruct>;
    type SerializeTupleVariant = FiniteCompound<S::SerializeTupleVariant>;
    type SerializeMap = FiniteCompound<S::SerializeMap>;
    type SerializeStruct = FiniteCompound<S::SerializeStruct>;
    type SerializeStructVariant = FiniteCompound<S::SerializeStructVariant>;
    fn serialize_bool(self, value: bool) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bool(value)
    }
    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i8(value)
    }
    fn serialize_i16(self, value: i16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i16(value)
    }
    fn serialize_i32(self, value: i32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i32(value)
    }
    fn serialize_i64(self, value: i64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i64(value)
    }
    fn serialize_i128(self, value: i128) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i128(value)
    }
    fn serialize_u8(self, value: u8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u8(value)
    }
    fn serialize_u16(self, value: u16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u16(value)
    }
    fn serialize_u32(self, value: u32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u32(value)
    }
    fn serialize_u64(self, value: u64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u64(value)
    }
    fn serialize_u128(self, value: u128) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u128(value)
    }
    fn serialize_char(self, value: char) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_char(value)
    }
    fn serialize_str(self, value: &str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_str(value)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bytes(value)
    }
    fn serialize_f32(self, value: f32) -> Result<Self::Ok, Self::Error> {
        if !value.is_finite() {
            return Err(S::Error::custom(
                "nonfinite float cannot be retained as JSON",
            ));
        }
        self.0.serialize_f32(value)
    }
    fn serialize_f64(self, value: f64) -> Result<Self::Ok, Self::Error> {
        if !value.is_finite() {
            return Err(S::Error::custom(
                "nonfinite float cannot be retained as JSON",
            ));
        }
        self.0.serialize_f64(value)
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_none()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_some(&Finite(value))
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit()
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_struct(name)
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_variant(name, index, variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_newtype_struct(name, &Finite(value))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0
            .serialize_newtype_variant(name, index, variant, &Finite(value))
    }
    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.0.serialize_seq(len).map(FiniteCompound)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.0.serialize_tuple(len).map(FiniteCompound)
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.0.serialize_tuple_struct(name, len).map(FiniteCompound)
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.0
            .serialize_tuple_variant(name, index, variant, len)
            .map(FiniteCompound)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.0.serialize_map(len).map(FiniteCompound)
    }
    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.0.serialize_struct(name, len).map(FiniteCompound)
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.0
            .serialize_struct_variant(name, index, variant, len)
            .map(FiniteCompound)
    }
}
impl<S: SerializeSeq> SerializeSeq for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_element(&Finite(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeTuple> SerializeTuple for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_element(&Finite(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeTupleStruct> SerializeTupleStruct for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_field(&Finite(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeTupleVariant> SerializeTupleVariant for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_field(&Finite(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeMap> SerializeMap for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_key(&Finite(value))
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_value(&Finite(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeStruct> SerializeStruct for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.0.serialize_field(key, &Finite(value))
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
        self.0.skip_field(key)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}
impl<S: SerializeStructVariant> SerializeStructVariant for FiniteCompound<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.0.serialize_field(key, &Finite(value))
    }
    fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
        self.0.skip_field(key)
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

// ---------------------------------------------------------------------------
// Streamed canonical JSON.
//
// `write_canonical` writes exactly the text of
// `config_resolution::canonical_json(&to_value(value)?)` (object keys sorted,
// array order kept, `serde_json::Value` conversion rules) and applies the
// secret screen of `validate_secret_free` in the same traversal order, without
// building the `Value` tree. Arrays are streamed element by element. An object
// is written key by key: one pass collects its keys; keys that already arrive
// strictly ascending are then written in one pass, and otherwise each key is
// written by a pass that serializes only the entry with that key, so no field
// is buffered.
//
// Precondition: the value must serialize repeatably, including its keys,
// values, variants and traversal order, and must have unique object keys.
// Digest equivalence with `research_digest` is guaranteed only within that
// domain. Derived serialization alone does not ensure it when a field has a
// stateful custom serializer or the data changes concurrently. The writer
// rejects repeated object keys (where `to_value` would keep the last value) and
// some detected shape changes, and pairs each value with its own key; it does
// not verify repeatability or guarantee rejection of every violation.

/// Error of the streamed canonical writer (serialization or secret screen).
#[derive(Debug)]
pub struct CanonicalError(pub String);
impl std::fmt::Display for CanonicalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for CanonicalError {}
impl serde::ser::Error for CanonicalError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

/// Where canonical text goes, how a secret is reported, and which top-level
/// object fields are left out.
pub(crate) struct CanonicalSink<'a> {
    pub(crate) write: &'a mut dyn FnMut(&str),
    pub(crate) context: &'a str,
    pub(crate) omit: &'a [&'a str],
}

/// The `serde_json::Value` of `value` without the top-level object fields named
/// in `omit`, built from streamed canonical text so the omitted fields are
/// never materialized. The kept part is secret-screened as by
/// [`crate::validate_secret_free`] under `context`; the omitted fields are not.
/// With a nonempty `omit`, `value` must serialize as a struct or map (not an
/// enum variant). The precondition of [`crate::research_digest_streamed`]
/// applies: the value must serialize repeatably with unique object keys.
pub fn to_value_omitting<T: Serialize + ?Sized>(
    value: &T,
    omit: &[&str],
    context: &str,
) -> Result<serde_json::Value, CanonicalError> {
    let mut text = String::new();
    let mut sink = CanonicalSink {
        write: &mut |part| text.push_str(part),
        context,
        omit,
    };
    write_canonical(value, &mut "$".to_string(), &mut sink)?;
    let view: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| CanonicalError(error.to_string()))?;
    if !omit.is_empty() && !view.is_object() {
        return Err(CanonicalError(
            "field omission requires a top-level struct or map".into(),
        ));
    }
    Ok(view)
}

/// Write the canonical JSON of `value` (see the section comment).
pub(crate) fn write_canonical<T: Serialize + ?Sized>(
    value: &T,
    path: &mut String,
    sink: &mut CanonicalSink<'_>,
) -> Result<(), CanonicalError> {
    match value.serialize(Probe)? {
        Shape::Text(text) => (sink.write)(&text),
        Shape::Str(text) => {
            if crate::secret::secret_marker(&text).is_some() {
                return Err(CanonicalError(format!(
                    "{} contains credential-shaped material at {path}",
                    sink.context
                )));
            }
            (sink.write)(&json_string(&text)?);
        }
        Shape::Wrapped => value.serialize(Passthrough { path, sink })?,
        Shape::Array => value.serialize(Elements { path, sink })?,
        Shape::Object(keys) => {
            write_object(&keys, path, sink, |target, path, sink| {
                value.serialize(Field {
                    target,
                    keys: &keys,
                    path,
                    sink,
                })
            })?;
        }
        Shape::Variant(variant, inner) => {
            if path == "$" && !sink.omit.is_empty() {
                return Err(CanonicalError(
                    "field omission requires a top-level struct or map, not an enum variant".into(),
                ));
            }
            // `Value` represents a data-carrying variant as {variant: inner}.
            check_key(&variant, path, sink)?;
            (sink.write)("{");
            (sink.write)(&json_string(&variant)?);
            (sink.write)(":");
            let saved = path.len();
            path.push('.');
            path.push_str(&variant);
            match inner {
                VariantInner::Newtype => value.serialize(Passthrough { path, sink })?,
                VariantInner::Tuple => value.serialize(Elements { path, sink })?,
                VariantInner::Struct(keys) => {
                    write_object(&keys, path, sink, |target, path, sink| {
                        value.serialize(Field {
                            target,
                            keys: &keys,
                            path,
                            sink,
                        })
                    })?
                }
            }
            path.truncate(saved);
            (sink.write)("}");
        }
    }
    Ok(())
}

fn json_string(text: &str) -> Result<String, CanonicalError> {
    serde_json::to_string(text).map_err(|error| CanonicalError(error.to_string()))
}

fn check_key(key: &str, path: &str, sink: &CanonicalSink<'_>) -> Result<(), CanonicalError> {
    if crate::secret::sensitive_key(key) {
        return Err(CanonicalError(format!(
            "{} contains a credential-bearing field at {path}.{key}",
            sink.context
        )));
    }
    Ok(())
}

/// Keys in `Value` object order (sorted), each written followed by its field.
/// Keys already strictly ascending (a sorted map) are written in one pass
/// (`field(None, ..)`); otherwise one pass per key writes only the entry with
/// that key (`field(Some(key), ..)`). A repeated key is rejected.
fn write_object(
    keys: &[String],
    path: &mut String,
    sink: &mut CanonicalSink<'_>,
    mut field: impl FnMut(
        Option<&str>,
        &mut String,
        &mut CanonicalSink<'_>,
    ) -> Result<(), CanonicalError>,
) -> Result<(), CanonicalError> {
    let omitting = path == "$" && !sink.omit.is_empty();
    if !omitting && keys.windows(2).all(|pair| pair[0] < pair[1]) {
        (sink.write)("{");
        field(None, path, sink)?;
        (sink.write)("}");
        return Ok(());
    }
    let mut sorted = keys.iter().map(String::as_str).collect::<Vec<_>>();
    sorted.sort_unstable();
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(CanonicalError(format!(
            "repeated object key {:?} at {path} is not supported by the streamed writer",
            pair[0]
        )));
    }
    (sink.write)("{");
    let mut first = true;
    for key in sorted {
        if omitting && sink.omit.contains(&key) {
            continue;
        }
        if !first {
            (sink.write)(",");
        }
        first = false;
        check_key(key, path, sink)?;
        (sink.write)(&json_string(key)?);
        (sink.write)(":");
        let saved = path.len();
        path.push('.');
        path.push_str(key);
        field(Some(key), path, sink)?;
        path.truncate(saved);
    }
    (sink.write)("}");
    Ok(())
}

/// The text `serde_json::Value` conversion gives a map key, taken from that
/// conversion itself (through [`to_value`], so with its finite rule).
fn map_key_text<T: Serialize + ?Sized>(key: &T) -> Result<String, CanonicalError> {
    struct OneKey<'a, T: ?Sized>(&'a T);
    impl<T: Serialize + ?Sized> Serialize for OneKey<'_, T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(1))?;
            map.serialize_entry(self.0, &())?;
            map.end()
        }
    }
    match to_value(&OneKey(key)).map_err(|error| CanonicalError(error.to_string()))? {
        serde_json::Value::Object(map) if map.len() == 1 => Ok(map
            .into_iter()
            .next()
            .map(|(key, _)| key)
            .unwrap_or_default()),
        _ => Err(CanonicalError(
            "map key conversion did not yield exactly one key".into(),
        )),
    }
}

enum VariantInner {
    Newtype,
    Tuple,
    Struct(Vec<String>),
}

/// The `Value` kind a serialization produces; containers are not traversed.
enum Shape {
    /// Canonical text of null, a boolean, a number or a byte array.
    Text(String),
    Str(String),
    /// `Some` or a newtype struct: the inner value is the value.
    Wrapped,
    Array,
    Object(Vec<String>),
    Variant(String, VariantInner),
}

fn number_text(value: serde_json::Number) -> Shape {
    Shape::Text(value.to_string())
}

fn float_text(value: f64) -> Result<Shape, CanonicalError> {
    if !value.is_finite() {
        return Err(CanonicalError(
            "nonfinite float cannot be retained as JSON".into(),
        ));
    }
    serde_json::Number::from_f64(value)
        .map(number_text)
        .ok_or_else(|| CanonicalError("nonfinite float cannot be retained as JSON".into()))
}

struct Probe;
struct ProbeSeq(Shape);
struct ProbeKeys(Vec<String>, Option<String>);

impl serde::Serializer for Probe {
    type Ok = Shape;
    type Error = CanonicalError;
    type SerializeSeq = ProbeSeq;
    type SerializeTuple = ProbeSeq;
    type SerializeTupleStruct = ProbeSeq;
    type SerializeTupleVariant = ProbeSeq;
    type SerializeMap = ProbeKeys;
    type SerializeStruct = ProbeKeys;
    type SerializeStructVariant = ProbeKeys;
    fn serialize_bool(self, value: bool) -> Result<Shape, CanonicalError> {
        Ok(Shape::Text(if value { "true" } else { "false" }.into()))
    }
    fn serialize_i8(self, value: i8) -> Result<Shape, CanonicalError> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i16(self, value: i16) -> Result<Shape, CanonicalError> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i32(self, value: i32) -> Result<Shape, CanonicalError> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i64(self, value: i64) -> Result<Shape, CanonicalError> {
        Ok(number_text(value.into()))
    }
    fn serialize_i128(self, value: i128) -> Result<Shape, CanonicalError> {
        if let Ok(value) = u64::try_from(value) {
            Ok(number_text(value.into()))
        } else if let Ok(value) = i64::try_from(value) {
            Ok(number_text(value.into()))
        } else {
            Err(CanonicalError("number out of range".into()))
        }
    }
    fn serialize_u8(self, value: u8) -> Result<Shape, CanonicalError> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u16(self, value: u16) -> Result<Shape, CanonicalError> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u32(self, value: u32) -> Result<Shape, CanonicalError> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u64(self, value: u64) -> Result<Shape, CanonicalError> {
        Ok(number_text(value.into()))
    }
    fn serialize_u128(self, value: u128) -> Result<Shape, CanonicalError> {
        u64::try_from(value)
            .map(|value| number_text(value.into()))
            .map_err(|_| CanonicalError("number out of range".into()))
    }
    fn serialize_f32(self, value: f32) -> Result<Shape, CanonicalError> {
        float_text(f64::from(value))
    }
    fn serialize_f64(self, value: f64) -> Result<Shape, CanonicalError> {
        float_text(value)
    }
    fn serialize_char(self, value: char) -> Result<Shape, CanonicalError> {
        Ok(Shape::Str(value.to_string()))
    }
    fn serialize_str(self, value: &str) -> Result<Shape, CanonicalError> {
        Ok(Shape::Str(value.to_owned()))
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<Shape, CanonicalError> {
        let items = value.iter().map(u8::to_string).collect::<Vec<_>>();
        Ok(Shape::Text(format!("[{}]", items.join(","))))
    }
    fn serialize_none(self) -> Result<Shape, CanonicalError> {
        Ok(Shape::Text("null".into()))
    }
    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<Shape, CanonicalError> {
        Ok(Shape::Wrapped)
    }
    fn serialize_unit(self) -> Result<Shape, CanonicalError> {
        Ok(Shape::Text("null".into()))
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Shape, CanonicalError> {
        Ok(Shape::Text("null".into()))
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Shape, CanonicalError> {
        Ok(Shape::Str(variant.into()))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<Shape, CanonicalError> {
        Ok(Shape::Wrapped)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: &T,
    ) -> Result<Shape, CanonicalError> {
        Ok(Shape::Variant(variant.into(), VariantInner::Newtype))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<ProbeSeq, CanonicalError> {
        Ok(ProbeSeq(Shape::Array))
    }
    fn serialize_tuple(self, _: usize) -> Result<ProbeSeq, CanonicalError> {
        Ok(ProbeSeq(Shape::Array))
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<ProbeSeq, CanonicalError> {
        Ok(ProbeSeq(Shape::Array))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<ProbeSeq, CanonicalError> {
        Ok(ProbeSeq(Shape::Variant(
            variant.into(),
            VariantInner::Tuple,
        )))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<ProbeKeys, CanonicalError> {
        Ok(ProbeKeys(Vec::new(), None))
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<ProbeKeys, CanonicalError> {
        Ok(ProbeKeys(Vec::new(), None))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<ProbeKeys, CanonicalError> {
        Ok(ProbeKeys(Vec::new(), Some(variant.into())))
    }
}

impl SerializeSeq for ProbeSeq {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), CanonicalError> {
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(self.0)
    }
}
impl SerializeTuple for ProbeSeq {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), CanonicalError> {
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(self.0)
    }
}
impl SerializeTupleStruct for ProbeSeq {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), CanonicalError> {
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(self.0)
    }
}
impl SerializeTupleVariant for ProbeSeq {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), CanonicalError> {
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(self.0)
    }
}
impl SerializeMap for ProbeKeys {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), CanonicalError> {
        self.0.push(map_key_text(key)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, _: &T) -> Result<(), CanonicalError> {
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(Shape::Object(self.0))
    }
}
impl SerializeStruct for ProbeKeys {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        self.0.push(key.into());
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        Ok(Shape::Object(self.0))
    }
}
impl SerializeStructVariant for ProbeKeys {
    type Ok = Shape;
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        self.0.push(key.into());
        Ok(())
    }
    fn end(self) -> Result<Shape, CanonicalError> {
        let variant = self.1.unwrap_or_default();
        Ok(Shape::Variant(variant, VariantInner::Struct(self.0)))
    }
}

/// Serializer accepting only the shapes reached through one probe kind; any
/// other call means the value serialized differently on a second pass.
fn unstable<T>() -> Result<T, CanonicalError> {
    Err(unstable_error())
}
fn unstable_error() -> CanonicalError {
    CanonicalError("value serialized differently on a repeated pass".into())
}

/// Writes the inner value of `Some`, a newtype struct or a newtype variant.
struct Passthrough<'p, 's, 'w> {
    path: &'p mut String,
    sink: &'s mut CanonicalSink<'w>,
}

/// Streams the elements of a sequence, tuple or tuple variant as an array.
struct Elements<'p, 's, 'w> {
    path: &'p mut String,
    sink: &'s mut CanonicalSink<'w>,
}
struct ElementWriter<'p, 's, 'w> {
    path: &'p mut String,
    sink: &'s mut CanonicalSink<'w>,
    index: usize,
}
impl ElementWriter<'_, '_, '_> {
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalError> {
        if self.index > 0 {
            (self.sink.write)(",");
        }
        let saved = self.path.len();
        self.path.push_str(&format!("[{}]", self.index));
        write_canonical(value, self.path, self.sink)?;
        self.path.truncate(saved);
        self.index += 1;
        Ok(())
    }
    fn finish(self) -> Result<(), CanonicalError> {
        (self.sink.write)("]");
        Ok(())
    }
}

/// Writes the entry whose key is `target` of a struct, map or struct variant,
/// or with no target every entry, each after its key (`keys` are the probed
/// keys in arrival order, which the single pass must reproduce).
struct Field<'k, 'p, 's, 'w> {
    target: Option<&'k str>,
    keys: &'k [String],
    path: &'p mut String,
    sink: &'s mut CanonicalSink<'w>,
}
struct FieldWriter<'k, 'p, 's, 'w> {
    target: Option<&'k str>,
    keys: &'k [String],
    index: usize,
    found: bool,
    pending_key: Option<String>,
    path: &'p mut String,
    sink: &'s mut CanonicalSink<'w>,
}
impl FieldWriter<'_, '_, '_, '_> {
    fn field<T: Serialize + ?Sized>(&mut self, key: &str, value: &T) -> Result<(), CanonicalError> {
        match self.target {
            Some(target) => {
                if key == target {
                    if self.found {
                        return Err(unstable_error());
                    }
                    self.found = true;
                    write_canonical(value, self.path, self.sink)?;
                }
            }
            None => {
                if self.keys.get(self.index).map(String::as_str) != Some(key) {
                    return Err(unstable_error());
                }
                if self.index > 0 {
                    (self.sink.write)(",");
                }
                check_key(key, self.path, self.sink)?;
                (self.sink.write)(&json_string(key)?);
                (self.sink.write)(":");
                let saved = self.path.len();
                self.path.push('.');
                self.path.push_str(key);
                write_canonical(value, self.path, self.sink)?;
                self.path.truncate(saved);
            }
        }
        self.index += 1;
        Ok(())
    }
    fn finish(self) -> Result<(), CanonicalError> {
        if self.index != self.keys.len() || (self.target.is_some() && !self.found) {
            return Err(unstable_error());
        }
        Ok(())
    }
}

macro_rules! only_compounds {
    () => {
        fn serialize_bool(self, _: bool) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_i8(self, _: i8) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_i16(self, _: i16) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_i32(self, _: i32) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_i64(self, _: i64) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_i128(self, _: i128) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_u8(self, _: u8) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_u16(self, _: u16) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_u32(self, _: u32) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_u64(self, _: u64) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_u128(self, _: u128) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_f32(self, _: f32) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_f64(self, _: f64) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_char(self, _: char) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_str(self, _: &str) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_bytes(self, _: &[u8]) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_none(self) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_unit(self) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_unit_struct(self, _: &'static str) -> Result<(), CanonicalError> {
            unstable()
        }
        fn serialize_unit_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
        ) -> Result<(), CanonicalError> {
            unstable()
        }
    };
}

impl<'p, 's, 'w> serde::Serializer for Passthrough<'p, 's, 'w> {
    type Ok = ();
    type Error = CanonicalError;
    type SerializeSeq = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTuple = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTupleStruct = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTupleVariant = serde::ser::Impossible<(), CanonicalError>;
    type SerializeMap = serde::ser::Impossible<(), CanonicalError>;
    type SerializeStruct = serde::ser::Impossible<(), CanonicalError>;
    type SerializeStructVariant = serde::ser::Impossible<(), CanonicalError>;
    only_compounds!();
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), CanonicalError> {
        write_canonical(value, self.path, self.sink)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), CanonicalError> {
        write_canonical(value, self.path, self.sink)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), CanonicalError> {
        write_canonical(value, self.path, self.sink)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, CanonicalError> {
        unstable()
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, CanonicalError> {
        unstable()
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, CanonicalError> {
        unstable()
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, CanonicalError> {
        unstable()
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, CanonicalError> {
        unstable()
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, CanonicalError> {
        unstable()
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, CanonicalError> {
        unstable()
    }
}

impl<'p, 's, 'w> serde::Serializer for Elements<'p, 's, 'w> {
    type Ok = ();
    type Error = CanonicalError;
    type SerializeSeq = ElementWriter<'p, 's, 'w>;
    type SerializeTuple = ElementWriter<'p, 's, 'w>;
    type SerializeTupleStruct = ElementWriter<'p, 's, 'w>;
    type SerializeTupleVariant = ElementWriter<'p, 's, 'w>;
    type SerializeMap = serde::ser::Impossible<(), CanonicalError>;
    type SerializeStruct = serde::ser::Impossible<(), CanonicalError>;
    type SerializeStructVariant = serde::ser::Impossible<(), CanonicalError>;
    only_compounds!();
    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, CanonicalError> {
        (self.sink.write)("[");
        Ok(ElementWriter {
            path: self.path,
            sink: self.sink,
            index: 0,
        })
    }
    fn serialize_tuple(self, length: usize) -> Result<Self::SerializeTuple, CanonicalError> {
        self.serialize_seq(Some(length))
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleStruct, CanonicalError> {
        self.serialize_seq(Some(length))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        length: usize,
    ) -> Result<Self::SerializeTupleVariant, CanonicalError> {
        self.serialize_seq(Some(length))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, CanonicalError> {
        unstable()
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, CanonicalError> {
        unstable()
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, CanonicalError> {
        unstable()
    }
}

impl SerializeSeq for ElementWriter<'_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), CanonicalError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}
impl SerializeTuple for ElementWriter<'_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), CanonicalError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}
impl SerializeTupleStruct for ElementWriter<'_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}
impl SerializeTupleVariant for ElementWriter<'_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}

impl<'k, 'p, 's, 'w> serde::Serializer for Field<'k, 'p, 's, 'w> {
    type Ok = ();
    type Error = CanonicalError;
    type SerializeSeq = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTuple = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTupleStruct = serde::ser::Impossible<(), CanonicalError>;
    type SerializeTupleVariant = serde::ser::Impossible<(), CanonicalError>;
    type SerializeMap = FieldWriter<'k, 'p, 's, 'w>;
    type SerializeStruct = FieldWriter<'k, 'p, 's, 'w>;
    type SerializeStructVariant = FieldWriter<'k, 'p, 's, 'w>;
    only_compounds!();
    fn serialize_some<T: Serialize + ?Sized>(self, _: &T) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: &T,
    ) -> Result<(), CanonicalError> {
        unstable()
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self::SerializeSeq, CanonicalError> {
        unstable()
    }
    fn serialize_tuple(self, _: usize) -> Result<Self::SerializeTuple, CanonicalError> {
        unstable()
    }
    fn serialize_tuple_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleStruct, CanonicalError> {
        unstable()
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, CanonicalError> {
        unstable()
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, CanonicalError> {
        Ok(FieldWriter {
            target: self.target,
            keys: self.keys,
            index: 0,
            found: false,
            pending_key: None,
            path: self.path,
            sink: self.sink,
        })
    }
    fn serialize_struct(
        self,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStruct, CanonicalError> {
        self.serialize_map(None)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, CanonicalError> {
        self.serialize_map(None)
    }
}

impl SerializeMap for FieldWriter<'_, '_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), CanonicalError> {
        self.pending_key = Some(map_key_text(key)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonicalError> {
        let key = self.pending_key.take().ok_or_else(unstable_error)?;
        self.field(&key, value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}
impl SerializeStruct for FieldWriter<'_, '_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CanonicalError> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}
impl SerializeStructVariant for FieldWriter<'_, '_, '_, '_> {
    type Ok = ();
    type Error = CanonicalError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CanonicalError> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), CanonicalError> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Serialize)]
    struct Nested {
        values: Vec<Option<f64>>,
    }
    #[test]
    fn nonfinite_nested_options_never_become_null() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(to_vec(&Nested {
                values: vec![Some(bad)]
            })
            .is_err());
            assert!(to_vec(&std::collections::BTreeMap::from([("bad", Some(bad))])).is_err());
        }
        let good = Nested {
            values: vec![None, Some(0.0), Some(f64::MIN_POSITIVE)],
        };
        assert_eq!(to_vec(&good).unwrap(), serde_json::to_vec(&good).unwrap());
    }

    mod streamed {
        use serde::{Serialize, Serializer};
        use std::collections::{BTreeMap, HashMap};

        #[derive(Serialize)]
        struct Unit;
        #[derive(Serialize)]
        struct Wrapper(u32);
        #[derive(Serialize)]
        struct Pair(i8, String);
        #[derive(Serialize)]
        enum Variants {
            Plain,
            Newtype(f32),
            Tuple(u8, Option<bool>),
            Record { zeta: char, alpha: Vec<u16> },
        }
        #[derive(Serialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum Tagged {
            First { value: i64 },
            Second { items: Vec<String> },
        }
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Untagged {
            Number(u64),
            Text(String),
        }
        #[derive(Serialize)]
        struct Inner {
            omega: f64,
            beta: Option<Box<Inner>>,
        }
        #[derive(Serialize)]
        struct Shapes {
            zulu: Unit,
            yankee: Wrapper,
            x_ray: Pair,
            whiskey: Vec<Variants>,
            victor: Vec<Tagged>,
            uniform: (Untagged, Untagged),
            tango: BTreeMap<String, Vec<f64>>,
            sierra: HashMap<String, i32>,
            romeo: BTreeMap<i64, bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            quebec: Option<u8>,
            #[serde(flatten)]
            papa: BTreeMap<String, String>,
            oscar: Box<Inner>,
            november: (i128, u128, (), [u8; 3]),
            mike: Vec<Vec<Option<String>>>,
            lima: String,
            empty_map: BTreeMap<String, u8>,
            empty_seq: Vec<u8>,
        }

        fn shapes() -> Shapes {
            Shapes {
                zulu: Unit,
                yankee: Wrapper(7),
                x_ray: Pair(-3, "pair".into()),
                whiskey: vec![
                    Variants::Plain,
                    Variants::Newtype(0.1),
                    Variants::Tuple(4, None),
                    Variants::Record {
                        zeta: 'z',
                        alpha: vec![1, 2],
                    },
                ],
                victor: vec![
                    Tagged::First { value: -9 },
                    Tagged::Second {
                        items: vec!["a".into()],
                    },
                ],
                uniform: (Untagged::Number(u64::MAX), Untagged::Text("u".into())),
                tango: BTreeMap::from([
                    ("b".into(), vec![1.5, -0.0, 1e300]),
                    ("a".into(), vec![f64::MIN_POSITIVE]),
                ]),
                sierra: (0..40).map(|i| (format!("key{i}"), i)).collect(),
                romeo: BTreeMap::from([(-1, true), (10, false), (2, true)]),
                quebec: None,
                papa: BTreeMap::from([("flat".into(), "f".into()), ("aaa".into(), "g".into())]),
                oscar: Box::new(Inner {
                    omega: 2.0,
                    beta: Some(Box::new(Inner {
                        omega: 3.25,
                        beta: None,
                    })),
                }),
                november: (i128::from(i64::MIN), u128::from(u64::MAX), (), [0, 1, 255]),
                mike: vec![vec![Some("x".into()), None], vec![]],
                lima:
                    "quote \" backslash \\ newline \n tab \t unicode \u{e9}\u{1F600} control \u{1}"
                        .into(),
                empty_map: BTreeMap::new(),
                empty_seq: Vec::new(),
            }
        }

        /// A map that repeats a key: `Value` keeps the last value; the streamed
        /// writer rejects it.
        struct Repeated;
        impl Serialize for Repeated {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("b", &1)?;
                map.serialize_entry("a", &2)?;
                map.serialize_entry("b", &3)?;
                map.end()
            }
        }

        fn digests_agree<T: Serialize>(value: &T) {
            assert_eq!(
                crate::research_digest_streamed(value).unwrap(),
                crate::research_digest(value).unwrap()
            );
        }

        #[test]
        fn streamed_digest_equals_the_value_digest() {
            digests_agree(&shapes());
            digests_agree(&Some(Wrapper(1)));
            digests_agree(&Option::<u8>::None);
            digests_agree(&"top level string");
            digests_agree(&1.0e-7_f64);
            digests_agree(&Variants::Newtype(f32::MAX));
            digests_agree(&BTreeMap::from([('c', 1), ('a', 2)]));
            let text = serde_json::to_string(&shapes()).unwrap();
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            digests_agree(&value);
        }

        #[test]
        fn streamed_digest_screens_secrets_like_the_value_digest() {
            #[derive(Serialize)]
            struct Holder {
                outer: Vec<BTreeMap<String, String>>,
            }
            let field = Holder {
                outer: vec![
                    BTreeMap::new(),
                    BTreeMap::from([("API-Key".into(), "x".into())]),
                ],
            };
            let material = Holder {
                outer: vec![BTreeMap::from([(
                    "note".into(),
                    "ghp_0123456789abcdefghijklmnopqrstuvwxyz".into(),
                )])],
            };
            for value in [field, material] {
                let streamed = crate::research_digest_streamed(&value).unwrap_err();
                let tree = crate::research_digest(&value).unwrap_err();
                assert_eq!(streamed.to_string(), tree.to_string());
            }
            let streamed = crate::research_digest_streamed(&vec![f64::NAN]).unwrap_err();
            let tree = crate::research_digest(&vec![f64::NAN]).unwrap_err();
            assert_eq!(streamed.to_string(), tree.to_string());
        }

        /// Alternates the order of its entries on every serialization.
        struct Alternating(std::cell::Cell<bool>);
        impl Serialize for Alternating {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeMap;
                let flip = self.0.get();
                self.0.set(!flip);
                let mut map = serializer.serialize_map(None)?;
                for (key, value) in if flip {
                    [("b", 20), ("a", 10)]
                } else {
                    [("a", 10), ("b", 20)]
                } {
                    map.serialize_entry(key, &value)?;
                }
                map.end()
            }
        }

        #[test]
        fn values_stay_paired_with_their_keys_across_passes() {
            #[derive(Serialize)]
            struct Flattened {
                zeta: u8,
                #[serde(flatten)]
                rest: Alternating,
            }
            let expected = crate::research_digest(&serde_json::json!({"a": 10, "b": 20})).unwrap();
            // Probed in order, the single writing pass sees the other order and
            // is rejected; probed out of order, each key's pass finds its value.
            let in_order = Alternating(std::cell::Cell::new(false));
            let error = crate::research_digest_streamed(&in_order).unwrap_err();
            assert!(error.to_string().contains("repeated pass"), "{error}");
            let reordered = Alternating(std::cell::Cell::new(true));
            assert_eq!(
                crate::research_digest_streamed(&reordered).unwrap(),
                expected
            );
            for start in [false, true] {
                let flattened = Flattened {
                    zeta: 1,
                    rest: Alternating(std::cell::Cell::new(start)),
                };
                assert_eq!(
                    crate::research_digest_streamed(&flattened).unwrap(),
                    crate::research_digest(&serde_json::json!({"a": 10, "b": 20, "zeta": 1}))
                        .unwrap()
                );
            }
        }

        #[test]
        fn repeated_keys_are_rejected_not_overwritten() {
            struct NanThenOne;
            impl Serialize for NanThenOne {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    use serde::ser::SerializeMap;
                    let mut map = serializer.serialize_map(None)?;
                    map.serialize_entry("x", &f64::NAN)?;
                    map.serialize_entry("x", &1)?;
                    map.end()
                }
            }
            #[derive(Serialize)]
            struct RenamedAndFlattened {
                #[serde(rename = "x")]
                first: f64,
                #[serde(flatten)]
                rest: BTreeMap<String, u8>,
            }
            assert!(crate::research_digest(&NanThenOne).is_err());
            assert!(crate::research_digest_streamed(&NanThenOne).is_err());
            let shadowed = RenamedAndFlattened {
                first: f64::NAN,
                rest: BTreeMap::from([("x".into(), 1)]),
            };
            assert!(crate::research_digest(&shadowed).is_err());
            assert!(crate::research_digest_streamed(&shadowed).is_err());
            for value in [&Repeated as &dyn erased::Erased, &vec![Repeated, Repeated]] {
                let error = value.streamed().unwrap_err();
                assert!(error.to_string().contains("repeated object key"), "{error}");
            }
        }

        mod erased {
            pub trait Erased {
                fn streamed(&self) -> Result<crate::ConfigDigest, crate::ConfigError>;
            }
            impl<T: serde::Serialize> Erased for T {
                fn streamed(&self) -> Result<crate::ConfigDigest, crate::ConfigError> {
                    crate::research_digest_streamed(self)
                }
            }
        }

        #[test]
        fn map_keys_follow_the_value_conversion() {
            struct FloatKeys(Vec<f64>);
            impl Serialize for FloatKeys {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    use serde::ser::SerializeMap;
                    let mut map = serializer.serialize_map(None)?;
                    for (index, key) in self.0.iter().enumerate() {
                        map.serialize_entry(key, &index)?;
                    }
                    map.end()
                }
            }
            for keys in [
                vec![1.25],
                vec![0.1, -0.0, 1e300, 5e-324, 123456789.0],
                vec![f64::MAX, f64::MIN_POSITIVE],
            ] {
                let value = FloatKeys(keys);
                assert_eq!(
                    crate::research_digest_streamed(&value).unwrap(),
                    crate::research_digest(&value).unwrap()
                );
            }
            for bad in [f64::NAN, f64::INFINITY] {
                assert!(crate::research_digest(&FloatKeys(vec![bad])).is_err());
                assert!(crate::research_digest_streamed(&FloatKeys(vec![bad])).is_err());
            }
            let keys = BTreeMap::from([(true, 1), (false, 2)]);
            assert_eq!(
                crate::research_digest_streamed(&keys).unwrap(),
                crate::research_digest(&keys).unwrap()
            );
        }

        #[test]
        fn omission_requires_a_top_level_struct_or_map() {
            for value in [
                Variants::Newtype(2.0),
                Variants::Tuple(1, None),
                Variants::Record {
                    zeta: 'z',
                    alpha: vec![],
                },
            ] {
                assert!(super::super::to_value_omitting(
                    &value,
                    &["Newtype", "Tuple", "Record"],
                    "test"
                )
                .is_err());
            }
            assert!(super::super::to_value_omitting(&vec![1, 2], &["x"], "test").is_err());
            assert!(super::super::to_value_omitting(&vec![1, 2], &[], "test").is_ok());
        }

        #[test]
        fn omitting_drops_only_top_level_fields() {
            let full = super::super::to_value(&shapes()).unwrap();
            let omitted =
                super::super::to_value_omitting(&shapes(), &["tango", "oscar", "omega"], "test")
                    .unwrap();
            let mut expected = full.clone();
            let object = expected.as_object_mut().unwrap();
            object.remove("tango");
            object.remove("oscar");
            assert_eq!(omitted, expected);
            assert_eq!(
                super::super::to_value_omitting(&shapes(), &[], "test").unwrap(),
                full
            );
        }
    }
}
