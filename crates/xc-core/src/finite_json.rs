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
}
