//! Versioned retained research payload codec. Requests and selected producer
//! ancestry remain in manifests; schema 2 contains only the numerical/evidence
//! statement. Schema-1 serialization remains exactly as originally defined.
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use xc_cache::DependencyRef;

#[derive(Clone, Debug)]
pub struct ResearchRecord<T> {
    pub schema_version: u32,
    pub semantics: String,
    pub kind: String,
    pub scope: String,
    /// Request-local convenience fields, absent from the schema-2 payload.
    pub source_dependencies: Vec<DependencyRef>,
    pub request: Value,
    pub data: T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRecord<T> {
    schema_version: u32,
    semantics: String,
    kind: String,
    scope: String,
    source_dependencies: Vec<DependencyRef>,
    request: Value,
    data: T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericalRecord<T> {
    schema_version: u32,
    semantics: String,
    kind: String,
    scope: String,
    data: T,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_rows: Option<usize>,
}

impl<T: Serialize> Serialize for ResearchRecord<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.schema_version {
            1 => LegacyRecord {
                schema_version: 1,
                semantics: self.semantics.clone(),
                kind: self.kind.clone(),
                scope: self.scope.clone(),
                source_dependencies: self.source_dependencies.clone(),
                request: self.request.clone(),
                data: &self.data,
            }
            .serialize(serializer),
            2 => NumericalRecord {
                schema_version: 2,
                semantics: self.semantics.clone(),
                kind: self.kind.clone(),
                scope: self.scope.clone(),
                expected_rows: self
                    .request
                    .get("expected_rows")
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok()),
                data: &self.data,
            }
            .serialize(serializer),
            _ => Err(serde::ser::Error::custom(
                "unsupported retained research payload schema",
            )),
        }
    }
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for ResearchRecord<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.get("schema_version").and_then(Value::as_u64) {
            Some(1) => {
                let record: LegacyRecord<T> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self {
                    schema_version: 1,
                    semantics: record.semantics,
                    kind: record.kind,
                    scope: record.scope,
                    source_dependencies: record.source_dependencies,
                    request: record.request,
                    data: record.data,
                })
            }
            Some(2) => {
                let record: NumericalRecord<T> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self {
                    schema_version: 2,
                    semantics: record.semantics,
                    kind: record.kind,
                    scope: record.scope,
                    source_dependencies: Vec::new(),
                    request: serde_json::json!({"expected_rows":record.expected_rows}),
                    data: record.data,
                })
            }
            _ => Err(serde::de::Error::custom(
                "unsupported retained research payload schema",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numerical_record_keeps_coverage_without_request_or_producer_wrappers() {
        let record = ResearchRecord {
            schema_version: 2,
            semantics: "fixture-v1".into(),
            kind: "ccm_compactness_analysis".into(),
            scope: "finite".into(),
            source_dependencies: vec![],
            request: serde_json::json!({"expected_rows":4,"producer_label":"not numerical"}),
            data: serde_json::json!({"outcome":"point_measurement","rows":[{"outcome":"point_measurement"}]}),
        };
        let wire = serde_json::to_value(&record).unwrap();
        assert_eq!(wire["expected_rows"], 4);
        assert!(wire.get("request").is_none() && wire.get("source_dependencies").is_none());
        let coverage = xc_cache::NumericalCoverage::from_value(&wire);
        assert_eq!(coverage.expected_rows, Some(4));
        assert_eq!(coverage.outcome, "partial_unresolved");
        let decoded: ResearchRecord<Value> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
        let legacy = ResearchRecord {
            schema_version: 1,
            ..record
        };
        let wire = serde_json::to_value(&legacy).unwrap();
        assert!(wire.get("expected_rows").is_none());
        assert_eq!(wire["request"]["producer_label"], "not numerical");
        let decoded: ResearchRecord<Value> = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
    }
}
