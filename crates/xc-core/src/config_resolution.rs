//! Canonical, layered effective-configuration resolution.
//!
//! Operational inputs are converted into explicit JSON layers, merged through
//! one documented precedence order, deserialized into a typed request, and
//! hashed only after validation. Environment overrides require an allowlist;
//! no process environment is read implicitly by this library.

use crate::{ConfigError, SolverConfig};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Increasing configuration precedence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSource {
    BuiltIn,
    User,
    Project,
    Run,
    CommandLine,
    Environment,
}

/// One explicit input to configuration resolution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationLayer {
    pub source: ConfigSource,
    pub name: String,
    pub value: Value,
    /// Required for environment layers. Entries are dotted leaf paths.
    pub override_allowlist: Option<BTreeSet<String>>,
}

impl ConfigurationLayer {
    pub fn from_serializable<T: Serialize>(
        source: ConfigSource,
        name: impl Into<String>,
        value: &T,
    ) -> Result<Self, ConfigResolutionError> {
        Ok(Self {
            source,
            name: name.into(),
            value: serde_json::to_value(value)
                .map_err(|error| ConfigResolutionError::Serialization(error.to_string()))?,
            override_allowlist: None,
        })
    }

    pub fn environment(
        name: impl Into<String>,
        value: Value,
        override_allowlist: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            source: ConfigSource::Environment,
            name: name.into(),
            value,
            override_allowlist: Some(override_allowlist.into_iter().collect()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConfigLayerRecord {
    pub source: ConfigSource,
    pub name: String,
    pub canonical_digest: String,
}

/// One non-default effective value and the layer that supplied it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConfigOverrideRecord {
    pub path: String,
    pub source: ConfigSource,
    pub effective_value: Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConfigDigest(pub String);

impl ConfigDigest {
    pub fn is_sha256(&self) -> bool {
        self.0.len() == 64
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }
}

impl Display for ConfigDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A validated typed configuration and its identity-bearing representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectiveConfiguration<T> {
    pub resolved: T,
    pub canonical_json: String,
    pub digest: ConfigDigest,
    pub layers: Vec<ConfigLayerRecord>,
    pub resolved_paths: BTreeMap<String, ConfigSource>,
    /// Display-ready ledger of every effective value not supplied by defaults.
    pub overrides: Vec<ConfigOverrideRecord>,
}

/// Validation hook implemented by every root configuration type.
pub trait ValidateResolvedConfig {
    fn validate_resolved(&self) -> Result<(), ConfigError>;
}

impl ValidateResolvedConfig for SolverConfig {
    fn validate_resolved(&self) -> Result<(), ConfigError> {
        self.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigResolutionError {
    EmptyLayers,
    DuplicateSource(ConfigSource),
    InvalidLayer(String),
    ForbiddenOverride { source: ConfigSource, path: String },
    Serialization(String),
    Validation(String),
}

impl Display for ConfigResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLayers => formatter.write_str("configuration resolution requires a layer"),
            Self::DuplicateSource(source) => {
                write!(
                    formatter,
                    "configuration source {source:?} appears more than once"
                )
            }
            Self::InvalidLayer(message) => {
                write!(formatter, "invalid configuration layer: {message}")
            }
            Self::ForbiddenOverride { source, path } => write!(
                formatter,
                "configuration source {source:?} may not override {path:?}"
            ),
            Self::Serialization(message) => {
                write!(formatter, "configuration serialization failed: {message}")
            }
            Self::Validation(message) => {
                write!(formatter, "configuration validation failed: {message}")
            }
        }
    }
}

impl Error for ConfigResolutionError {}

/// Resolve, type-check, validate, canonicalize, and hash configuration layers.
pub fn resolve_configuration<T>(
    layers: impl IntoIterator<Item = ConfigurationLayer>,
) -> Result<EffectiveConfiguration<T>, ConfigResolutionError>
where
    T: DeserializeOwned + Serialize + ValidateResolvedConfig,
{
    let mut layers: Vec<_> = layers.into_iter().collect();
    if layers.is_empty() {
        return Err(ConfigResolutionError::EmptyLayers);
    }
    layers.sort_by_key(|layer| layer.source);
    let mut seen = BTreeSet::new();
    let mut merged = Value::Object(Map::new());
    let mut resolved_paths = BTreeMap::new();
    let mut records = Vec::with_capacity(layers.len());

    for layer in layers {
        if !seen.insert(layer.source) {
            return Err(ConfigResolutionError::DuplicateSource(layer.source));
        }
        if layer.name.trim().is_empty() {
            return Err(ConfigResolutionError::InvalidLayer(
                "layer name must be nonempty".to_owned(),
            ));
        }
        if !layer.value.is_object() {
            return Err(ConfigResolutionError::InvalidLayer(format!(
                "layer {:?} must be a JSON object",
                layer.name
            )));
        }
        let leaf_paths = collect_leaf_paths(&layer.value);
        match (layer.source, &layer.override_allowlist) {
            (ConfigSource::Environment, Some(allowlist)) => {
                for path in &leaf_paths {
                    if !allowlist.contains(path) {
                        return Err(ConfigResolutionError::ForbiddenOverride {
                            source: layer.source,
                            path: path.clone(),
                        });
                    }
                }
            }
            (ConfigSource::Environment, None) => {
                return Err(ConfigResolutionError::InvalidLayer(
                    "environment layer requires an explicit override allowlist".to_owned(),
                ));
            }
            (_, Some(_)) => {
                return Err(ConfigResolutionError::InvalidLayer(
                    "override allowlists are valid only for environment layers".to_owned(),
                ));
            }
            (_, None) => {}
        }

        let canonical_layer = canonical_json(&layer.value)?;
        records.push(ConfigLayerRecord {
            source: layer.source,
            name: layer.name,
            canonical_digest: sha256_hex(canonical_layer.as_bytes()),
        });
        merge_values(
            &mut merged,
            layer.value,
            "",
            layer.source,
            &mut resolved_paths,
        )?;
    }

    let resolved: T = serde_json::from_value(merged).map_err(|error| {
        ConfigResolutionError::Serialization(format!(
            "effective configuration does not match its typed schema: {error}"
        ))
    })?;
    resolved
        .validate_resolved()
        .map_err(|error| ConfigResolutionError::Validation(error.to_string()))?;
    let resolved_value = serde_json::to_value(&resolved)
        .map_err(|error| ConfigResolutionError::Serialization(error.to_string()))?;
    let canonical_json = canonical_json(&resolved_value)?;
    let digest = ConfigDigest(sha256_hex(canonical_json.as_bytes()));
    let overrides = resolved_paths
        .iter()
        .filter(|(_, source)| **source != ConfigSource::BuiltIn)
        .map(|(path, source)| {
            let effective_value = value_at_path(&resolved_value, path).ok_or_else(|| {
                ConfigResolutionError::Serialization(format!(
                    "resolved configuration does not contain recorded path {path:?}"
                ))
            })?;
            Ok(ConfigOverrideRecord {
                path: path.clone(),
                source: *source,
                effective_value: effective_value.clone(),
            })
        })
        .collect::<Result<Vec<_>, ConfigResolutionError>>()?;

    Ok(EffectiveConfiguration {
        resolved,
        canonical_json,
        digest,
        layers: records,
        resolved_paths,
        overrides,
    })
}

fn value_at_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(value, |current, component| {
        current.as_object()?.get(component)
    })
}

fn merge_values(
    target: &mut Value,
    incoming: Value,
    prefix: &str,
    source: ConfigSource,
    resolved_paths: &mut BTreeMap<String, ConfigSource>,
) -> Result<(), ConfigResolutionError> {
    match (target, incoming) {
        (Value::Object(target), Value::Object(incoming)) => {
            for (key, value) in incoming {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if value.is_object() {
                    let child = target
                        .entry(key)
                        .or_insert_with(|| Value::Object(Map::new()));
                    if !child.is_object() {
                        *child = Value::Object(Map::new());
                    }
                    merge_values(child, value, &path, source, resolved_paths)?;
                } else {
                    target.insert(key, value);
                    resolved_paths.insert(path, source);
                }
            }
            Ok(())
        }
        _ => Err(ConfigResolutionError::InvalidLayer(
            "configuration merge encountered a non-object root".to_owned(),
        )),
    }
}

fn collect_leaf_paths(value: &Value) -> BTreeSet<String> {
    fn recurse(value: &Value, prefix: &str, output: &mut BTreeSet<String>) {
        if let Value::Object(object) = value {
            for (key, child) in object {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if child.is_object() {
                    recurse(child, &path, output);
                } else {
                    output.insert(path);
                }
            }
        }
    }
    let mut output = BTreeSet::new();
    recurse(value, "", &mut output);
    output
}

pub(crate) fn canonical_json(value: &Value) -> Result<String, ConfigResolutionError> {
    fn write(value: &Value, output: &mut String) -> Result<(), ConfigResolutionError> {
        match value {
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::String(value) => output.push_str(
                &serde_json::to_string(value)
                    .map_err(|error| ConfigResolutionError::Serialization(error.to_string()))?,
            ),
            Value::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    write(value, output)?;
                }
                output.push(']');
            }
            Value::Object(values) => {
                output.push('{');
                let mut keys: Vec<_> = values.keys().collect();
                keys.sort();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    output.push_str(&serde_json::to_string(key).map_err(|error| {
                        ConfigResolutionError::Serialization(error.to_string())
                    })?);
                    output.push(':');
                    write(&values[key], output)?;
                }
                output.push('}');
            }
        }
        Ok(())
    }

    let mut output = String::new();
    write(value, &mut output)?;
    Ok(output)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing into String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssuranceLevel, PrecisionPolicy, Reproducibility, StoppingPolicy, Subspace};
    use serde_json::json;

    fn base_config() -> SolverConfig {
        SolverConfig {
            target: crate::EigenTarget::AlgebraicSmallest,
            subspace: Subspace::Full,
            assurance: AssuranceLevel::Computed,
            precision: PrecisionPolicy::default(),
            stopping: StoppingPolicy::default(),
            reproducibility: Reproducibility::Deterministic,
            algorithm_preferences: vec!["reference".to_owned()],
            allow_lower_precision_seed: false,
            allow_randomized_seed: false,
        }
    }

    #[test]
    fn precedence_is_canonical_and_visible() {
        let built_in = ConfigurationLayer::from_serializable(
            ConfigSource::BuiltIn,
            "built-in defaults",
            &base_config(),
        )
        .unwrap();
        let project = ConfigurationLayer {
            source: ConfigSource::Project,
            name: "project policy".to_owned(),
            value: json!({"precision": {"initial_bits": 512}}),
            override_allowlist: None,
        };
        let command_line = ConfigurationLayer {
            source: ConfigSource::CommandLine,
            name: "command line".to_owned(),
            value: json!({"precision": {"initial_bits": 1024}}),
            override_allowlist: None,
        };
        let effective: EffectiveConfiguration<SolverConfig> =
            resolve_configuration([command_line, built_in, project]).unwrap();
        assert_eq!(effective.resolved.precision.initial_bits, 1024);
        assert_eq!(
            effective.resolved_paths.get("precision.initial_bits"),
            Some(&ConfigSource::CommandLine)
        );
        assert!(effective.overrides.iter().any(|record| {
            record.path == "precision.initial_bits"
                && record.source == ConfigSource::CommandLine
                && record.effective_value == json!(1024)
        }));

        let repeated: EffectiveConfiguration<SolverConfig> = resolve_configuration([
            ConfigurationLayer::from_serializable(
                ConfigSource::BuiltIn,
                "built-in defaults",
                &base_config(),
            )
            .unwrap(),
            ConfigurationLayer {
                source: ConfigSource::Project,
                name: "project policy".to_owned(),
                value: json!({"precision": {"initial_bits": 512}}),
                override_allowlist: None,
            },
            ConfigurationLayer {
                source: ConfigSource::CommandLine,
                name: "command line".to_owned(),
                value: json!({"precision": {"initial_bits": 1024}}),
                override_allowlist: None,
            },
        ])
        .unwrap();
        assert_eq!(effective.digest, repeated.digest);
        assert_eq!(effective.canonical_json, repeated.canonical_json);
    }

    #[test]
    fn permitted_operational_override_is_last_and_visible() {
        let built_in = ConfigurationLayer::from_serializable(
            ConfigSource::BuiltIn,
            "built-in defaults",
            &base_config(),
        )
        .unwrap();
        let command_line = ConfigurationLayer {
            source: ConfigSource::CommandLine,
            name: "command line".to_owned(),
            value: json!({"precision": {"initial_bits": 1024}}),
            override_allowlist: None,
        };
        let environment = ConfigurationLayer::environment(
            "explicit operational override",
            json!({"precision": {"initial_bits": 2048}}),
            ["precision.initial_bits".to_owned()],
        );

        let effective: EffectiveConfiguration<SolverConfig> =
            resolve_configuration([environment, command_line, built_in]).unwrap();
        assert_eq!(effective.resolved.precision.initial_bits, 2048);
        assert_eq!(
            effective.resolved_paths.get("precision.initial_bits"),
            Some(&ConfigSource::Environment)
        );
        assert!(effective.overrides.iter().any(|record| {
            record.path == "precision.initial_bits"
                && record.source == ConfigSource::Environment
                && record.effective_value == json!(2048)
        }));
        let displayed = serde_json::to_string_pretty(&effective).unwrap();
        assert!(displayed.contains("explicit operational override"));
        assert!(displayed.contains("precision.initial_bits"));
    }

    #[test]
    fn environment_override_requires_explicit_permission() {
        let built_in = ConfigurationLayer::from_serializable(
            ConfigSource::BuiltIn,
            "built-in defaults",
            &base_config(),
        )
        .unwrap();
        let environment = ConfigurationLayer::environment(
            "permitted environment",
            json!({"precision": {"initial_bits": 512}, "assurance": "certified"}),
            ["precision.initial_bits".to_owned()],
        );
        let error = resolve_configuration::<SolverConfig>([built_in, environment]).unwrap_err();
        assert!(matches!(
            error,
            ConfigResolutionError::ForbiddenOverride {
                source: ConfigSource::Environment,
                path
            } if path == "assurance"
        ));
    }

    #[test]
    fn unknown_fields_fail_typed_resolution() {
        let mut value = serde_json::to_value(base_config()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("mystery".to_owned(), Value::Bool(true));
        let layer = ConfigurationLayer {
            source: ConfigSource::BuiltIn,
            name: "invalid".to_owned(),
            value,
            override_allowlist: None,
        };
        assert!(matches!(
            resolve_configuration::<SolverConfig>([layer]),
            Err(ConfigResolutionError::Serialization(_))
        ));
    }
}
