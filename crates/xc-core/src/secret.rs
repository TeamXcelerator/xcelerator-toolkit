use crate::ConfigError;
use serde::Serialize;
use serde_json::Value;

const SENSITIVE_KEYS: &[&str] = &[
    "access_token",
    "api_key",
    "authorization_header",
    "client_secret",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "signed_url",
];

/// Rejects credential-shaped fields and values before a record is persisted.
///
/// The check reports only the JSON path and marker class; it never includes
/// the suspected secret in an error or log message.
pub fn validate_secret_free<T: Serialize>(record: &T, context: &str) -> Result<(), ConfigError> {
    let value = serde_json::to_value(record)
        .map_err(|error| ConfigError::new(format!("{context} serialization failed: {error}")))?;
    inspect_value(&value, "$", context)
}

fn inspect_value(value: &Value, path: &str, context: &str) -> Result<(), ConfigError> {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                let child_path = format!("{path}.{key}");
                let normalized = key.to_ascii_lowercase().replace(['-', ' '], "_");
                if SENSITIVE_KEYS.contains(&normalized.as_str()) {
                    return Err(ConfigError::new(format!(
                        "{context} contains a credential-bearing field at {child_path}"
                    )));
                }
                inspect_value(child, &child_path, context)?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                inspect_value(child, &format!("{path}[{index}]"), context)?;
            }
        }
        Value::String(text) if secret_marker(text).is_some() => {
            return Err(ConfigError::new(format!(
                "{context} contains credential-shaped material at {path}"
            )));
        }
        _ => {}
    }
    Ok(())
}

fn secret_marker(text: &str) -> Option<&'static str> {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("-----begin private key-----") // SECRET_AUDIT_PATTERN
        || lower.contains("-----begin rsa private key-----") // SECRET_AUDIT_PATTERN
        || lower.contains("-----begin openssh private key-----") // SECRET_AUDIT_PATTERN
        || lower.contains("x-amz-signature=")
        || lower.contains("x-goog-signature=")
    {
        return Some("key_or_signed_url");
    }
    for prefix in [
        "github_pat_",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "xoxb-",
        "xoxp-",
    ] {
        if lower
            .find(prefix)
            .is_some_and(|index| trimmed.len().saturating_sub(index + prefix.len()) >= 16)
        {
            return Some("provider_token");
        }
    }
    let bytes = trimmed.as_bytes();
    if bytes.len() == 20
        && (trimmed.starts_with("AKIA") || trimmed.starts_with("ASIA"))
        && bytes.iter().all(u8::is_ascii_alphanumeric)
    {
        return Some("cloud_access_key");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_sensitive_fields_without_echoing_values() {
        let record = json!({"nested": {"access-token": "do-not-echo"}});
        let error = validate_secret_free(&record, "fixture").unwrap_err();
        assert!(error.to_string().contains("$.nested.access-token"));
        assert!(!error.to_string().contains("do-not-echo"));
    }

    #[test]
    fn rejects_provider_tokens_hidden_in_free_text() {
        let record = json!({"notes": ["github_pat_abcdefghijklmnopqrstuvwxyz123456"]}); // SECRET_AUDIT_PATTERN
        assert!(validate_secret_free(&record, "fixture").is_err());
    }

    #[test]
    fn permits_nonsecret_token_and_key_terminology() {
        let record = json!({
            "cancellation_token": "user_requested",
            "semantic_key": "sha256:abc",
            "authorization_evidence_digest": "00"
        });
        validate_secret_free(&record, "fixture").unwrap();
    }
}
