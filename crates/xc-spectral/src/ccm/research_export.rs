//! Small derived exports for offline navigation. Canonical artifacts remain authoritative.
use super::{extended_research::ExtendedAnalysis, retained_evidence::ResearchRecord};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::PathBuf};
use xc_cache::{ArtifactManifest, ContentDigest};
struct FieldPolicy<'a> {
    rules: Vec<(&'a str, &'a str)>,
}
impl<'a> FieldPolicy<'a> {
    fn parse(text: &'a str) -> Result<Self> {
        let mut rules = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let Some((kind, prefix)) = line.split_once(':') else {
                bail!("invalid compact field policy at line {}", index + 1);
            };
            if !matches!(kind, "prefix" | "indexed")
                || prefix.is_empty()
                || !prefix
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_')
                || rules.contains(&(kind, prefix))
            {
                bail!("invalid compact field policy at line {}", index + 1);
            }
            rules.push((kind, prefix));
        }
        if rules.is_empty() {
            bail!("invalid compact field policy: empty policy");
        }
        Ok(Self { rules })
    }
    fn scalar_field(&self, name: &str) -> bool {
        !self.rules.iter().any(|(kind, prefix)| {
            name.strip_prefix(*prefix).is_some_and(|suffix| {
                *kind == "prefix"
                    || (!suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()))
            })
        })
    }
}
fn fields<'a>(
    values: &'a BTreeMap<String, String>,
    policy: &FieldPolicy<'_>,
) -> BTreeMap<&'a str, &'a str> {
    values
        .iter()
        .filter(|(k, _)| policy.scalar_field(k))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

pub(crate) fn compact(
    record: &ResearchRecord<ExtendedAnalysis>,
    manifest: Option<&ArtifactManifest>,
) -> Result<Value> {
    let policy = FieldPolicy::parse(include_str!("compact_field_policy.txt"))?;
    let d = &record.data;
    Ok(
        json!({"schema_version":1,"export_scope":"derived exact scalar view; canonical artifact is authoritative; no new validation", "source_manifest":manifest,
        "artifact_digest":manifest.map(|m|&m.content_digest),
        "report":{"kind":record.kind,"source_dependencies":record.source_dependencies,"scope":record.scope,
        "data":{"lambda_squared":d.lambda_squared,"n_modes":d.n_modes,"source_precision_bits":d.source_precision_bits,"working_precision_bits":d.working_precision_bits,
            "outcome":d.outcome,"convention":d.convention,"assurance":d.assurance,"reason":d.reason,"values":fields(&d.values, &policy),
            "rows":d.rows.iter().map(|r|json!({"ordinal":r.ordinal,"label":r.label,"outcome":r.outcome,"values":fields(&r.values, &policy),"notes":r.notes})).collect::<Vec<_>>()}}}),
    )
}
fn write(
    record: &ResearchRecord<ExtendedAnalysis>,
    manifest: Option<&ArtifactManifest>,
) -> Result<()> {
    let directory = std::env::var_os("XC_RESEARCH_SUMMARY_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            xc_cache::ManagedArtifactCacheConfig::from_environment()
                .ok()
                .flatten()
                .map(|c| c.cache_root.join("research-summaries"))
        });
    let Some(directory) = directory else {
        return Ok(());
    };
    let payload = serde_json::to_vec(&compact(record, manifest)?)?;
    if payload.len() > 64 << 20 {
        bail!("compact summary exceeds 64 MiB; canonical measurements remain retained");
    }
    let digest = ContentDigest::sha256(&payload);
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{}.json", digest.0));
    // Content-derived names preserve previous summaries and permit independent readers.
    let temporary = directory.join(format!(
        "{}.{}-{}.tmp",
        digest.0,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);
        match std::fs::hard_link(&temporary, &path) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if std::fs::metadata(&path)?.len() != payload.len() as u64
                    || ContentDigest::sha256(&std::fs::read(&path)?) != digest
                {
                    bail!("existing compact summary is damaged; preserved for inspection");
                }
            }
            Err(e) => return Err(e.into()),
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result?;
    Ok(())
}
pub(crate) fn emit(record: &ResearchRecord<ExtendedAnalysis>, manifest: Option<&ArtifactManifest>) {
    if let Err(e) = write(record, manifest) {
        eprintln!("research summary unavailable: {e}; canonical artifact retained");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_field_policy_preserves_nonindices_and_excludes_unbounded_indices() {
        let policy = FieldPolicy::parse(include_str!("compact_field_policy.txt")).unwrap();
        for rule in include_str!("compact_field_policy.txt").lines() {
            let (kind, prefix) = rule.split_once(':').unwrap();
            assert!(matches!(kind, "prefix" | "indexed") && !prefix.is_empty());
            assert!(!policy.scalar_field(&format!("{prefix}{}", "9".repeat(100))));
        }
        for name in ["tail_", "tail_+1", "tail_١", "tail_energy_lift"] {
            assert!(policy.scalar_field(name), "{name}");
        }
    }
    #[test]
    fn malformed_policy_fails_before_any_fields_are_exported() {
        for text in [
            "",
            "\n",
            "prefix:ok\n\n",
            "unknown:x",
            "prefix:",
            "prefix:x y",
            "prefix:x:y",
            "prefix:x\nprefix:x",
        ] {
            assert!(FieldPolicy::parse(text).is_err(), "{text:?}");
        }
        assert!(FieldPolicy::parse("prefix:a_\r\nindexed:b_\r\n").is_ok());
    }
    #[test]
    fn compact_view_omits_dense_fields_and_preserves_tiny_decimal_text() {
        let value = "-1.234567890123456789e-2000";
        let m = BTreeMap::from([
            ("model_vector_coefficient_0".into(), "1".into()),
            ("tail_0".into(), "2".into()),
            ("tail_energy_lift".into(), value.into()),
        ]);
        assert_eq!(
            fields(
                &m,
                &FieldPolicy::parse(include_str!("compact_field_policy.txt")).unwrap()
            ),
            BTreeMap::from([("tail_energy_lift", value)])
        );
    }
}
