// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Publication-ready tables and convergence-dataset exports.
//!
//! Values are exported from exact strings already present in result objects;
//! this layer never reparses or rounds numerical output. A bundle contains
//! CSV, LaTeX, and JSON views plus a digest-bound manifest, all carrying the
//! same requirement identifiers and provenance.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

const PROVENANCE_COLUMNS: [(&str, &str); 7] = [
    ("requirement_ids", "Requirement IDs"),
    ("toolkit_version", "Toolkit version"),
    ("release_tag", "Release tag"),
    ("source_revision", "Source revision"),
    ("resolved_configuration_digest", "Configuration digest"),
    ("execution_fingerprint_digest", "Execution fingerprint"),
    ("input_artifact_digests", "Input artifact digests"),
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicationColumnAlignment {
    Left,
    Center,
    Right,
}

impl PublicationColumnAlignment {
    fn latex(&self) -> char {
        match self {
            Self::Left => 'l',
            Self::Center => 'c',
            Self::Right => 'r',
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationColumn {
    pub key: String,
    pub heading: String,
    pub unit: Option<String>,
    pub alignment: PublicationColumnAlignment,
}

impl PublicationColumn {
    pub fn new(
        key: impl Into<String>,
        heading: impl Into<String>,
        alignment: PublicationColumnAlignment,
    ) -> Self {
        Self {
            key: key.into(),
            heading: heading.into(),
            unit: None,
            alignment,
        }
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationProvenance {
    pub toolkit_version: String,
    pub release_tag: String,
    pub source_revision: String,
    pub resolved_configuration_digest: String,
    pub execution_fingerprint_digest: String,
    pub input_artifact_digests: Vec<String>,
}

impl PublicationProvenance {
    fn validate(&self) -> Result<(), PublicationExportError> {
        if self.toolkit_version.trim().is_empty()
            || self.release_tag != format!("v{}", self.toolkit_version)
            || self.source_revision.trim().is_empty()
        {
            return Err(PublicationExportError::InvalidTable(
                "toolkit version, matching release tag, and source revision are required"
                    .to_owned(),
            ));
        }
        validate_digest(
            &self.resolved_configuration_digest,
            "resolved configuration",
        )?;
        validate_digest(&self.execution_fingerprint_digest, "execution fingerprint")?;
        for digest in &self.input_artifact_digests {
            validate_digest(digest, "input artifact")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationRow {
    pub values: BTreeMap<String, String>,
}

impl PublicationRow {
    pub fn new(values: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationTable {
    pub schema_version: u32,
    pub table_id: String,
    pub caption: String,
    pub requirement_ids: Vec<String>,
    pub columns: Vec<PublicationColumn>,
    pub rows: Vec<PublicationRow>,
    pub provenance: PublicationProvenance,
}

impl PublicationTable {
    pub fn validate(&self) -> Result<(), PublicationExportError> {
        if self.schema_version != 1 {
            return Err(PublicationExportError::InvalidTable(format!(
                "unsupported table schema version {}",
                self.schema_version
            )));
        }
        if !is_identifier(&self.table_id) {
            return Err(PublicationExportError::InvalidTable(
                "table_id must contain only ASCII letters, digits, '-' or '_'".to_owned(),
            ));
        }
        if self.caption.trim().is_empty() || self.columns.is_empty() || self.rows.is_empty() {
            return Err(PublicationExportError::InvalidTable(
                "caption, columns, and rows must be nonempty".to_owned(),
            ));
        }
        let requirement_ids: BTreeSet<_> = self.requirement_ids.iter().collect();
        if self.requirement_ids.is_empty()
            || requirement_ids.len() != self.requirement_ids.len()
            || self
                .requirement_ids
                .iter()
                .any(|value| value.trim().is_empty())
        {
            return Err(PublicationExportError::InvalidTable(
                "requirement IDs must be nonempty and unique".to_owned(),
            ));
        }
        let keys: BTreeSet<_> = self.columns.iter().map(|column| &column.key).collect();
        if keys.len() != self.columns.len()
            || self.columns.iter().any(|column| {
                !is_identifier(&column.key)
                    || column.heading.trim().is_empty()
                    || PROVENANCE_COLUMNS
                        .iter()
                        .any(|(reserved, _)| *reserved == column.key)
            })
        {
            return Err(PublicationExportError::InvalidTable(
                "column keys must be valid, unique, and not reserved provenance keys".to_owned(),
            ));
        }
        for (index, row) in self.rows.iter().enumerate() {
            let row_keys: BTreeSet<_> = row.values.keys().collect();
            if row_keys != keys {
                return Err(PublicationExportError::InvalidTable(format!(
                    "row {index} does not contain exactly the declared columns"
                )));
            }
        }
        self.provenance.validate()
    }

    pub fn render_csv(&self) -> Result<String, PublicationExportError> {
        self.validate()?;
        let mut output = String::new();
        let headers: Vec<String> = self
            .columns
            .iter()
            .map(PublicationColumn::heading_with_unit)
            .chain(
                PROVENANCE_COLUMNS
                    .iter()
                    .map(|(_, heading)| (*heading).to_owned()),
            )
            .collect();
        output.push_str(&csv_record(headers.iter().map(String::as_str)));
        output.push('\n');
        let requirements = self.requirement_ids.join(";");
        let input_artifacts = self.provenance.input_artifact_digests.join(";");
        for row in &self.rows {
            let provenance_values = [
                requirements.as_str(),
                self.provenance.toolkit_version.as_str(),
                self.provenance.release_tag.as_str(),
                self.provenance.source_revision.as_str(),
                self.provenance.resolved_configuration_digest.as_str(),
                self.provenance.execution_fingerprint_digest.as_str(),
                input_artifacts.as_str(),
            ];
            let values = self
                .columns
                .iter()
                .map(|column| row.values[&column.key].as_str())
                .chain(provenance_values);
            output.push_str(&csv_record(values));
            output.push('\n');
        }
        Ok(output)
    }

    pub fn render_latex(&self) -> Result<String, PublicationExportError> {
        self.validate()?;
        let requirements = self.requirement_ids.join(";");
        let mut output = format!(
            "% requirements: {}\n% toolkit_version: {}\n% release_tag: {}\n% source_revision: {}\n% resolved_configuration_digest: {}\n% execution_fingerprint_digest: {}\n% input_artifact_digests: {}\n",
            latex_escape(&requirements),
            latex_escape(&self.provenance.toolkit_version),
            latex_escape(&self.provenance.release_tag),
            latex_escape(&self.provenance.source_revision),
            latex_escape(&self.provenance.resolved_configuration_digest),
            latex_escape(&self.provenance.execution_fingerprint_digest),
            latex_escape(&self.provenance.input_artifact_digests.join(";")),
        );
        let alignment: String = self
            .columns
            .iter()
            .map(|column| column.alignment.latex())
            .collect();
        output.push_str(&format!(
            "\\begin{{table}}\n\\centering\n\\begin{{tabular}}{{{alignment}}}\n"
        ));
        output.push_str(
            &self
                .columns
                .iter()
                .map(|column| latex_escape(&column.heading_with_unit()))
                .collect::<Vec<_>>()
                .join(" & "),
        );
        output.push_str(" \\\\\n\\hline\n");
        for row in &self.rows {
            output.push_str(
                &self
                    .columns
                    .iter()
                    .map(|column| latex_escape(&row.values[&column.key]))
                    .collect::<Vec<_>>()
                    .join(" & "),
            );
            output.push_str(" \\\\\n");
        }
        output.push_str(&format!(
            "\\end{{tabular}}\n\\caption{{{}}}\n\\label{{tab:{}}}\n\\end{{table}}\n",
            latex_escape(&self.caption),
            latex_escape(&self.table_id)
        ));
        Ok(output)
    }

    pub fn export_bundle(&self) -> Result<PublicationExportBundle, PublicationExportError> {
        self.validate()?;
        let table_json = serde_json::to_vec_pretty(self)
            .map_err(|error| PublicationExportError::Serialization(error.to_string()))?;
        let csv = self.render_csv()?.into_bytes();
        let latex = self.render_latex()?.into_bytes();
        let artifacts = vec![
            manifest_artifact("table.json", "application/json", &table_json),
            manifest_artifact("table.csv", "text/csv", &csv),
            manifest_artifact("table.tex", "application/x-latex", &latex),
        ];
        let manifest = PublicationExportManifest {
            schema_version: 1,
            table_id: self.table_id.clone(),
            requirement_ids: self.requirement_ids.clone(),
            source_revision: self.provenance.source_revision.clone(),
            artifacts,
        };
        let manifest_json = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| PublicationExportError::Serialization(error.to_string()))?;
        Ok(PublicationExportBundle {
            table_json,
            csv,
            latex,
            manifest_json,
            manifest,
        })
    }
}

impl PublicationColumn {
    fn heading_with_unit(&self) -> String {
        self.unit.as_ref().map_or_else(
            || self.heading.clone(),
            |unit| format!("{} ({unit})", self.heading),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvergenceTableRow {
    pub sequence_index: u64,
    pub lambda_squared: String,
    pub n_modes: u64,
    pub precision_bits: u64,
    pub root_count: u64,
    pub minimum_accuracy_digits: String,
    pub median_accuracy_digits: String,
    pub index_penalty_digits: String,
    pub completion_status: String,
}

/// Builds the canonical publication table for a finite CCM convergence sequence.
///
/// # Mathematical semantics
/// Projects ordered finite-observation rows into fixed publication columns while
/// retaining completion status, root count, precision, and measured accuracy.
/// It does not infer a continuum limit from those rows.
///
/// # Precision
/// Numeric observations are accepted as canonical decimal strings so export
/// does not round them through binary64; precision bits remain explicit.
///
/// # Failure states
/// Empty, out-of-order, incomplete, noncanonical, or provenance-inconsistent
/// rows return `PublicationExportError` before a table is emitted.
///
/// # Assurance and validity
/// The table preserves the supplied finite evidence and provenance but does not
/// independently certify the underlying computation.
///
/// # Cache effects
/// Construction is in-memory and has no cache or filesystem side effects.
/// Writing or archiving the rendered bundle is a separate explicit operation.
///
/// # Example
/// Compiled example: `crates/xc-core/examples/publication_export.rs`.
pub fn ccm_convergence_publication_table(
    table_id: impl Into<String>,
    caption: impl Into<String>,
    rows: &[ConvergenceTableRow],
    provenance: PublicationProvenance,
) -> Result<PublicationTable, PublicationExportError> {
    if rows.is_empty()
        || rows.iter().any(|row| {
            row.sequence_index == 0
                || row.n_modes == 0
                || row.precision_bits < 53
                || row.root_count == 0
                || row.lambda_squared.trim().is_empty()
                || row.minimum_accuracy_digits.trim().is_empty()
                || row.median_accuracy_digits.trim().is_empty()
                || row.index_penalty_digits.trim().is_empty()
                || row.completion_status.trim().is_empty()
        })
    {
        return Err(PublicationExportError::InvalidTable(
            "convergence rows require positive sequence/N/K, at least 53 bits, and complete exact-string summaries"
                .to_owned(),
        ));
    }
    let columns = vec![
        PublicationColumn::new(
            "sequence_index",
            "Sequence",
            PublicationColumnAlignment::Right,
        ),
        PublicationColumn::new(
            "lambda_squared",
            "lambda squared",
            PublicationColumnAlignment::Right,
        ),
        PublicationColumn::new("n_modes", "N", PublicationColumnAlignment::Right),
        PublicationColumn::new("precision_bits", "p", PublicationColumnAlignment::Right)
            .with_unit("bits"),
        PublicationColumn::new("root_count", "K", PublicationColumnAlignment::Right),
        PublicationColumn::new(
            "minimum_accuracy_digits",
            "D min",
            PublicationColumnAlignment::Right,
        ),
        PublicationColumn::new(
            "median_accuracy_digits",
            "D median",
            PublicationColumnAlignment::Right,
        ),
        PublicationColumn::new(
            "index_penalty_digits",
            "Index penalty",
            PublicationColumnAlignment::Right,
        ),
        PublicationColumn::new(
            "completion_status",
            "Status",
            PublicationColumnAlignment::Left,
        ),
    ];
    let rows = rows
        .iter()
        .map(|row| {
            PublicationRow::new([
                ("sequence_index".to_owned(), row.sequence_index.to_string()),
                ("lambda_squared".to_owned(), row.lambda_squared.clone()),
                ("n_modes".to_owned(), row.n_modes.to_string()),
                ("precision_bits".to_owned(), row.precision_bits.to_string()),
                ("root_count".to_owned(), row.root_count.to_string()),
                (
                    "minimum_accuracy_digits".to_owned(),
                    row.minimum_accuracy_digits.clone(),
                ),
                (
                    "median_accuracy_digits".to_owned(),
                    row.median_accuracy_digits.clone(),
                ),
                (
                    "index_penalty_digits".to_owned(),
                    row.index_penalty_digits.clone(),
                ),
                (
                    "completion_status".to_owned(),
                    row.completion_status.clone(),
                ),
            ])
        })
        .collect();
    let table = PublicationTable {
        schema_version: 1,
        table_id: table_id.into(),
        caption: caption.into(),
        requirement_ids: vec!["REP-010".to_owned(), "CCM-012".to_owned()],
        columns,
        rows,
        provenance,
    };
    table.validate()?;
    Ok(table)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationExportArtifact {
    pub path: String,
    pub media_type: String,
    pub sha256: String,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationExportManifest {
    pub schema_version: u32,
    pub table_id: String,
    pub requirement_ids: Vec<String>,
    pub source_revision: String,
    pub artifacts: Vec<PublicationExportArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationExportBundle {
    pub table_json: Vec<u8>,
    pub csv: Vec<u8>,
    pub latex: Vec<u8>,
    pub manifest_json: Vec<u8>,
    pub manifest: PublicationExportManifest,
}

impl PublicationExportBundle {
    /// Atomically make a new export directory visible. Existing destinations
    /// and staging directories are rejected rather than overwritten.
    pub fn write_new(&self, destination: &Path) -> Result<(), PublicationExportError> {
        if destination.exists() {
            return Err(PublicationExportError::Io(format!(
                "destination already exists: {}",
                destination.display()
            )));
        }
        let parent = destination.parent().ok_or_else(|| {
            PublicationExportError::Io("destination must have a parent directory".to_owned())
        })?;
        let name = destination
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                PublicationExportError::Io("destination name must be Unicode".to_owned())
            })?;
        let staging = parent.join(format!(".{name}.staging"));
        if staging.exists() {
            return Err(PublicationExportError::Io(format!(
                "staging directory already exists: {}",
                staging.display()
            )));
        }
        fs::create_dir(&staging).map_err(io_error)?;
        let result = self
            .write_staging(&staging)
            .and_then(|()| fs::rename(&staging, destination).map_err(io_error));
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    fn write_staging(&self, staging: &Path) -> Result<(), PublicationExportError> {
        for (name, bytes) in [
            ("table.json", self.table_json.as_slice()),
            ("table.csv", self.csv.as_slice()),
            ("table.tex", self.latex.as_slice()),
            ("manifest.json", self.manifest_json.as_slice()),
        ] {
            fs::write(staging.join(name), bytes).map_err(io_error)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum PublicationExportError {
    InvalidTable(String),
    Serialization(String),
    Io(String),
}

impl Display for PublicationExportError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTable(message) => write!(f, "invalid publication table: {message}"),
            Self::Serialization(message) => {
                write!(f, "publication export serialization failed: {message}")
            }
            Self::Io(message) => write!(f, "publication export I/O failed: {message}"),
        }
    }
}

impl Error for PublicationExportError {}

fn io_error(error: std::io::Error) -> PublicationExportError {
    PublicationExportError::Io(error.to_string())
}

fn validate_digest(value: &str, label: &str) -> Result<(), PublicationExportError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PublicationExportError::InvalidTable(format!(
            "{label} digest must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn csv_record<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    values
        .into_iter()
        .map(csv_escape)
        .collect::<Vec<_>>()
        .join(",")
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn latex_escape(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        output.push_str(match character {
            '\\' => "\\textbackslash{}",
            '&' => "\\&",
            '%' => "\\%",
            '$' => "\\$",
            '#' => "\\#",
            '_' => "\\_",
            '{' => "\\{",
            '}' => "\\}",
            '~' => "\\textasciitilde{}",
            '^' => "\\textasciicircum{}",
            _ => {
                output.push(character);
                continue;
            }
        });
    }
    output
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn manifest_artifact(path: &str, media_type: &str, bytes: &[u8]) -> PublicationExportArtifact {
    PublicationExportArtifact {
        path: path.to_owned(),
        media_type: media_type.to_owned(),
        sha256: sha256(bytes),
        byte_length: bytes.len() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> PublicationProvenance {
        PublicationProvenance {
            toolkit_version: "0.13.0".to_owned(),
            release_tag: "v0.13.0".to_owned(),
            source_revision: "abc1234".to_owned(),
            resolved_configuration_digest: "a".repeat(64),
            execution_fingerprint_digest: "b".repeat(64),
            input_artifact_digests: vec!["c".repeat(64)],
        }
    }

    fn table() -> PublicationTable {
        ccm_convergence_publication_table(
            "ccm-growth",
            "Finite CCM convergence path",
            &[ConvergenceTableRow {
                sequence_index: 1,
                lambda_squared: "13".to_owned(),
                n_modes: 120,
                precision_bits: 512,
                root_count: 50,
                minimum_accuracy_digits: "18.25".to_owned(),
                median_accuracy_digits: "31.5".to_owned(),
                index_penalty_digits: "4.75".to_owned(),
                completion_status: "successful".to_owned(),
            }],
            provenance(),
        )
        .unwrap()
    }

    #[test]
    fn convergence_exports_fixed_columns_and_provenance_without_rounding() {
        let table = table();
        let csv = table.render_csv().unwrap();
        assert!(csv.contains("lambda squared,N,p (bits),K,D min,D median,Index penalty,Status"));
        assert!(csv.contains("13,120,512,50,18.25,31.5,4.75,successful"));
        assert!(csv.contains("abc1234"));
        assert!(csv.contains("v0.13.0"));
        assert!(csv.contains(&"a".repeat(64)));
        let latex = table.render_latex().unwrap();
        assert!(latex.contains("% requirements: REP-010;CCM-012"));
        assert!(latex.contains("% release_tag: v0.13.0"));
        assert!(latex.contains("18.25 & 31.5 & 4.75"));
    }

    #[test]
    fn bundle_manifest_binds_every_rendered_artifact() {
        let bundle = table().export_bundle().unwrap();
        for (record, bytes) in bundle.manifest.artifacts.iter().zip([
            bundle.table_json.as_slice(),
            bundle.csv.as_slice(),
            bundle.latex.as_slice(),
        ]) {
            assert_eq!(record.sha256, sha256(bytes));
            assert_eq!(record.byte_length, bytes.len() as u64);
        }
    }

    #[test]
    fn write_new_is_atomic_and_refuses_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "xc-publication-export-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir(&root).unwrap();
        let destination = root.join("bundle");
        let bundle = table().export_bundle().unwrap();
        bundle.write_new(&destination).unwrap();
        assert_eq!(fs::read(destination.join("table.csv")).unwrap(), bundle.csv);
        assert!(bundle.write_new(&destination).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_or_extra_columns_fail_closed() {
        let mut table = table();
        table.rows[0].values.remove("root_count");
        table.rows[0]
            .values
            .insert("invented".to_owned(), "value".to_owned());
        assert!(table.validate().is_err());
    }

    #[test]
    fn publication_bundle_rejects_a_tag_that_does_not_match_its_toolkit_version() {
        let mut table = table();
        table.provenance.release_tag = "v0.14.0".to_owned();
        assert!(table.export_bundle().is_err());
    }
}
