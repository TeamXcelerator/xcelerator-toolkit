use xc_core::{ccm_convergence_publication_table, ConvergenceTableRow, PublicationProvenance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provenance = PublicationProvenance {
        toolkit_version: env!("CARGO_PKG_VERSION").to_owned(),
        release_tag: format!("v{}", env!("CARGO_PKG_VERSION")),
        source_revision: "example-revision".to_owned(),
        resolved_configuration_digest: "a".repeat(64),
        execution_fingerprint_digest: "b".repeat(64),
        input_artifact_digests: vec!["c".repeat(64)],
    };
    let table = ccm_convergence_publication_table(
        "ccm-convergence-example",
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
        provenance,
    )?;
    let bundle = table.export_bundle()?;
    println!("{}", String::from_utf8(bundle.manifest_json)?);
    Ok(())
}
