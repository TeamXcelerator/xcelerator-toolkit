//! Fresh witnesses for archive metadata validation boundaries.
use xc_core::*;

fn manifest() -> ScholarlyArchiveManifest {
    let roles = [
        ScholarlyArchiveArtifactRole::TaggedSource,
        ScholarlyArchiveArtifactRole::DependencyLock,
        ScholarlyArchiveArtifactRole::Requirements,
        ScholarlyArchiveArtifactRole::TechnicalDesign,
        ScholarlyArchiveArtifactRole::ReleaseNotes,
        ScholarlyArchiveArtifactRole::Traceability,
        ScholarlyArchiveArtifactRole::CitationMetadata,
        ScholarlyArchiveArtifactRole::License,
        ScholarlyArchiveArtifactRole::ReproducibilityManifest,
        ScholarlyArchiveArtifactRole::TrustSnapshot,
        ScholarlyArchiveArtifactRole::CertificateBundle,
        ScholarlyArchiveArtifactRole::EssentialReferenceArtifact,
    ];
    ScholarlyArchiveManifest {
        schema_version: 1,
        release: ScholarlyReleaseMetadata {
            title: "Audit fixture".into(),
            version: "1.0.0".into(),
            release_date: "2026-09-28".into(),
            abstract_text: "Finite fixture".into(),
            license_identifier: "other".into(),
            repository: "https://example.invalid/repo".into(),
            authors: vec![ArchiveAuthor {
                given_names: "Audit".into(),
                family_names: "Fixture".into(),
                name_suffix: None,
                orcid: "https://orcid.org/0009-0003-9724-3104".into(),
            }],
            keywords: vec!["fixture".into()],
            preferred_citation: "Audit fixture".into(),
        },
        tag: "v1.0.0".into(),
        source_revision: "a".repeat(40),
        dependency_lock_digest: "c".repeat(64),
        requirements_digest: "c".repeat(64),
        technical_design_digest: "c".repeat(64),
        trust_snapshot_digest: "c".repeat(64),
        artifacts: roles
            .into_iter()
            .enumerate()
            .map(|(i, role)| ScholarlyArchiveArtifact {
                path: format!("artifact-{i:02}"),
                role,
                media_type: "application/octet-stream".into(),
                sha256: "c".repeat(64),
                byte_length: 1,
            })
            .collect(),
        created_at_utc: "2026-09-28T12:00:00Z".into(),
        finite_claim_statement: "Finite fixture only".into(),
    }
}

#[test]
fn archive_provenance_digests_must_bind_their_artifact_roles() {
    let mut m = manifest();
    m.validate().unwrap();
    m.dependency_lock_digest = "b".repeat(64);
    assert_ne!(m.dependency_lock_digest, m.artifacts[1].sha256);
    assert!(m.validate().is_err());
    let mut m = manifest();
    let mut duplicate = m.artifacts[1].clone();
    duplicate.path = "last-duplicate-lock".into();
    m.artifacts.push(duplicate);
    assert!(
        m.validate().is_err(),
        "a singular provenance digest cannot ambiguously bind two artifacts"
    );
}

fn receipt(plan: &ScholarlyArchivePlan) -> ArchiveDepositReceipt {
    ArchiveDepositReceipt {
        schema_version: 1,
        manifest_digest: plan.manifest_digest.clone(),
        tag: plan.manifest.tag.clone(),
        archive_provider: "fixture".into(),
        record_id: "1".into(),
        provider_record_url: "https://example.invalid/1".into(),
        provider_evidence_sha256: "d".repeat(64),
        doi: "10.1234/fixture".into(),
        deposited_at_utc: "2026-09-28T12:00:00.1Z".into(),
        verified_at_utc: "2026-09-28T12:00:01Z".into(),
        immutable: true,
        objects: plan
            .manifest
            .artifacts
            .iter()
            .map(|a| ArchiveDepositObject {
                path: a.path.clone(),
                sha256: a.sha256.clone(),
                byte_length: a.byte_length,
            })
            .collect(),
    }
}

#[test]
fn valid_fractional_second_deposit_must_not_be_rejected_as_earlier() {
    let plan = build_scholarly_archive_plan(manifest()).unwrap();
    let receipt = receipt(&plan);
    assert!(
        verify_archive_deposit_receipt(&plan, &receipt).is_ok(),
        "12:00:00.1 UTC is later than 12:00:00 UTC"
    );
}

#[test]
fn timestamp_comparison_preserves_all_fraction_digits() {
    let mut m = manifest();
    m.created_at_utc = "2026-09-28T12:00:00.1000000000000000000001Z".into();
    let plan = build_scholarly_archive_plan(m).unwrap();
    let mut r = receipt(&plan);
    assert!(verify_archive_deposit_receipt(&plan, &r).is_err());
    r.deposited_at_utc = "2026-09-28T12:00:00.10000000000000000000010Z".into();
    r.verified_at_utc = "2026-09-28T12:00:00.100000000000000000000100Z".into();
    verify_archive_deposit_receipt(&plan, &r).unwrap();
    r.verified_at_utc = "2026-09-28T12:00:00.1Z".into();
    assert!(verify_archive_deposit_receipt(&plan, &r).is_err());
}

#[test]
fn invalid_calendar_dates_and_time_labels_fail_closed() {
    for date in [
        "2026-02-29",
        "1900-02-29",
        "2026-04-31",
        "2026-00-01",
        "2026-01-00",
        "0000-01-01",
    ] {
        let mut m = manifest();
        m.release.release_date = date.into();
        assert!(m.validate().is_err(), "accepted invalid date {date}");
    }
    for date in ["2000-02-29", "2024-02-29", "2026-12-31"] {
        let mut m = manifest();
        m.release.release_date = date.into();
        m.validate().unwrap();
    }
    for time in [
        "24:00:00Z",
        "12:60:00Z",
        "12:00:60Z",
        "12:00:00.Z",
        "12:00:00.aZ",
        "12:00:00+00:00",
        "12-00-00Z",
    ] {
        let mut m = manifest();
        m.created_at_utc = format!("2026-09-28T{time}");
        assert!(m.validate().is_err(), "accepted invalid timestamp {time}");
    }
}
