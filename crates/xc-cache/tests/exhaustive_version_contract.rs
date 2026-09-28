use xc_cache::ToolkitVersion;
#[test]
fn prerelease_numeric_identifiers_follow_numeric_precedence() {
    let low = ToolkitVersion::parse("1.0.0-beta.2").unwrap();
    let high = ToolkitVersion::parse("1.0.0-beta.11").unwrap();
    assert!(low < high);
}
#[test]
fn numeric_prerelease_identifiers_precede_nonnumeric_identifiers() {
    assert!(ToolkitVersion::parse("1.0.0-1").unwrap() < ToolkitVersion::parse("1.0.0--a").unwrap());
}
#[test]
fn version_parser_rejects_malformed_numeric_and_prerelease_fields() {
    for text in [
        "01.0.0",
        "+1.0.0",
        "1.0.0-a..b",
        "1.0.0-01",
        "1.0.0-a/b",
        "1.0.0- ",
    ] {
        assert!(ToolkitVersion::parse(text).is_err(), "accepted {text}");
    }
}

#[test]
fn compatibility_maximum_cannot_be_removed_by_manifest() {
    let mut policy = xc_cache::artifact_family_compatibility_policy("fixture").unwrap();
    policy.maximum_reader_version = Some(ToolkitVersion::parse("0.13.9").unwrap());
    let v = ToolkitVersion::parse("0.13.0").unwrap();
    assert!(policy.validate_manifest_versions(1, &v, &v, None).is_err());
}
#[test]
fn compatibility_rejects_reversed_manifest_reader_window() {
    let policy = xc_cache::artifact_family_compatibility_policy("fixture").unwrap();
    let v = ToolkitVersion::parse("0.13.0").unwrap();
    let high = ToolkitVersion::parse("0.14.0").unwrap();
    assert!(policy
        .validate_manifest_versions(1, &v, &high, Some(&v))
        .is_err());
}
#[test]
fn compatibility_revalidates_directly_constructed_versions() {
    let policy = xc_cache::artifact_family_compatibility_policy("fixture").unwrap();
    let mut v = ToolkitVersion::parse("0.14.0").unwrap();
    v.prerelease = Some("bad..version".into());
    assert!(policy.validate_manifest_versions(1, &v, &v, None).is_err());
}

#[test]
fn standard_prerelease_precedence_sequence_is_preserved() {
    let values = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    .map(|s| ToolkitVersion::parse(s).unwrap());
    assert!(values.windows(2).all(|p| p[0] < p[1]));
    assert!(
        ToolkitVersion::parse("1.0.0-999999999999999999999999999999").unwrap()
            < ToolkitVersion::parse("1.0.0-1000000000000000000000000000000").unwrap()
    );
}
