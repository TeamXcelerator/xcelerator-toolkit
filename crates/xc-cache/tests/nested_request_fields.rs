use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use xc_cache::{CacheNetworkRegistry, GitHubRepositoryEndpoint, ToolkitVersion};

fn strict<T: DeserializeOwned>(valid: Value) {
    let decoded: T = serde_json::from_value(valid.clone()).expect("valid shape remains accepted");
    drop(decoded);
    for field in ["prerelase", "unexpected_policy", "precision_bit"] {
        let mut bad = valid.clone();
        bad.as_object_mut()
            .unwrap()
            .insert(field.into(), json!("manufactured typo"));
        assert!(serde_json::from_value::<T>(bad)
            .err()
            .expect("unknown fields must fail")
            .to_string()
            .contains("unknown field"));
    }
}

#[test]
fn every_nested_request_type_rejects_unrecognized_fields() {
    strict::<ToolkitVersion>(json!({"major":1,"minor":0,"patch":0,"prerelease":"rc.3"}));
    strict::<ToolkitVersion>(json!({"major":1,"minor":0,"patch":0}));
    let endpoint = json!({"shard_id":"manufactured","owner":"fixture","repository":"fixture","branch":"main","visibility":"local","enabled_for_read":true,"enabled_for_write":false,"clone_via_ssh":false});
    strict::<GitHubRepositoryEndpoint>(endpoint.clone());
    strict::<CacheNetworkRegistry>(json!({"schema_version":1,"repositories":[]}));
    let mut bad = endpoint;
    bad["enabled_for_wrtie"] = json!(true);
    assert!(serde_json::from_value::<CacheNetworkRegistry>(
        json!({"schema_version":1,"repositories":[bad]})
    )
    .is_err());
}
