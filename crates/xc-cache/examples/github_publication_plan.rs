use xc_cache::{
    plan_github_publication, CacheNetworkRegistry, CacheRepositoryRegistry, CacheVisibility,
    GitHubRepositoryEndpoint, RepositoryShard, GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
};

fn main() {
    let capacity = CacheRepositoryRegistry {
        schema_version: 1,
        shards: vec![RepositoryShard {
            id: "weil-private-001".to_owned(),
            repository: "example-org/restricted-weil-shard-0001".to_owned(),
            visibility: CacheVisibility::Private,
            artifact_kinds: vec!["ccm_weil_eigenvector".to_owned()],
            reachable_payload_bytes: 62_000_000_000,
            estimated_history_bytes: 2_000_000_000,
            safe_payload_limit_bytes: GITHUB_SAFE_REPOSITORY_PAYLOAD_BYTES,
            writable: true,
        }],
    };
    let network = CacheNetworkRegistry {
        schema_version: 1,
        repositories: vec![GitHubRepositoryEndpoint {
            shard_id: "weil-private-001".to_owned(),
            owner: "TeamXcelerator".to_owned(),
            repository: "restricted-weil-shard-0001".to_owned(),
            branch: "main".to_owned(),
            visibility: CacheVisibility::Private,
            enabled_for_read: true,
            enabled_for_write: true,
            clone_via_ssh: true,
        }],
    };
    let plan = plan_github_publication(
        &capacity,
        &network,
        "ccm_weil_eigenvector",
        CacheVisibility::Private,
        3_000_000_000,
        300_000_000,
        false,
    )
    .unwrap();
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
