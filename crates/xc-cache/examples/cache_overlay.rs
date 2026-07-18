use std::collections::BTreeMap;
use std::path::PathBuf;
use xc_cache::{
    ArtifactDraft, ArtifactKey, CacheLayer, CachePolicy, CacheQuality, CacheResolver, CacheStore,
    CacheVisibility, FilesystemCacheStore, ToolkitVersion,
};

fn version(value: &str) -> ToolkitVersion {
    ToolkitVersion::parse(value).expect("valid version")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from("target/example-cache-overlay");
    let local = FilesystemCacheStore::new(
        "local-private",
        root.join("private"),
        true,
        CacheVisibility::Private,
    );
    let public = FilesystemCacheStore::new(
        "published-public",
        root.join("public"),
        true,
        CacheVisibility::Public,
    );
    let key = ArtifactKey::new(
        "ccm.weil_eigenvector",
        "lambda_sq=13/n=120/prec=3338/even",
        br#"{"lambda_sq":13,"n":120,"precision_bits":3338,"subspace":"even"}"#,
    )?;
    let draft = |quality, visibility| ArtifactDraft {
        schema_version: 1,
        key: key.clone(),
        producer_toolkit_version: version("0.13.0"),
        minimum_reader_version: version("0.13.0"),
        maximum_reader_version: None,
        quality,
        visibility,
        immutable: true,
        dependencies: Vec::new(),
        tags: BTreeMap::new(),
        provenance_digest: None,
    };
    local.put(
        &draft(CacheQuality::Validated, CacheVisibility::Private),
        b"private work-in-progress payload",
    )?;
    public.put(
        &draft(CacheQuality::Published, CacheVisibility::Public),
        b"reviewed public payload",
    )?;

    let resolver = CacheResolver::new(vec![
        CacheLayer {
            precedence: 0,
            store: Box::new(local),
        },
        CacheLayer {
            precedence: 10,
            store: Box::new(public),
        },
    ]);
    let policy = CachePolicy {
        current_toolkit_version: version("0.13.0"),
        minimum_quality: CacheQuality::Published,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Private, CacheVisibility::Public],
    };
    let artifact = resolver.resolve(&key, &policy)?;
    println!(
        "resolved {} from {} with quality {:?}",
        artifact.manifest.content_digest, artifact.layer_name, artifact.manifest.quality
    );
    Ok(())
}
