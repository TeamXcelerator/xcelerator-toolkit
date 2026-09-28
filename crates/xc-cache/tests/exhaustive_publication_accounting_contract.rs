use std::collections::BTreeMap;
use xc_cache::*;
fn record() -> PayloadBatchRecord {
    let key = ContentDigest::sha256(b"transaction");
    PayloadBatchRecord {
        schema_version: 1,
        transaction_id: key.0.clone(),
        idempotency_key: key,
        destination: PublicationDestination::Public,
        authorized_repository: "owner/repo".into(),
        shard_id: "shard".into(),
        branch: "main".into(),
        sequence: 0,
        payload_parent_head: "parent".into(),
        payload_commit_id: "commit".into(),
        planned_payload_bytes: 4,
        newly_committed_payload_bytes: 4,
        objects: vec![PayloadBatchObjectRecord {
            repository_path: "objects/a".into(),
            size_bytes: 4,
            content_digest: ContentDigest::sha256(b"ABCD"),
            newly_introduced: true,
        }],
    }
}
#[test]
fn distinct_paths_with_identical_bytes_count_one_new_blob() {
    let mut r = record();
    let mut second = r.objects[0].clone();
    second.repository_path = "objects/b".into();
    r.objects.push(second);
    r.planned_payload_bytes = 8;
    r.schema_version = 2;
    assert!(
        r.validate().is_ok(),
        "valid physical paths may share one content blob: {:?}",
        r.validate()
    );
}
#[test]
fn payload_batch_identity_must_be_a_digest() {
    let mut r = record();
    r.transaction_id = "../../outside".into();
    r.idempotency_key = ContentDigest(r.transaction_id.clone());
    assert!(r.validate().is_err());
}
fn ledger() -> CapacityLedger {
    CapacityLedger {
        schema_version: 1,
        shard_id: "shard".into(),
        hard_capacity_bytes: 1000,
        warning_reserve_bytes: 100,
        first_seen_immutable_payload_bytes: 0,
        manifest_index_receipt_bytes: 0,
        estimated_history_bytes: 0,
        emergency_reserve_bytes: 10,
        abandoned_reachable_bytes: 0,
        last_reconciled_commit: "commit".into(),
        reconciliation_digest: ContentDigest::sha256(b"reconcile"),
    }
}
#[test]
fn current_capacity_total_cannot_be_silently_saturated() {
    let mut l = ledger();
    l.first_seen_immutable_payload_bytes = u64::MAX;
    assert!(l.assess_addition(0, 0, 0).is_err());
}
#[test]
fn projected_capacity_total_cannot_be_silently_saturated() {
    assert!(ledger().assess_addition(u64::MAX, 0, 0).is_err());
}
#[test]
fn empty_transaction_is_never_complete() {
    let d = ContentDigest::sha256(b"identity");
    let j = PublicationTransactionJournal {
        schema_version: 1,
        transaction_id: d.0.clone(),
        idempotency_key: d.clone(),
        semantic_digest: d.clone(),
        target_manifest_digests: BTreeMap::new(),
        payload_digest: d.clone(),
        policy_digest: d,
        targets: BTreeMap::new(),
    };
    assert!(!j.complete());
}

#[test]
fn repeated_blob_size_or_newness_must_agree() {
    let mut r = record();
    let mut b = r.objects[0].clone();
    b.repository_path = "objects/b".into();
    r.objects.push(b);
    r.schema_version = 2;
    r.planned_payload_bytes = 8;
    r.objects[1].newly_introduced = false;
    assert!(r.validate().is_err());
    r.objects[1].newly_introduced = true;
    r.objects[1].size_bytes = 5;
    r.planned_payload_bytes = 9;
    assert!(r.validate().is_err());
}
