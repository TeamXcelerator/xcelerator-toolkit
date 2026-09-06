#!/usr/bin/env python3
"""Read-only inventory of locally available CCM artifact shard repositories.

This checks metadata and a named semantic defect rule. It is NOT a numerical
re-solve, package-hash audit, or certification of unflagged artifacts. It never
contacts GitHub, deletes objects, changes dispositions, or publishes a report.
Do not commit reports containing private manifest identities to public repos.

Usage: python3 tools/ccm_artifact_impact.py /path/to/shard1 /path/to/shard2 \
    --output /private/audit/impact.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

HEX = re.compile(r"^[0-9a-f]{64}$")
KNOWN_BAD_SEMANTICS = {
    "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v1",
    "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2",
}
CORRECTED_SEMANTICS = "ccm-cutoff-free-sector-gap-certificate-v0.14.4-v3"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def inventory(shards: list[Path], verify_manifest_bytes: bool = False) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one local shard root is required")
    records: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    metadata_errors: list[str] = []
    indexes_seen = 0
    manifests_seen = 0
    roots_seen: set[Path] = set()
    for original in shards:
        root = original.resolve(strict=True)
        if not root.is_dir() or root in roots_seen:
            raise ValueError(f"not a unique shard directory: {root}")
        roots_seen.add(root)
        config = read_json(root / "cache-repository.json")
        repository = config.get("repository", root.name)
        index_paths = sorted((root / "indexes").glob("*/*.json"))
        manifest_paths = sorted((root / "manifests").glob("*/*.json"))
        if not index_paths and manifest_paths:
            metadata_errors.append(f"{repository}: manifests exist but no managed indexes found")
        for path in index_paths:
            indexes_seen += 1
            try:
                data = read_json(path)
                entries = data.get("entries")
                if not isinstance(entries, list):
                    raise ValueError("index has no entries list")
                for item in entries:
                    digest = item.get("manifest_digest", "")
                    if not isinstance(digest, str) or not HEX.fullmatch(digest):
                        raise ValueError("invalid index manifest digest")
                    if item.get("disposition") == "active":
                        active.add(digest)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                metadata_errors.append(f"{repository}:{path.relative_to(root)}: {error}")
        for path in manifest_paths:
            manifests_seen += 1
            try:
                digest = path.stem
                if not HEX.fullmatch(digest):
                    raise ValueError("manifest filename is not a SHA-256 identity")
                raw = path.read_bytes()
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("manifest is not a JSON object")
                if verify_manifest_bytes and hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError("manifest raw-byte SHA-256 differs from filename")
                semantic = data.get("semantic_key")
                payload = data.get("canonical_payload")
                if not isinstance(semantic, dict) or not isinstance(payload, dict):
                    raise ValueError("manifest has no semantic key or canonical payload")
                dependencies = payload.get("dependencies")
                if not isinstance(dependencies, list):
                    raise ValueError("manifest dependencies are not a list")
                dependency_digests = []
                for dep in dependencies:
                    value = dep.get("manifest_digest", "")
                    if not isinstance(value, str) or not HEX.fullmatch(value):
                        raise ValueError("invalid dependency manifest digest")
                    dependency_digests.append(value)
                record = {
                    "manifest_digest": digest,
                    "repository": repository,
                    "path": str(path.relative_to(root)),
                    "artifact_family": data.get("artifact_family"),
                    "artifact_kind": semantic.get("artifact_kind"),
                    "mathematical_semantics_version": semantic.get("mathematical_semantics_version"),
                    "producer_toolkit_version": data.get("producer_toolkit_version"),
                    "dependencies": dependency_digests,
                    "manifest_byte_hash_checked": verify_manifest_bytes,
                }
                previous = records.get(digest)
                if previous is not None:
                    comparable = {key: value for key, value in record.items() if key not in {"repository", "path"}}
                    old = {key: value for key, value in previous.items() if key not in {"repository", "path"}}
                    if old != comparable:
                        raise ValueError("same manifest identity has inconsistent metadata across shards")
                else:
                    records[digest] = record
            except (OSError, ValueError, TypeError, AttributeError) as error:
                metadata_errors.append(f"{repository}:{path.relative_to(root)}: {error}")

    missing_active = sorted(active - records.keys())
    reverse: dict[str, set[str]] = defaultdict(set)
    missing_dependencies: set[str] = set()
    directly_affected: set[str] = set()
    needs_review: set[str] = set()
    for digest, record in records.items():
        for dep in record["dependencies"]:
            reverse[dep].add(digest)
            if dep not in records:
                missing_dependencies.add(dep)
        if record["artifact_kind"] == "ccm_sector_gap_certificate":
            version = record["mathematical_semantics_version"]
            if version in KNOWN_BAD_SEMANTICS:
                directly_affected.add(digest)
            elif version != CORRECTED_SEMANTICS:
                needs_review.add(digest)

    affected = set(directly_affected)
    queue = deque(sorted(directly_affected))
    while queue:
        parent = queue.popleft()
        for child in sorted(reverse.get(parent, ())):
            if child not in affected:
                affected.add(child)
                queue.append(child)
    listed = []
    for digest in sorted(records):
        record = records[digest].copy()
        record["active"] = digest in active
        record["impact"] = (
            "known_defective_assembly_semantics" if digest in directly_affected
            else "depends_on_known_defective_artifact" if digest in affected
            else "unrecognized_certificate_semantics_requires_review" if digest in needs_review
            else "not_flagged_by_this_defect_rule"
        )
        listed.append(record)
    return {
        "schema_version": 1,
        "scope": "read_only_metadata_impact_inventory_not_numerical_validation",
        "defect_rule": "cutoff_free_zero_mode_omitted_finite_endpoint_correction",
        "shards_requested": len(shards),
        "indexes_read": indexes_seen,
        "manifest_files_read": manifests_seen,
        "unique_manifests": len(records),
        "active_manifests": len(active),
        "kind_counts": dict(sorted(Counter(r["artifact_kind"] for r in records.values()).items())),
        "directly_affected_count": len(directly_affected),
        "affected_including_descendants_count": len(affected),
        "active_affected_count": len(active & affected),
        "active_unrecognized_certificate_count": len(active & needs_review),
        "missing_active_manifest_digests": missing_active,
        "missing_dependency_manifest_digests": sorted(missing_dependencies),
        "metadata_errors": metadata_errors,
        "coverage_complete_for_supplied_shards": not metadata_errors and not missing_active and not missing_dependencies,
        "payload_packages_verified": False,
        "numerical_results_recomputed": False,
        "artifacts": listed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-manifest-bytes", action="store_true",
                        help="also hash raw manifest files; does not verify numerical packages")
    args = parser.parse_args()
    try:
        report = inventory(args.shards, args.verify_manifest_bytes)
        output = args.output.resolve()
        if any(output == root.resolve() or root.resolve() in output.parents for root in args.shards):
            raise ValueError("write the audit report outside the immutable shard directories")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    except (OSError, ValueError) as error:
        print(f"inventory failed: {error}", file=sys.stderr)
        return 2
    print(f"Read {report['unique_manifests']} unique manifests across {report['shards_requested']} shards.")
    print(f"Active affected: {report['active_affected_count']}; active certificate semantics needing review: {report['active_unrecognized_certificate_count']}.")
    print("Unflagged does not mean numerically verified.")
    if (report["active_affected_count"] or report["active_unrecognized_certificate_count"]
            or not report["coverage_complete_for_supplied_shards"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
