"""Synthetic-only tests for the read-only cross-shard impact inventory."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("impact", Path(__file__).with_name("ccm_artifact_impact.py"))
impact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(impact)


def artifact(root, digit, kind, version, dependencies=(), active=True):
    digest = digit * 64
    path = root / "manifests" / digest[:2] / (digest + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artifact_family": "ccm-evidence",
        "semantic_key": {"artifact_kind": kind, "mathematical_semantics_version": version},
        "canonical_payload": {"dependencies": [{"manifest_digest": d * 64} for d in dependencies]},
    }))
    index = root / "indexes" / "ccm-evidence" / (digit + ".json")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"entries": [{"manifest_digest": digest, "disposition": "active" if active else "retired"}]}))
    return digest


class ImpactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "cache-repository.json").write_text(json.dumps({"repository": "synthetic/private-shard"}))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_children_of_defective_certificate_are_flagged(self):
        artifact(self.root, "a", "ccm_tau_matrix", "ordinary-gl-v1")
        artifact(self.root, "b", "ccm_sector_gap_certificate", "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2", ["a"])
        artifact(self.root, "c", "ccm_convergence_diagnostics", "summary-v1", ["b"])
        artifact(self.root, "d", "ccm_certificate_bundle", "ccm-exact-point-source-root-certificate-v0.13.0-v1", ["a"])
        report = impact.inventory([self.root])
        rows = {row["manifest_digest"][0]: row for row in report["artifacts"]}
        self.assertEqual(report["active_affected_count"], 2)
        self.assertEqual(rows["a"]["impact"], "not_flagged_by_this_defect_rule")
        self.assertEqual(rows["d"]["impact"], "not_flagged_by_this_defect_rule")
        self.assertTrue(report["coverage_complete_for_supplied_shards"])
        self.assertFalse(report["numerical_results_recomputed"])

    def test_unavailable_dependencies_block_complete_coverage(self):
        artifact(self.root, "a", "ccm_tau_matrix", "ordinary-v1", ["b"])
        report = impact.inventory([self.root])
        self.assertFalse(report["coverage_complete_for_supplied_shards"])
        self.assertEqual(report["missing_dependency_manifest_digests"], ["b" * 64])

    def test_retired_bad_certificate_is_not_counted_as_active(self):
        artifact(self.root, "a", "ccm_sector_gap_certificate", "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2", active=False)
        report = impact.inventory([self.root])
        self.assertEqual(report["directly_affected_count"], 1)
        self.assertEqual(report["active_affected_count"], 0)

    def test_unknown_certificate_semantics_require_review(self):
        artifact(self.root, "a", "ccm_sector_gap_certificate", "unrecognized-route")
        self.assertEqual(impact.inventory([self.root])["active_unrecognized_certificate_count"], 1)

    def test_requested_raw_byte_hash_check_detects_mismatch(self):
        artifact(self.root, "a", "ccm_tau_matrix", "ordinary-v1")
        report = impact.inventory([self.root], verify_manifest_bytes=True)
        self.assertTrue(report["metadata_errors"])
        self.assertFalse(report["coverage_complete_for_supplied_shards"])

    def test_duplicate_shard_roots_are_rejected(self):
        with self.assertRaises(ValueError):
            impact.inventory([self.root, self.root])


if __name__ == "__main__":
    unittest.main()
