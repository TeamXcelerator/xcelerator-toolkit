#!/usr/bin/env python3
"""Destructive-maintenance regressions using disposable local Git fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


SCRIPT = Path(__file__).with_name("manage_disk_space.ps1").resolve()
POWERSHELL = shutil.which("powershell.exe")


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows maintenance command")
class DiskMaintenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xc-disk-maintenance-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "--quiet")
        self.journal = self.repo / "target" / "old-run"
        self.transport = self.journal / "git-transport"
        self.bare = self.transport / ("a" * 64) / "remote.git"
        self.git("init", "--quiet", "--bare", str(self.bare))
        self.durable = self.journal / "family-batches" / "receipt.json"
        self.durable.parent.mkdir()
        self.durable.write_bytes(b'{"completed":true}\n')
        self.age()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def age(self):
        old = time.time() - 30 * 86400
        for path in [*self.transport.rglob("*"), self.transport]:
            os.utime(path, (old, old))

    def run_tool(self, *args):
        return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "RemoteSigned", "-File", str(SCRIPT),
                               "-RepositoryRoot", str(self.repo), *args],
                              capture_output=True, text=True)

    def clean(self, journal="target/old-run"):
        return self.run_tool("-CleanTransport", "-JournalRoot", journal)

    def test_report_does_not_delete(self):
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["transports"][0]["status"], "inspected")
        self.assertTrue(self.transport.exists())

    def test_default_root_is_the_scripts_checkout(self):
        copied = self.repo / "tools" / SCRIPT.name
        copied.parent.mkdir()
        shutil.copyfile(SCRIPT, copied)
        result = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "RemoteSigned",
                                 "-File", str(copied)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(json.loads(result.stdout)["repository"]), self.repo)

    def test_cleanup_preserves_durable_records_and_writes_receipt(self):
        before = self.durable.read_bytes()
        result = self.clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(report["removed_bytes"], 0)
        self.assertFalse(self.transport.exists())
        self.assertEqual(self.durable.read_bytes(), before)
        receipt = json.loads(Path(report["receipt"]).read_text(encoding="utf-8-sig"))
        self.assertEqual(receipt["transports"][0]["status"], "removed")

    def test_cleanup_recognizes_interrupted_git_index(self):
        index = self.bare.parent / ("publication-index-" + "b" * 64 + ".lock")
        index.touch()
        self.age()
        result = self.clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.transport.exists())
        self.assertTrue(self.durable.exists())

    def test_requires_explicit_selection(self):
        result = self.run_tool("-CleanTransport")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires explicit", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_rejects_outside_target(self):
        result = self.clean("target/../../outside")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("strictly inside", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_rejects_recent_transport(self):
        (self.bare / "HEAD").touch()
        result = self.clean()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("modified recently", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_rejects_unrecognized_files(self):
        (self.transport / "research.json").write_text("precious")
        self.age()
        result = self.clean()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unrecognized transport contents", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_report_keeps_totals_when_transport_needs_review(self):
        (self.transport / "research.json").write_text("precious")
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertGreater(report["total_bytes"], 0)
        self.assertEqual(report["transports"][0]["status"], "review_required")
        self.assertTrue(self.transport.exists())

    def test_rejects_tracked_transport(self):
        tracked = self.repo / "target" / "tracked" / "git-transport"
        tracked.mkdir(parents=True)
        (tracked / "precious.txt").write_text("tracked")
        self.git("add", "target/tracked/git-transport/precious.txt")
        result = self.clean("target/tracked")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracked files", result.stderr)
        self.assertTrue(tracked.exists())

    def test_rejects_local_tags(self):
        blob = subprocess.run(["git", "--git-dir=" + str(self.bare), "hash-object", "-w", "--stdin"],
                              input="keep", text=True, capture_output=True, check=True).stdout.strip()
        self.git("--git-dir=" + str(self.bare), "update-ref", "refs/tags/keep", blob)
        self.age()
        result = self.clean()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local branches or tags", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_rejects_live_publisher_lock(self):
        import msvcrt
        with (self.journal / "git-transport.lock").open("w+b") as lock:
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                result = self.clean()
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(self.transport.exists())
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)

    def test_rejects_junction(self):
        alias = self.repo / "target" / "alias"
        command = f"New-Item -ItemType Junction -Path '{alias}' -Target '{self.journal}' | Out-Null"
        subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], check=True, capture_output=True)
        result = self.clean("target/alias")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reparse point", result.stderr)
        self.assertTrue(self.transport.exists())

    def test_rejects_nested_junction_and_preserves_destination(self):
        alias = self.transport / "linked-evidence"
        command = f"New-Item -ItemType Junction -Path '{alias}' -Target '{self.durable.parent}' | Out-Null"
        subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], check=True, capture_output=True)
        result = self.clean()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reparse point", result.stderr)
        self.assertTrue(self.durable.exists())


if __name__ == "__main__":
    unittest.main()
