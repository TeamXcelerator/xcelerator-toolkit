#!/usr/bin/env python3
"""Compare local Git import/packing policies on one verified retained part. No network."""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


def fingerprint(stream, limit=None):
    """Hash actual bytes read; stop before accepting an oversized imported blob."""
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            return digest.hexdigest(), count
        count += len(chunk)
        if limit is not None and count > limit:
            raise SystemExit("Imported object size differs from verified input")
        digest.update(chunk)


def verify_import(repo, oid, expected_digest, expected_size):
    """Verify the bytes Git actually imported, independently of the mutable input path."""
    with subprocess.Popen(
        ["git", "--no-replace-objects", "-C", str(repo), "cat-file", "blob", oid],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as process:
        try:
            digest, count = fingerprint(process.stdout, expected_size)
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdout.close()
        status = process.wait()
    if status:
        raise SystemExit("Imported object cannot be read")
    if count != expected_size:
        raise SystemExit("Imported object size differs from verified input")
    if digest != expected_digest:
        raise SystemExit("Imported object digest differs from verified input")


def write_results(output, digest, size, rows, complete):
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(dict(
        scope="local verified-part import and packing only; no network",
        input_sha256=digest, input_bytes=size, complete=complete, rows=rows,
    ), indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True, help="New directory; retained for inspection")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise SystemExit("Expected digest must be a lowercase SHA-256 value")
    source = args.part.resolve()
    with source.open("rb") as stream:
        digest, size = fingerprint(stream)
    if digest != args.sha256:
        raise SystemExit("Input digest mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, policy in enumerate(["default", "no-loose-compression", "no-loose-compression", "default"]):
        repo = args.output / ("repo-" + str(index))
        subprocess.run(["git", "init", "--bare", "--quiet", str(repo)], check=True)
        opts = [] if policy == "default" else ["-c", "core.looseCompression=0"]
        start = time.perf_counter()
        oid = subprocess.check_output([
            "git", "-C", str(repo), *opts, "hash-object", "-w", "--", str(source),
        ]).decode().strip()
        ingest = time.perf_counter() - start
        # Verification is outside the import timing and precedes reporting.
        verify_import(repo, oid, digest, size)
        pack = args.output / (str(index) + ".pack")
        start = time.perf_counter()
        with pack.open("wb") as stream:
            subprocess.run([
                "git", "-C", str(repo), "-c", "pack.window=0", "pack-objects", "--stdout",
            ], input=(oid + "\n").encode(), stdout=stream, stderr=subprocess.PIPE, check=True)
        packed = time.perf_counter() - start
        with pack.open("rb") as stream:
            packhash, packsize = fingerprint(stream)
        row = dict(policy=policy, import_seconds=ingest, pack_seconds=packed,
                   combined_seconds=ingest + packed, pack_bytes=packsize,
                   pack_sha256=packhash, oid=oid)
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_results(args.output, digest, size, rows, False)
    if len({row["oid"] for row in rows}) != 1:
        raise SystemExit("Git object identity changed")
    if len({row["pack_sha256"] for row in rows}) != 1:
        raise SystemExit("Packed bytes differ; inspect before claiming byte identity")
    write_results(args.output, digest, size, rows, True)


if __name__ == "__main__":
    main()
