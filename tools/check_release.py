#!/usr/bin/env python3
"""Run offline local release checks and retain exit status, counts, and log hashes."""
import argparse, hashlib, json, os, re, subprocess, time, sys
from pathlib import Path
if not __debug__:
 raise SystemExit("release qualification requires Python without -O or PYTHONOPTIMIZE")
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tier", choices=["native", "hp"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--complete", action="store_true", help="also regenerate/check schemas, docs, research tools, digest, and HP backfill")
p.add_argument("--metadata-root", type=Path, help="optional sibling artifact repositories for full mirror/registration checks")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
if a.complete:
 from qualify_research_assets import validation_engine, source_digest, repository_bytes
 validation_engine()
 source_digest(require_committed=True)
 repository_bytes()
a.output.mkdir(parents=True, exist_ok=True)
feature = ["--features", "hp,arb"] if a.tier == "hp" else []
consumer = ["--features", "hp"] if a.tier == "hp" else []
profile = ["--release"] if a.tier == "hp" else []
checks = [
 ("workspace-tests", ["cargo", "test", "--workspace", "--all-targets", *profile, *feature, "--locked", "--offline"]),
 ("workspace-clippy", ["cargo", "clippy", "--workspace", "--all-targets", *feature, "--locked", "--offline", "--", "-D", "warnings"]),
 ("consumer-tests", ["cargo", "test", "--manifest-path", "tests/external-consumer/Cargo.toml", *consumer, "--locked", "--offline"]),
 ("consumer-clippy", ["cargo", "clippy", "--manifest-path", "tests/external-consumer/Cargo.toml", "--all-targets", *consumer, "--locked", "--offline", "--", "-D", "warnings"]),
 ("rustdoc", ["cargo", "doc", "--workspace", "--no-deps", *feature, "--locked", "--offline"]),
 ("doctests", ["cargo", "test", "--workspace", "--doc", *feature, "--locked", "--offline"]),
]
env = dict(os.environ, RUSTDOCFLAGS="-D warnings", XC_CACHE_REMOTE="none", XC_PUBLISH_EXECUTE="false")
records = []
for name, command in checks:
 print(f"START {name}", flush=True)
 log = a.output / (name + ".log")
 start = time.monotonic()
 with log.open("wb") as f:
  result = subprocess.run(command, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT)
 data = log.read_bytes()
 counts = re.findall(rb"test result: .*? (\d+) passed; (\d+) failed; (\d+) ignored", data)
 record = dict(check=name, command=command, exit_code=result.returncode, seconds=round(time.monotonic()-start,3), log_sha256=hashlib.sha256(data).hexdigest())
 if counts: record["tests"] = dict(zip(["passed","failed","ignored"], map(sum,zip(*(map(int,x) for x in counts)))))
 records.append(record)
 (a.output / "results.json").write_text(json.dumps(dict(tier=a.tier,checks=records),indent=2)+"\n")
 print(f"DONE {name}: {record}", flush=True)
 if result.returncode:
  print(data[-8000:].decode("utf8", errors="replace"), flush=True)
  raise SystemExit(result.returncode)

if a.complete:
    assets = [sys.executable, str(root / "tools/qualify_research_assets.py"), "--require-committed-source", "--output", str(a.output.resolve() / "assets")]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute(): target = root / target
    suffix = ".exe" if os.name == "nt" else ""
    if a.tier == "hp":
        subprocess.run(["cargo", "build", "-p", "xc-spectral", "--release", "--example", "ccm_research_backfill", "--features", "hp,arb", "--offline", "--locked"], cwd=root, env=env, check=True)
        assets += ["--backfill-binary", str(target / "release/examples" / ("ccm_research_backfill" + suffix))]
    if a.metadata_root:
        subprocess.run(["cargo", "build", "-p", "xc-cache", "--example", "audit_kind_registration", "--offline", "--locked"], cwd=root, env=env, check=True)
        assets += ["--metadata-root", str(a.metadata_root.resolve()), "--registration-binary", str(target / "debug/examples" / ("audit_kind_registration" + suffix))]
    subprocess.run(assets, cwd=root, env=env, check=True)
