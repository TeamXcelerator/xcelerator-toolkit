#!/usr/bin/env python3
"""Run offline local release checks and retain exit status, counts, and log hashes."""
import argparse, atexit, hashlib, json, os, re, shutil, subprocess, tempfile, time, sys
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
# Tests compute into a throwaway cache root so a release check never fills the
# operator's per-user cache; it is removed when the checks finish.
cache_root = Path(tempfile.mkdtemp(prefix="xc-release-cache-"))
atexit.register(shutil.rmtree, cache_root, True)
env = dict(os.environ, RUSTDOCFLAGS="-D warnings", XC_CACHE_REMOTE="none", XC_PUBLISH_EXECUTE="false", XC_CACHE_ROOT=str(cache_root))

def checkout_files():
 """Untracked and ignored paths, excluding Cargo build output (a target/ beside a Cargo.toml)."""
 out = subprocess.run(["git", "status", "--porcelain", "--ignored", "--untracked-files=all"], cwd=root, capture_output=True, text=True, check=True).stdout
 def build_output(path):
  head, sep, _ = ("/" + path).partition("/target/")
  return bool(sep) and (root / head.lstrip("/") / "Cargo.toml").is_file()
 return {line[3:] for line in out.splitlines() if line[:2] in ("??", "!!") and not build_output(line[3:])}

# Test scratch belongs in xc_core::test_support::TestDir, which lives outside
# the checkout and removes itself; nothing may write scratch into the tree.
in_tree_scratch = [str(f.relative_to(root)) for f in root.glob("crates/**/*.rs") if '"test-tmp"' in f.read_text(encoding="utf-8")]
if in_tree_scratch:
 raise SystemExit("in-checkout test scratch paths: " + ", ".join(in_tree_scratch))
before = checkout_files()
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

output = a.output.resolve()
own_output = output.relative_to(root).as_posix() + "/" if output.is_relative_to(root) else None
left_behind = sorted(f for f in checkout_files() - before if not (own_output and f.startswith(own_output)))
if (root / "target" / "test-tmp").exists():
 left_behind.append("target/test-tmp/")
if left_behind:
 raise SystemExit("release checks left files in the checkout: " + ", ".join(left_behind[:20]))

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
