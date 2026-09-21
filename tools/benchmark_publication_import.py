#!/usr/bin/env python3
"""Compare local Git import/packing policies on one verified retained part. No network."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part",type=Path,required=True)
    ap.add_argument("--sha256",required=True)
    ap.add_argument("--output",type=Path,required=True,help="New directory; retained for inspection")
    args=ap.parse_args();source=args.part.resolve()
    with source.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=args.sha256:
            raise SystemExit("Input digest mismatch")
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,policy in enumerate(["default","no-loose-compression","no-loose-compression","default"]):
        repo=args.output/("repo-"+str(index))
        subprocess.run(["git","init","--bare","--quiet",str(repo)],check=True)
        opts=[] if policy=="default" else ["-c","core.looseCompression=0"]
        start=time.perf_counter()
        oid=subprocess.check_output(["git","-C",str(repo),*opts,"hash-object","-w","--",str(source)]).decode().strip()
        ingest=time.perf_counter()-start
        pack=args.output/(str(index)+".pack");start=time.perf_counter()
        with pack.open("wb") as f:
            subprocess.run(["git","-C",str(repo),"-c","pack.window=0","pack-objects","--stdout"],input=(oid+"\n").encode(),stdout=f,stderr=subprocess.PIPE,check=True)
        packed=time.perf_counter()-start
        with pack.open("rb") as f:packhash=hashlib.file_digest(f,"sha256").hexdigest()
        row=dict(policy=policy,import_seconds=ingest,pack_seconds=packed,combined_seconds=ingest+packed,pack_bytes=pack.stat().st_size,pack_sha256=packhash,oid=oid)
        rows.append(row);print(json.dumps(row),flush=True)
        (args.output/"results.json").write_text(json.dumps(dict(scope="local verified-part import and packing only; no network",input_sha256=args.sha256,input_bytes=source.stat().st_size,rows=rows),indent=2)+"\n",encoding="utf-8")
    if len({r["oid"] for r in rows})!=1:raise SystemExit("Git object identity changed")
    if len({r["pack_sha256"] for r in rows})!=1:raise SystemExit("Packed bytes differ; inspect before claiming byte identity")
if __name__=="__main__":main()
