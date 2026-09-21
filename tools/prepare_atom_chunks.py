#!/usr/bin/env python3
"""Split a data-only atom JSONL file into authenticated chunks without changing scalar text."""
import argparse,hashlib,json
from pathlib import Path

def prepare(source,destination,kind,chunk_rows=10000,maximum_bytes=8<<30):
    if chunk_rows<=0 or maximum_bytes<=0:raise ValueError('limits must be positive')
    destination.mkdir(parents=True,exist_ok=False)
    fields={'coordinate','signed_weight','family'} if kind=='band' else {'ordinal','coordinate','weight','family','partition'}
    chunks=[];writer=None;total=0;count=0;chunk_count=0;digest=None;size=0
    def close():
        nonlocal writer
        if writer:
            writer.close();writer=None
            chunks.append(dict(relative_path=f'part-{len(chunks):06d}.jsonl',sha256=digest.hexdigest(),bytes=size,rows=chunk_count))
    try:
        with source.open('rb') as stream:
            while True:
                line=stream.readline((1<<20)+1)
                if not line:break
                if len(line)>1<<20:raise ValueError('atom line exceeds 1 MiB')
                value=json.loads(line)
                if not isinstance(value,dict) or set(value)!=fields:raise ValueError('atom row has incorrect fields')
                if not all(isinstance(value[k],str) for k in fields-{'ordinal'}):raise ValueError('atom scalar and label fields must be strings')
                if kind=='weighted' and (type(value['ordinal']) is not int or value['ordinal']<=0):raise ValueError('ordinal must be a positive integer')
                line=line.rstrip(b'\r\n')+b'\n';total+=len(line)
                if total>maximum_bytes:raise ValueError('input byte budget exceeded')
                if writer is None:
                    writer=(destination/f'part-{len(chunks):06d}.jsonl').open('xb');digest=hashlib.sha256();size=0;chunk_count=0
                writer.write(line);digest.update(line);size+=len(line);chunk_count+=1;count+=1
                if chunk_count==chunk_rows:close()
        close()
        if not count:raise ValueError('empty atom table')
        policy=dict(maximum_atoms=count,maximum_input_bytes=total,weighted_chunks=chunks if kind=='weighted' else [],band_chunks=chunks if kind=='band' else [],evaluations=[],cutoffs=[])
        (destination/'atom-analysis.json').write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
        return policy
    finally:
        if writer:writer.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=['weighted','band'],required=True);p.add_argument('--rows-per-chunk',type=int,default=10000);p.add_argument('--maximum-input-bytes',type=int,default=8<<30);a=p.parse_args()
    result=prepare(a.input,a.output,a.kind,a.rows_per_chunk,a.maximum_input_bytes)
    print(json.dumps(dict(rows=result['maximum_atoms'],bytes=result['maximum_input_bytes'],policy=str(a.output/'atom-analysis.json'),scope='byte-authenticated data preparation; numerical validity checked by the Toolkit')))
if __name__=='__main__':main()
