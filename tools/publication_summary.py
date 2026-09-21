#!/usr/bin/env python3
"""Summarize durable publication attempts. Timings overlap; bytes are not wire traffic."""
import argparse, collections, json
from pathlib import Path

def summarize(paths):
    attempts=[]
    for path in sorted(set(paths)):
        phases=collections.defaultdict(lambda:dict(calls=0,seconds=0.0,failed_calls=0,byte_counters={}))
        malformed=[];finished=None;pending=None
        with path.open(encoding='utf-8-sig') as stream:
            for number,line in enumerate(stream,1):
                try:
                    event=json.loads(line)
                    if event.get('schema_version')!=1 or not isinstance(event.get('details'),dict):raise ValueError('unsupported event')
                    phase=event['phase'];stats=phases[phase];stats['calls']+=1
                    seconds=float(event['elapsed_seconds'])
                    if not 0<=seconds<float('inf'):raise ValueError('invalid duration')
                    stats['seconds']+=seconds
                    detail=event['details']
                    if detail.get('success') is False:stats['failed_calls']+=1
                    for name,value in detail.items():
                        if 'bytes' in name and isinstance(value,int) and value>=0:
                            stats['byte_counters'][name]=stats['byte_counters'].get(name,0)+value
                        if 'remaining' in name or 'pending' in name:pending={name:value}
                    if phase=='attempt_finished':finished=detail.get('success')
                except (ValueError,KeyError,TypeError) as error:malformed.append(dict(line=number,reason=str(error)))
        attempts.append(dict(path=str(path),completion='succeeded' if finished is True else 'failed' if finished is False else 'incomplete_or_unknown',phases=dict(phases),latest_pending=pending,malformed=malformed))
    return dict(scope='phase timings overlap; do not sum phases into wall time; byte counters are not network throughput; canonical publication report governs success',attempts=attempts)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',nargs='+',type=Path);parser.add_argument('--output',type=Path);a=parser.parse_args()
    files=[p for source in a.paths for p in (source.rglob('attempt-*.jsonl') if source.is_dir() else [source])]
    result=summarize(files);text=json.dumps(result,indent=2)+'\n'
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    else:print(text,end='')
if __name__=='__main__':main()
