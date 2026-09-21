#!/usr/bin/env python3
"""Synthetic band interruption/recovery and worker determinism benchmark; no network."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--atoms',type=int,default=4096);p.add_argument('--degree',type=int,default=24);a=p.parse_args()
 binary=a.binary.resolve();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
 if not 6<=a.degree<a.atoms:raise ValueError('require 6 <= degree < atoms')
 def enc(x):return json.dumps(x,separators=(',',':')).encode()
 def sha(x):return hashlib.sha256(x).hexdigest()
 payload=enc(dict(schema_version=3,lambda_squared='9',n_modes=1,precision_bits=192,force_even=True,eigenvalue='3',eigenvector=['0','1','0']))
 digest=sha(payload);version=dict(major=0,minor=15,patch=1,prerelease=None)
 manifest=dict(schema_version=1,key=dict(kind='ccm_weil_eigenpair',logical_key='synthetic-band-benchmark',parameters_digest=sha(b'synthetic-band-benchmark')),content_digest=digest,size_bytes=len(payload),objects=[dict(content_digest=digest,size_bytes=len(payload))],created_unix_seconds=1,producer_toolkit_version=version,minimum_reader_version=version,maximum_reader_version=None,quality='validated',visibility='local',immutable=True,dependencies=[],tags={},provenance_digest=None)
 (root/'state.json').write_bytes(payload);(root/'state.manifest.json').write_bytes(enc(manifest))
 atoms=b''.join(enc(dict(coordinate=str(j+1),signed_weight='1',family='zero'))+b'\n' for j in range(a.atoms));(root/'atoms.jsonl').write_bytes(atoms)
 policy=dict(maximum_atoms=a.atoms,maximum_input_bytes=len(atoms),band_chunks=[dict(relative_path='atoms.jsonl',sha256=sha(atoms),bytes=len(atoms),rows=a.atoms)],evaluations=[],cutoffs=[])
 model=dict(degree=a.degree,coordinate='positive integer fixture coordinate',definition_digest=sha(b'finite equal atom weights'),atoms=[],coverage='complete finite synthetic grid',hypotheses=['finite positive measure'],borrowed_inputs=[],input_energy=None,scoring_roots=[])
 external=enc(dict(schema_version=1,source_eigenpair=digest,lambda_squared='9',n_modes=1,precision_bits=192,convention_id='synthetic fixture',definition_digest=sha(b'benchmark'),approximation_scope='finite synthetic benchmark, not CCM data',run_once=dict(completion=dict(atom_analysis=policy,band=model))))
 (root/'external.json').write_bytes(external)
 task=dict(operation='extended_research',diagnostic='band_reconstruction',state=dict(manifest='state.manifest.json',payload='state.json'),input='external.json',input_sha256=sha(external))
 def plan(name):
  path=root/(name+'.json');path.write_bytes(enc(dict(schema_version=1,approved_payload_digests=[digest],cache_root='cache-budget' if name in ['limited','raised'] else 'cache-'+name,jobs=[dict(id='band',task=task)])));return path
 def run(name,workers,checkpoint,stop=False,budget=None):
  env=dict(os.environ,XC_RESEARCH_CHECKPOINT_DIR=str(root/checkpoint),XC_RESEARCH_SUMMARY_DIR=str(root/('summaries-'+name)),RAYON_NUM_THREADS=str(workers))
  if budget is not None:env['XC_RESEARCH_BASIS_BYTES']=str(budget)
  start=time.perf_counter();proc=subprocess.Popen([str(binary),str(plan(name)),str(root/('out-'+name))],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
  killed=False
  with (root/(name+'.log')).open('w') as log:
   for line in proc.stdout:
    log.write(line);log.flush()
    if stop and 'band recurrence 5/' in line and 'started' in line:
     proc.terminate();killed=True
   code=proc.wait()
  result=dict(name=name,workers=workers,seconds=time.perf_counter()-start,exit_code=code,interrupted=killed)
  if stop:
   assert killed,'did not reach interruption point'
  else:
   assert code==0,(name,code)
   packet=json.loads((root/('out-'+name)/'band.json').read_text());report=packet['report'];assert report['data']['outcome']==('unresolved' if budget==1 else 'point_measurement'),report['data'].get('reason');result['report_sha256']=sha(enc(report));result['outcome']=report['data']['outcome']
  return result
 limited=run('limited',1,'checkpoint-budget',budget=1);raised=run('raised',1,'checkpoint-budget')
 assert limited['report_sha256']!=raised['report_sha256'],'raising the budget reused the limited report'
 results=[run('interrupted',1,'checkpoint-shared',True),run('resumed',1,'checkpoint-shared'),run('cold-one',1,'checkpoint-cold-one'),run('cold-four',4,'checkpoint-cold-four')]
 assert len({r['report_sha256'] for r in results[1:]})==1,'resumption or worker count changed report'
 log=(root/'resumed.log').read_text();assert re.search(r'band recurrence resumed at degree [1-9]',log), 'recurrence did not resume'
 result=dict(scope='synthetic local finite signed-band benchmark; no campaign or network claim',atoms=a.atoms,degree=a.degree,precision_bits=256,identical_reports=True,resumed_from_saved_degree=True,budget_raise_recomputed=True,budget_runs=[limited,raised],runs=results)
 (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
