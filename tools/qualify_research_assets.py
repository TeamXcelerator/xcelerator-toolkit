#!/usr/bin/env python3
"""Offline release asset qualification; no GitHub writes or scientific campaigns."""
import argparse, hashlib, importlib.metadata, json, os, re, subprocess, sys, tempfile
from pathlib import Path
if not __debug__:
 raise SystemExit("release qualification requires Python without -O or PYTHONOPTIMIZE")
ROOT=Path(__file__).resolve().parents[1]
def require(condition, message):
 if not condition:
  raise ValueError(message)

def source_digest(root=ROOT, require_committed=False):
 def git(*args):
  return subprocess.check_output(['git','-C',str(root),'-c','safe.directory='+root.as_posix(),*args])
 pins_path=root/'tools/byte_exact_inputs.json'
 pins=json.loads(pins_path.read_text(encoding='utf-8')) if pins_path.exists() else {}
 def normalize(name,raw):return raw if name in pins else raw.replace(b'\r\n',b'\n')
 def source(name):
  return name in pins or name in {'.gitattributes','tools/byte_exact_inputs.json','tools/handmaintained_schemas.json','tools/requirements-validation.txt'} or name.startswith('tools/') and name.endswith('.py') or Path(name).suffix in {'.rs','.c','.h'} or Path(name).name in {'Cargo.toml','Cargo.lock','compact_field_policy.txt'}
 tracked=set(git('ls-files','--cached','-z').decode().split('\0'))-{''}
 others=set(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))-{''}
 names=sorted(n for n in tracked|others if source(n))
 head=git('rev-parse','HEAD').decode().strip()
 committed=sorted(n for n in git('ls-tree','-rz','--name-only',head).decode().split('\0') if n and source(n))
 def digest(items, read):
  h=hashlib.sha256()
  for name in items:
   h.update(name.encode()+b'\0'+normalize(name,read(name))+b'\0')
  return h.hexdigest()
 committed_bytes={n:git('show',head+':'+n) for n in committed}
 changed=sorted(n for n in set(names)|set(committed) if n not in names or n not in committed_bytes or normalize(n,(root/n).read_bytes())!=normalize(n,committed_bytes[n]))
 untracked=sorted(n for n in others if source(n))
 clean=not changed and not untracked
 require(not require_committed or clean, 'qualified source differs from HEAD: '+str(changed or untracked))
 return {'files':names,'sha256':digest(names,lambda n:(root/n).read_bytes()),'encoding':'sorted relative UTF-8 path + NUL + file bytes (LF-normalized except byte_exact_inputs) + NUL','byte_exact_inputs':pins,
         'git_source':{'head':head,'matches_head':clean,'untracked_source_files':untracked,'source_files_differing_from_head':changed,'head_source_sha256':digest(committed,committed_bytes.__getitem__)}}

def validation_engine():
 try:
  import jsonschema
 except ImportError as error:
  raise SystemExit('Install validation prerequisites first: python -m pip install -r tools/requirements-validation.txt') from error
 return 'jsonschema '+importlib.metadata.version('jsonschema')+' Draft202012Validator'

def repository_bytes(root=ROOT):
 def git(*args):
  return subprocess.check_output(['git','-C',str(root),'-c','safe.directory='+root.as_posix(),*args])
 entries=git('ls-files','--eol','-z').decode().split('\0')
 for entry in filter(None,entries):
  info,name=entry.split('\t',1)
  require(not info.startswith(('i/crlf','i/mixed')), 'noncanonical text in Git index: '+name)
 names=[e.split('\t',1)[1] for e in filter(None,entries)]
 scripts=[n for n in names if n.endswith(('.py','.sh'))]
 for name in scripts:
  for label,raw in [('HEAD',git('show','HEAD:'+name)),('index',git('show',':'+name)),('worktree',(root/name).read_bytes())]:
   require(b'\r' not in raw and not raw.startswith(b'\xef\xbb\xbf'), label+' script contains CR or BOM: '+name)
 pins_path=root/'tools/byte_exact_inputs.json'
 pins=json.loads(pins_path.read_text(encoding='utf-8')) if pins_path.exists() else {}
 for name,expected in pins.items():
  for label,raw in [('HEAD',git('show','HEAD:'+name)),('index',git('show',':'+name)),('worktree',(root/name).read_bytes())]:
   require(hashlib.sha256(raw).hexdigest()==expected, label+' byte-exact input differs from its pinned digest: '+name)
 return {'index_text_is_canonical':True,'scripts_checked':scripts,'byte_exact_inputs':pins}

def qualify_schemas(generated_dir, schema_dir, pins_path):
 generated={p.name:p for p in generated_dir.glob('*.json')}
 pins=json.loads(pins_path.read_text(encoding='utf-8'))
 present={p.name for p in schema_dir.glob('*.json')}
 require(not (generated.keys() & pins.keys()), 'schema appears in generated and hand-maintained sets')
 require(present == generated.keys() | pins.keys(), 'schema inventory differs from generated plus pinned schemas')
 from jsonschema import Draft202012Validator
 for name in sorted(present):
  raw=(schema_dir/name).read_bytes().replace(b'\r\n',b'\n')
  Draft202012Validator.check_schema(json.loads(raw))
  if name in generated:
   require(raw==generated[name].read_bytes().replace(b'\r\n',b'\n'), 'generated schema drift: '+name)
  else:
   require(hashlib.sha256(raw).hexdigest()==pins[name], 'hand-maintained schema drift: '+name)
 return {'generated':sorted(generated),'hand_maintained':pins,'total':len(present)}

def run(command):
 subprocess.run([str(x) for x in command],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--backfill-binary',type=Path);p.add_argument('--metadata-root',type=Path);p.add_argument('--registration-binary',type=Path);p.add_argument('--require-committed-source',action='store_true',help='reject source bytes or inventory differing from HEAD');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 result={'scope':'offline software qualification, no historical backfill or scientific run','schema_engine':validation_engine(),'source_digest':source_digest(require_committed=a.require_committed_source),'repository_bytes':repository_bytes()}
 with tempfile.TemporaryDirectory(prefix='xc-schema-') as d:
  run([sys.executable,ROOT/'tools/generate_research_schemas.py','--output',d])
  result['schema_coverage']=qualify_schemas(Path(d),ROOT/'docs/schemas',ROOT/'tools/handmaintained_schemas.json')
  result['schema_regeneration']=len(result['schema_coverage']['generated'])
 # Follow internal Markdown links from both entry points.
 seen=set();todo=[ROOT/'README.md',ROOT/'docs/README.md']
 while todo:
  f=todo.pop().resolve()
  if f in seen:continue
  seen.add(f)
  for target in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8-sig')):
   if '://' in target or target.startswith('#'):continue
   target=target.split('#')[0].strip('<>')
   if not target:continue
   child=(f.parent/target).resolve()
   if child.suffix=='.md' and child.is_file() and child.is_relative_to(ROOT):todo.append(child)
 missing=[p.name for p in (ROOT/'docs').glob('*.md') if p.resolve() not in seen]
 require(not missing, ('unreachable documentation',missing))
 result['documentation_reachable']=len(list((ROOT/'docs').glob('*.md')))
 for script in ['test_ccm_artifact_impact.py','test_research_tools.py','test_release_assets.py']:
  run([sys.executable,ROOT/'tools'/script])
 result['inventory_and_research_tools']='passed'
 from jsonschema import Draft202012Validator
 if a.backfill_binary:
  packets=out/'packets';run([sys.executable,ROOT/'tools/test_research_backfill.py',a.backfill_binary.resolve(),'--packets',packets])
  tests=[];negative=0
  for f in sorted(packets.glob('*.json')):
   if f.name=='acceptance.json':continue
   packet=json.loads(f.read_text());report=packet['report'];kind=report.get('kind') or packet['manifest']['key']['kind'];schema=json.loads((ROOT/'docs/schemas'/(kind.replace('_','-')+'-v1.schema.json')).read_text());v=Draft202012Validator(schema);v.validate(report);tests.append(kind)
   require(not (report.get('data',{}).get('reason') or '').startswith(';'), 'leading separator in report reason: '+kind)
   if 'diagnostic' in report.get('data',{}):
    bad=json.loads(json.dumps(report));bad['request']['semantics']='unknown';require(not v.is_valid(bad), 'schema accepted negative mutation: '+kind);negative+=1
    bad=json.loads(json.dumps(report));bad['request']['unexpected']=1;require(not v.is_valid(bad), 'schema accepted negative mutation: '+kind);negative+=1
    if report['data']['diagnostic'] in ('band_reconstruction','transform_enclosure'):
     require(report['request'].get('arb_available') is True, 'Arb producer omitted capability identity')
     for invalid in ['true',1,None,{}]:
      bad=json.loads(json.dumps(report));bad['request']['arb_available']=invalid;require(not v.is_valid(bad), 'schema accepted non-Boolean backend capability');negative+=1
     legacy=json.loads(json.dumps(report));legacy['request'].pop('arb_available');require(v.is_valid(legacy), 'schema rejected historical capability-unspecified packet')
     no_arb=json.loads(json.dumps(report));no_arb['request']['arb_available']=False;require(v.is_valid(no_arb), 'schema rejected Boolean false capability')
    if report['data']['diagnostic']=='energy_allowance':
     for name in ['conditional_energy_allowance','conditional_vector_allowance','trial_energy_magnitude']:
      bad=json.loads(json.dumps(report));bad['data']['values'].pop(name,None);require(not v.is_valid(bad), 'schema accepted missing allowance interpretation: '+name);negative+=1
     bad=json.loads(json.dumps(report));bad['data']['reason']=None;require(not v.is_valid(bad), 'schema accepted missing allowance reason');negative+=1
     bad=json.loads(json.dumps(report));bad['request']['allowance_interpretation']='unknown';require(not v.is_valid(bad), 'schema accepted unknown allowance interpretation');negative+=1
    # Every required present core field must actually be enforced.
    from generate_research_schemas import CORE
    keys=CORE.get(report['data']['diagnostic'],[])
    if keys and report['data']['outcome'] in ['point_measurement','certified_finite_enclosure','conditional_bound_expression']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(keys[0],None);require(not v.is_valid(bad), 'schema accepted negative mutation: '+kind);negative+=1
  require(len(set(tests))==30, 'expected thirty distinct producer payloads');result['payload_schema_kinds']=tests;result['negative_schema_checks']=negative
  run([sys.executable,ROOT/'tools/benchmark_band_recovery.py',a.backfill_binary.resolve(),'--output',out/'band-recovery'])
  result['band_recovery']=json.loads((out/'band-recovery/results.json').read_text())
 if a.metadata_root:
  require(a.registration_binary, '--metadata-root needs --registration-binary')
  from generate_research_schemas import KINDS,filename
  pairs=[]
  for lane in ['private','public']:
   registry=a.metadata_root/f'xcelerator-cache-{lane}-registry'
   for family in ['ccm-evidence','ccm-distance','prolate']:
    shard=a.metadata_root/f'xcelerator-cache-{lane}-{family}-0001';run([a.registration_binary.resolve(),registry/'families'/f'{family}.json',shard/'cache-repository.json']);pairs.append([lane,family])
   for kind,(family,_,_) in KINDS.items():
    name=filename(kind);expected=(ROOT/'docs/schemas'/name).read_bytes().replace(b'\r\n',b'\n')
    for parent in [registry,a.metadata_root/f'xcelerator-cache-{lane}-{family}-0001']:
     require((parent/'schemas'/name).read_bytes().replace(b'\r\n',b'\n')==expected, ('schema mirror mismatch',parent,name))
  for lane in ['private','public']:
   for name,family in [('ccm-state-geometry-analysis-v1.schema.json','ccm-evidence'),('ccm-reference-preparation-v1.schema.json','prolate')]:
    expected=(ROOT/'docs/schemas'/name).read_bytes().replace(b'\r\n',b'\n')
    for parent in [a.metadata_root/f'xcelerator-cache-{lane}-registry',a.metadata_root/f'xcelerator-cache-{lane}-{family}-0001']:
     require((parent/'schemas'/name).read_bytes().replace(b'\r\n',b'\n')==expected, ('schema mirror mismatch',parent,name))
  result['metadata_pairs']=pairs
 else:result['metadata_pairs']='not requested; supply --metadata-root for mirror qualification'
 (out/'assets.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PASS release assets and source digest',result['source_digest']['sha256'])
if __name__=='__main__':main()
