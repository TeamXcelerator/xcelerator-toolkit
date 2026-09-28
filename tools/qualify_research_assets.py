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
  return name in pins or (Path(name).parent.name == '.cargo' and Path(name).name in {'config','config.toml'}) or Path(name).name in {'rust-toolchain','rust-toolchain.toml'} or name in {'.gitattributes','tools/byte_exact_inputs.json','tools/handmaintained_schemas.json','tools/requirements-validation.txt'} or name.startswith('tools/') and name.endswith('.py') or Path(name).suffix in {'.rs','.c','.h'} or Path(name).name in {'Cargo.toml','Cargo.lock','compact_field_policy.txt'}
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
  require('attr/-text' in info or not info.startswith(('i/crlf','i/mixed')), 'noncanonical text in Git index: '+name)
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
   # Check the current revision contract on actual producer outputs. Shape
   # validation does not certify their numerical values or archived revisions.
   bad=json.loads(json.dumps(report));bad['semantics']='unknown-future-revision';require(not v.is_valid(bad), 'schema accepted unknown report semantics: '+kind);negative+=1
   for name in ['source_unit_arithmetic','compactness_arithmetic','maximum_additional_guard_bits','atom_arithmetic','atom_coordinate_serialization','maximum_atom_guard_bits','maximum_atom_exponent_span_bits','cluster_arithmetic','maximum_cluster_guard_bits','cluster_precision_policy','weighted_profile_arithmetic','maximum_weighted_profile_guard_bits','weighted_profile_output','projection_arithmetic','projection_output','l2_normalization_arithmetic','signed_channel_arithmetic','energy_arithmetic','maximum_energy_guard_bits','energy_output','directional_arithmetic','maximum_directional_guard_bits','directional_output','root_point_precision','allowance_arithmetic','maximum_allowance_guard_bits','allowance_output','transform_arithmetic','maximum_transform_guard_bits','curvature_output','resolution_arithmetic','resolution_output','observation_arithmetic','observation_output','comparison_arithmetic','comparison_output','maximum_comparison_guard_bits','finite_transfer_arithmetic','finite_transfer_output','maximum_finite_transfer_guard_bits','consistency_arithmetic','consistency_output','maximum_consistency_guard_bits','cluster_basis_arithmetic','cluster_operator_arithmetic','cluster_operator_output','maximum_cluster_operator_guard_bits','complex_arithmetic','complex_point_construction','complex_output','maximum_complex_guard_bits','complex_root_point_precision','tail_model_arithmetic','maximum_tail_form_exact_bits','transport_arithmetic','transport_output','maximum_transport_guard_bits','enclosure_point_precision','enclosure_decimal_output','signed_band_arithmetic','maximum_signed_band_exact_bits','signed_band_inverse_arithmetic','maximum_signed_band_inverse_guard_bits','polynomial_band_arithmetic','polynomial_root_output','maximum_polynomial_band_exact_bits','polynomial_root_window','acceptance_rounding','relative_change_arithmetic']:
    if name not in report.get('request',{}):continue
    for remove in [True,False]:
     bad=json.loads(json.dumps(report))
     if remove:bad['request'].pop(name)
     else:bad['request'][name]='unknown-arithmetic'
     require(not v.is_valid(bad), 'schema accepted missing/unknown arithmetic contract: '+kind+' '+name);negative+=1
   if kind=='ccm_tail_operator_analysis':
    name='model_linear_algebra_arithmetic'
    require(report['request'].get(name)=='exact_stored_dot_product_stages_and_tail_bound_v0.15.2-v2', 'current tail producer lacks its arithmetic stamp')
    bad=json.loads(json.dumps(report));bad['request'][name]='unknown-arithmetic'
    require(not v.is_valid(bad), 'schema accepted unknown tail model arithmetic');negative+=1
    legacy=json.loads(json.dumps(report));legacy['request'].pop(name)
    require(v.is_valid(legacy), 'schema no longer reads legacy tail reports without the new arithmetic stamp')
   if kind=='ccm_complex_transform_analysis':
    measured=next((k for k,row in enumerate(report['data']['rows']) if row['outcome']=='point_measurement'),None)
    require(measured is not None,'complex producer yielded no measured row')
    for name in ['arithmetic_precision_bits','z_re_lower','value_re_lower','derivative_re_upper','normalized_im_lower','log_derivative_re_upper','normalization_denominator_resolved']:
     bad=json.loads(json.dumps(report));bad['data']['rows'][measured]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing complex field: '+name);negative+=1
    for name in ['arithmetic_precision_bits','normalization_anchor_lower','normalization_anchor_upper']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing complex anchor field: '+name);negative+=1
   if kind=='ccm_operator_cluster_analysis' and report['data']['rows']:
    for name in ['arithmetic_precision_bits','coupling_gram_lower','coupling_gram_upper','compressed_operator_lower']:
     bad=json.loads(json.dumps(report));bad['data']['rows'][0]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing cluster field: '+name);negative+=1
    for name in ['column_selection_threshold','source_leakage_squared_lower']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing cluster report field: '+name);negative+=1
   if kind in ('ccm_finite_section_transfer','ccm_consistency_analysis'):
    first='low_residual_squared' if kind=='ccm_finite_section_transfer' else 'action_difference_norm'
    measured=next((k for k,row in enumerate(report['data']['rows']) if first in row['values']),None)
    if measured is not None:
     for name in ['arithmetic_precision_bits',first+'_lower',first+'_upper']:
      bad=json.loads(json.dumps(report));bad['data']['rows'][measured]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing finite transfer/consistency field: '+name);negative+=1
   if kind=='ccm_observable_budget_analysis':
    for name in ['arithmetic_precision_bits','transform_origin_lower','origin_absolute_terms_upper']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing observation field: '+name);negative+=1
   if kind=='ccm_configuration_comparison':
    measured=next((k for k,row in enumerate(report['data']['rows']) if 'signed_energy_difference' in row['values']),None)
    if measured is not None:
     for name in ['arithmetic_precision_bits','signed_energy_difference_lower','comparison_C_upper']:
      bad=json.loads(json.dumps(report));bad['data']['rows'][measured]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing comparison field: '+name);negative+=1
   if kind=='ccm_resolution_budget_analysis':
    measured=next((k for k,row in enumerate(report['data']['rows']) if 'transform' in row['values']),None)
    require(measured is not None,'resolution producer yielded no measured row')
    for name in ['arithmetic_precision_bits','transform_lower','derivative_upper']:
     bad=json.loads(json.dumps(report));bad['data']['rows'][measured]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing resolution field: '+name);negative+=1
    for name in ['finite_curvature_expression_upper','relative_tolerance_lower']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing resolution field: '+name);negative+=1
   if kind=='ccm_energy_allowance_analysis':
    for name in ['arithmetic_precision_bits','denominator_lower','conditional_energy_allowance_upper','scale_comparison_resolved']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing allowance field: '+name);negative+=1
   if kind=='ccm_directional_response_analysis':
    measured=next((k for k,row in enumerate(report['data']['rows']) if 'directional_energy' in row['values']),None)
    require(measured is not None,'directional producer yielded no measured row')
    for name in ['arithmetic_precision_bits','directional_energy_lower','rational_root_condition_upper','root_condition_tolerance']:
     bad=json.loads(json.dumps(report));bad['data']['rows'][measured]['values'].pop(name);require(not v.is_valid(bad),'schema accepted missing directional field: '+name);negative+=1
   if kind=='ccm_arithmetic_energy_analysis':
    for name in ['arithmetic_precision_bits','total_tau_energy_lower','energy_closure_defect_upper']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad), 'schema accepted missing energy enclosure field: '+name);negative+=1
    bad=json.loads(json.dumps(report));bad['data']['rows'][0]['values'].pop('energy_lower');require(not v.is_valid(bad), 'schema accepted missing component energy enclosure');negative+=1
   if kind=='ccm_signed_transform_analysis':
    for name in ['arithmetic_precision_bits','normalization_precision_bits']:
     bad=json.loads(json.dumps(report));bad['data']['values'].pop(name);require(not v.is_valid(bad), 'schema accepted missing signed-channel precision: '+name);negative+=1
   if kind=='ccm_reference_projection_analysis':
    for name in ['pivot_metric','arithmetic_precision_bits','arithmetic_enclosures']:
     bad=json.loads(json.dumps(report));bad['data'].pop(name);require(not v.is_valid(bad), 'schema accepted missing projection arithmetic field: '+name);negative+=1
    for name in ['source_center','reference_center','signed_unit_overlap','difference_norm_squared','fit_residual_norm_squared','minimum_pivot']:
     if name not in report['data']['arithmetic_enclosures']:continue
     bad=json.loads(json.dumps(report));bad['data']['arithmetic_enclosures'].pop(name);require(not v.is_valid(bad), 'schema accepted missing projection enclosure: '+name);negative+=1
    bad=json.loads(json.dumps(report));bad['data']['arithmetic_enclosures']['source_center']=['0'];require(not v.is_valid(bad), 'schema accepted malformed projection enclosure');negative+=1
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
