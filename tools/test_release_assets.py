#!/usr/bin/env python3
"""Release guard regressions: dirty sources, schema inventory and optimized Python."""
import hashlib, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
import qualify_research_assets as assets

class ReleaseAssets(unittest.TestCase):
 def test_source_record_distinguishes_committed_modified_and_untracked(self):
  with tempfile.TemporaryDirectory(prefix='xc-source-') as d:
   root=Path(d)
   def git(*args):
    return subprocess.run(['git','-C',d,'-c','user.name=Test','-c','user.email=test@example.invalid',*args],check=True,capture_output=True)
   git('init');(root/'lib.rs').write_text('// base\n');git('add','.');git('commit','-m','fixture')
   baseline=assets.source_digest(root,require_committed=True)
   self.assertTrue(baseline['git_source']['matches_head'])
   (root/'lib.rs').write_text('// changed\n');(root/'new.rs').write_text('// extra\n')
   dirty=assets.source_digest(root)
   self.assertEqual(dirty['git_source']['untracked_source_files'],['new.rs'])
   self.assertEqual(dirty['git_source']['source_files_differing_from_head'],['lib.rs','new.rs'])
   self.assertEqual(dirty['git_source']['head_source_sha256'],baseline['sha256'])
   self.assertNotEqual(dirty['sha256'],baseline['sha256'])
   with self.assertRaises(ValueError):assets.source_digest(root,require_committed=True)
   git('add','.');git('commit','-m','updated fixture')
   self.assertEqual(assets.source_digest(root,require_committed=True)['sha256'],dirty['sha256'])
 def test_cargo_configuration_is_bound_to_qualified_source(self):
  with tempfile.TemporaryDirectory(prefix='xc-config-source-') as d:
   root=Path(d)
   def git(*args):
    return subprocess.run(['git','-C',d,'-c','user.name=Test','-c','user.email=test@example.invalid',*args],check=True,capture_output=True)
   git('init');(root/'lib.rs').write_text('// base\n');git('add','.');git('commit','-m','fixture')
   baseline=assets.source_digest(root,require_committed=True)
   for name in ['.cargo/config.toml','.cargo/config','tests/consumer/.cargo/config.toml','rust-toolchain.toml','rust-toolchain']:
    with self.subTest(name=name):
     path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# build input\n')
     dirty=assets.source_digest(root)
     self.assertIn(name,dirty['files'])
     self.assertIn(name,dirty['git_source']['untracked_source_files'])
     self.assertNotEqual(dirty['sha256'],baseline['sha256'])
     with self.assertRaises(ValueError):assets.source_digest(root,require_committed=True)
     git('add',name);git('commit','-m','config fixture')
     committed=assets.source_digest(root,require_committed=True)
     self.assertEqual(committed['sha256'],dirty['sha256'])
     path.write_text('# modified build input\n')
     with self.assertRaises(ValueError):assets.source_digest(root,require_committed=True)
     path.unlink()
     with self.assertRaises((ValueError,FileNotFoundError)):assets.source_digest(root,require_committed=True)
     git('add',name);git('commit','-m','remove config fixture')
 def test_all_schemas_have_exactly_one_guard(self):
  with tempfile.TemporaryDirectory(prefix='xc-schema-guard-') as d:
   root=Path(d);g=root/'generated';s=root/'schemas';g.mkdir();s.mkdir();pins=root/'pins.json'
   raw=b'{"type":"object"}\n'
   (g/'generated.json').write_bytes(raw);(s/'generated.json').write_bytes(raw);(s/'manual.json').write_bytes(raw)
   pins.write_text(json.dumps({'manual.json':hashlib.sha256(raw).hexdigest()}))
   self.assertEqual(assets.qualify_schemas(g,s,pins)['total'],2)
   for name in ['generated.json','manual.json']:
    (s/name).write_text('{"type":"string"}')
    with self.assertRaises(ValueError):assets.qualify_schemas(g,s,pins)
    (s/name).write_bytes(raw)
   (s/'unregistered.json').write_bytes(raw)
   with self.assertRaises(ValueError):assets.qualify_schemas(g,s,pins)
 def test_checks_survive_python_optimization(self):
  result=subprocess.run([sys.executable,'-O','-c',"from qualify_research_assets import require; require(False, 'negative gate')"],cwd=Path(__file__).resolve().parent,capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0);self.assertIn('requires Python without -O',result.stderr)
 def test_executable_tools_have_a_real_shebang(self):
  root=Path(__file__).resolve().parents[1]
  for file in sorted((root/'tools').glob('*.py')):
   name=file.relative_to(root).as_posix()
   revision=subprocess.run(['git','-C',str(root),'-c','safe.directory='+root.as_posix(),'show','HEAD:'+name],capture_output=True)
   # New local tools still require byte checks before they are committed.
   # Only a path absent from HEAD may omit the historical-byte comparison.
   if revision.returncode:
    tracked=subprocess.check_output(['git','-C',str(root),'ls-tree','--name-only','HEAD','--',name])
    self.assertFalse(tracked,revision.stderr.decode(errors='replace'))
   versions=([revision.stdout] if revision.returncode==0 else [])+[file.read_bytes()]
   for raw in versions:
    if any(version.startswith(b'#!') for version in versions) or file.name in {'prepare_atom_chunks.py','publication_summary.py'}:
     self.assertTrue(raw.startswith(b'#!/usr/bin/env python3\n'),name)
    self.assertNotIn(b'\r',raw,name)
 def test_repository_guard_rejects_staged_crlf_and_byte_exact_mutations(self):
  with tempfile.TemporaryDirectory(prefix='xc-byte-guard-') as d:
   root=Path(d);(root/'tools').mkdir();(root/'data').mkdir()
   def git(*args,input=None):
    return subprocess.check_output(['git','-C',d,'-c','core.autocrlf=false','-c','user.name=Test','-c','user.email=test@example.invalid',*args],input=input,stderr=subprocess.PIPE)
   raw=b'["1.000"]\n';(root/'data/ref.json').write_bytes(raw)
   (root/'tools/run.py').write_bytes(b'#!/usr/bin/env python3\n')
   (root/'tools/byte_exact_inputs.json').write_text(json.dumps({'data/ref.json':hashlib.sha256(raw).hexdigest()}))
   git('init');git('add','.');git('commit','-m','fixture')
   self.assertTrue(assets.repository_bytes(root)['index_text_is_canonical'])
   (root/'data/ref.json').write_bytes(raw.replace(b'\n',b'\r\n'))
   with self.assertRaisesRegex(ValueError,'byte-exact'):assets.repository_bytes(root)
   with self.assertRaises(ValueError):assets.source_digest(root,require_committed=True)
   (root/'data/ref.json').write_bytes(raw)
   blob=git('hash-object','-w','--stdin',input=b'#!/usr/bin/env python3\r\n').decode().strip()
   git('update-index','--cacheinfo','100644,'+blob+',tools/run.py')
   with self.assertRaisesRegex(ValueError,'noncanonical text'):assets.repository_bytes(root)
 def test_byte_exact_logs_are_preserved_without_relaxing_script_guards(self):
  with tempfile.TemporaryDirectory(prefix='xc-raw-log-') as d:
   root=Path(d)
   def git(*args):
    return subprocess.check_output(['git','-C',d,'-c','core.autocrlf=false','-c','user.name=Test','-c','user.email=test@example.invalid',*args],stderr=subprocess.PIPE)
   (root/'.gitattributes').write_bytes(b'*.log binary\n*.py binary\n')
   raw=b'raw Windows evidence\r\n';(root/'audit.log').write_bytes(raw)
   git('init');git('add','.');git('commit','-m','raw log fixture')
   self.assertTrue(assets.repository_bytes(root)['index_text_is_canonical'])
   self.assertEqual(git('show','HEAD:audit.log'),raw)
   (root/'bad.py').write_bytes(b'#!/usr/bin/env python3\r\n')
   git('add','.');git('commit','-m','invalid script fixture')
   with self.assertRaisesRegex(ValueError,'script contains CR'):assets.repository_bytes(root)
class ExhaustiveSchemaContracts(unittest.TestCase):
 def test_current_producer_revision_is_accepted_and_future_revision_rejected(self):
  import re,generate_research_schemas as schemas
  from jsonschema import Draft202012Validator
  root=Path(__file__).resolve().parents[1]
  current=re.search(r'pub const SEMANTICS: &str = "([^"]+)"',(root/'crates/xc-spectral/src/ccm/retained_evidence.rs').read_text(encoding='utf-8')).group(1)
  with tempfile.TemporaryDirectory() as t:
   output=Path(t);schemas.generate(output)
   for kind in schemas.KINDS:
    value=json.loads((output/schemas.filename(kind)).read_text())
    rule=value['properties']['semantics'];validator=Draft202012Validator(rule)
    self.assertTrue(validator.is_valid(current),kind)
    self.assertFalse(validator.is_valid('ccm-retained-research-observations-v999'),kind)
 def test_tail_model_arithmetic_accepts_current_identity_and_preserves_legacy_absence(self):
  import generate_research_schemas as schemas
  from jsonschema import Draft202012Validator
  with tempfile.TemporaryDirectory() as t:
   schemas.generate(Path(t))
   schema=json.loads((Path(t)/schemas.filename('ccm_tail_operator_analysis')).read_text())
   request=schema['properties']['request']
   name='model_linear_algebra_arithmetic'
   self.assertIn(name,request['properties'])
   self.assertNotIn(name,request['required'])  # Archived revisions did not carry this field.
   validator=Draft202012Validator(request['properties'][name])
   self.assertTrue(validator.is_valid('exact_stored_dot_product_stages_and_tail_bound_v0.15.2-v2'))
   for value in ['unknown-arithmetic',None,False,1,'exact_stored_dot_product_stages_v0.15.2-v1']:
    self.assertFalse(validator.is_valid(value),repr(value))
 def test_decimal_schema_accepts_explicit_positive_sign(self):
  import generate_research_schemas as schemas
  from jsonschema import Draft202012Validator
  validator=Draft202012Validator(schemas.SC)
  for value in ['+1','+.25e-123','+1.','-0','0.001E+50']:
   self.assertTrue(validator.is_valid(value),value)
 def test_decimal_schema_rejects_trailing_newline(self):
  import generate_research_schemas as schemas
  from jsonschema import Draft202012Validator
  validator=Draft202012Validator(schemas.SC)
  for value in ['1\n','+1\n','NaN','inf','١',' 1','1 ']:
   self.assertFalse(validator.is_valid(value),repr(value))
 def test_digest_schema_rejects_trailing_newline(self):
  import generate_research_schemas as schemas
  from jsonschema import Draft202012Validator
  validator=Draft202012Validator(schemas.D)
  self.assertTrue(validator.is_valid('a'*64))
  self.assertFalse(validator.is_valid('a'*64+'\n'))




class StrictDigestSchemas(unittest.TestCase):
 def test_every_sha256_pattern_rejects_trailing_newline(self):
  import re
  root=Path(__file__).resolve().parents[1]
  checked=0
  def patterns(value):
   if isinstance(value,dict):
    if 'pattern' in value:yield value['pattern']
    for child in value.values():yield from patterns(child)
   elif isinstance(value,list):
    for child in value:yield from patterns(child)
  for path in (root/'docs/schemas').glob('*.schema.json'):
   for pattern in patterns(json.loads(path.read_text(encoding='utf-8'))):
    if '[0-9a-f]{64}' in pattern:
     checked+=1
     self.assertIsNotNone(re.search(pattern,'a'*64),path.name)
     for bad in ['a'*64+'\n','a'*63,'A'*64,'a'*65]:self.assertIsNone(re.search(pattern,bad),path.name)
  self.assertGreater(checked,7)

if __name__=='__main__':unittest.main()
