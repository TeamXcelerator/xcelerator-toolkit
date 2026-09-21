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
   committed=subprocess.check_output(['git','-C',str(root),'-c','safe.directory='+root.as_posix(),'show','HEAD:'+name])
   for raw in [committed,file.read_bytes()]:
    if committed.startswith(b'#!') or file.name in {'prepare_atom_chunks.py','publication_summary.py'}:
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
if __name__=='__main__':unittest.main()
