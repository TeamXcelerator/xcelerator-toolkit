#!/usr/bin/env python3
"""Offline regression tests for compact research navigation and operational summaries."""
import hashlib,json,sqlite3,tempfile,unittest
from pathlib import Path
import research_query,publication_summary,prepare_atom_chunks

class Tools(unittest.TestCase):
 def test_shared_field_policy(self):
  for kind,prefix in research_query.FIELD_POLICY:
   self.assertIn(kind,('prefix','indexed'));self.assertTrue(prefix)
   self.assertFalse(research_query.scalar_field(prefix+'9'*100))
  for name in ['tail_','tail_+1','tail_١','tail_energy_lift']:
   self.assertTrue(research_query.scalar_field(name),name)

 def test_malformed_policy_is_rejected_before_indexing(self):
  for text in ['', '\n', 'prefix:ok\n\n', 'unknown:x', 'prefix:', 'prefix:x y', 'prefix:x:y', 'prefix:x\nprefix:x']:
   with self.assertRaisesRegex(ValueError, 'invalid compact field policy'):
    research_query.parse_field_policy(text)
  self.assertEqual(len(research_query.parse_field_policy('prefix:a_\r\nindexed:b_\r\n')),2)
 def test_exact_decimal_index_and_missing_files_are_explicit(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);packet=root/'report.json';value='-1.1234567890123456789012345e-2000'
   packet.write_text(json.dumps(dict(artifact_digest='a'*64,report=dict(kind='test',data=dict(lambda_squared='250',n_modes=750,source_precision_bits=6708,values={'energy':value,'model_vector_coefficient_0':'1'},rows=[dict(ordinal=19,label='declared_reference',outcome='unresolved_denominator',values={'spacing':'0.5'},notes=['caller join'])])))))
   bad=root/'bad.json';bad.write_text('{bad')
   db=root/'lookup.sqlite';self.assertEqual(research_query.build([packet,bad,root/'absent.json'],db),2)
   with sqlite3.connect(db) as c:
    self.assertEqual(c.execute("select value from measurements where observable='energy'").fetchone()[0],value)
    self.assertEqual(c.execute("select count(*) from inventory where status='unassessed'").fetchone()[0],2)
   c.close()
   with self.assertRaises(ValueError):research_query.build([packet],db)
 def test_publication_keeps_failed_and_interrupted_attempts(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);a=root/'attempt-a.jsonl';b=root/'attempt-b.jsonl'
   a.write_text('\n'.join(json.dumps(e) for e in [dict(schema_version=1,phase='git_push',elapsed_seconds=2,details={'success':False}),dict(schema_version=1,phase='attempt_finished',elapsed_seconds=5,details={'success':False})])+'\n')
   b.write_text(json.dumps(dict(schema_version=1,phase='destination_verification',elapsed_seconds=1,details={'bytes':10,'reused_verified_bytes':10}))+'\n{truncated')
   r=publication_summary.summarize([a,b]);self.assertEqual(r['attempts'][0]['completion'],'failed');self.assertEqual(r['attempts'][1]['completion'],'incomplete_or_unknown');self.assertEqual(len(r['attempts'][1]['malformed']),1)
   self.assertEqual(r['attempts'][1]['phases']['destination_verification']['byte_counters']['reused_verified_bytes'],10)
 def test_chunk_builder_preserves_scalar_strings_and_binds_every_byte(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'input.jsonl';row=dict(coordinate='1.0000000000000000000000001',signed_weight='1e-1000',family='zero');source.write_text((json.dumps(row)+'\n')*3)
   output=root/'chunks';p=prepare_atom_chunks.prepare(source,output,'band',2)
   self.assertEqual([c['rows'] for c in p['band_chunks']],[2,1])
   for c in p['band_chunks']:
    raw=(output/c['relative_path']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),c['sha256']);self.assertEqual(json.loads(raw.splitlines()[0]),row)
   with self.assertRaises(FileExistsError):prepare_atom_chunks.prepare(source,output,'band')
if __name__=='__main__':unittest.main()
