#!/usr/bin/env python3
"""Offline regression tests for compact research navigation and operational summaries."""
import hashlib,json,sqlite3,tempfile,unittest
from contextlib import closing
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
class ExhaustiveTools(unittest.TestCase):
 def test_exhaustive_packet_failure_rolls_back_prior_scalar_rows(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);packet=root/'partial.json';packet.write_text(json.dumps(dict(values={'good':'1'},rows=[dict(values=['bad'])])))
   database=root/'index.sqlite';self.assertEqual(research_query.build([packet],database),0)
   with closing(sqlite3.connect(database)) as c:
    self.assertEqual(c.execute('select count(*) from measurements').fetchone()[0],0)
    self.assertEqual(c.execute('select status from inventory').fetchone()[0],'unassessed')
 def test_exhaustive_index_dimensions_do_not_use_sqlite_lossy_coercion(self):
  for value in ['9007199254740993.0',1.5,True,1<<100]:
   with self.subTest(value=value),tempfile.TemporaryDirectory() as t:
    root=Path(t);packet=root/'report.json';packet.write_text(json.dumps(dict(n_modes=value,values={'energy':'1'})))
    database=root/'index.sqlite';self.assertEqual(research_query.build([packet],database),0)
    with closing(sqlite3.connect(database)) as c:
     self.assertEqual(c.execute('select count(*) from measurements').fetchone()[0],0)
     self.assertEqual(c.execute('select status from inventory').fetchone()[0],'unassessed')
 def test_exhaustive_decimal_navigation_uses_ascii_grammar(self):
  records=list(research_query.rows(Path('report.json'),dict(values={'ascii':'1.25e-10','unicode':'١.٢٥'})))
  self.assertEqual([row['observable'] for row in records],['ascii'])
 def test_exhaustive_atom_kind_rejected_before_output_creation(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);source=root/'atoms.jsonl';source.write_text(json.dumps(dict(ordinal=1,coordinate='1',weight='2',family='zero',partition='all'))+'\n')
   output=root/'chunks'
   with self.assertRaises(ValueError):prepare_atom_chunks.prepare(source,output,'unknown')
   self.assertFalse(output.exists())
 def test_exhaustive_malformed_publication_event_does_not_mutate_statistics(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'attempt.jsonl';p.write_text(json.dumps(dict(schema_version=1,phase='bad',elapsed_seconds=-1,details={'bytes':10}))+'\n')
   result=publication_summary.summarize([p])['attempts'][0]
   self.assertEqual(result['phases'],{});self.assertEqual(len(result['malformed']),1)
 def test_exhaustive_nonobject_publication_event_is_reported(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'attempt.jsonl';p.write_text('[]\n')
   result=publication_summary.summarize([p])['attempts'][0]
   self.assertEqual(result['phases'],{});self.assertEqual(len(result['malformed']),1)
 def test_exhaustive_publication_aggregate_overflow_and_bool_bytes(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'attempt.jsonl';p.write_text('\n'.join(json.dumps(dict(schema_version=1,phase='phase',elapsed_seconds=1e308,details={'bytes':True})) for _ in range(2))+'\n')
   result=publication_summary.summarize([p])['attempts'][0]
   self.assertEqual(result['phases']['phase']['seconds'],1e308)
   self.assertEqual(result['phases']['phase']['calls'],1)
   self.assertEqual(result['phases']['phase']['byte_counters'],{})
   self.assertEqual(len(result['malformed']),1)

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

class OperationalQualificationGuards(unittest.TestCase):
    def test_band_recovery_rejects_optimized_python_before_writes(self):
        self.optimized_guard("benchmark_band_recovery.py","--output")
    def test_backfill_acceptance_rejects_optimized_python_before_writes(self):
        self.optimized_guard("test_research_backfill.py","--packets")
    def optimized_guard(self,tool,option):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/"output";script=Path(__file__).resolve().parent/tool
            result=subprocess.run([sys.executable,"-O",str(script),str(root/"absent-binary"),option,str(output)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("requires Python without -O",result.stderr)
            self.assertFalse(output.exists())
    def test_band_arguments_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/"output";script=Path(__file__).resolve().parent/"benchmark_band_recovery.py"
            result=subprocess.run([sys.executable,str(script),sys.executable,"--output",str(output),"--degree","5"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(output.exists())


class OperationalEnvironmentGuards(unittest.TestCase):
    def test_synthetic_tools_clear_ambient_research_and_publication_configuration(self):
        import benchmark_band_recovery, test_research_backfill
        settings={"XC_RESEARCH_BASIS_BYTES":"1","XC_RESEARCH_SUMMARY_DIR":"outside",
                  "XC_RESEARCH_CHECKPOINT_DIR":"outside", "XC_CACHE_ROOT":"outside",
                  "XC_CACHE_REMOTE":"remote", "XC_PUBLISH_EXECUTE":"true",
                  "PATH":os.environ.get("PATH","")}
        with patch.dict(os.environ,settings,clear=True):
            for module in [benchmark_band_recovery,test_research_backfill]:
                env=module.local_environment()
                self.assertFalse(any(k.startswith("XC_RESEARCH_") for k in env))
                self.assertNotIn("XC_CACHE_ROOT",env)
                self.assertEqual(env["XC_CACHE_REMOTE"],"none")
                self.assertEqual(env["XC_PUBLISH_TARGET"],"none")
                self.assertEqual(env["XC_PUBLISH_EXECUTE"],"false")
                self.assertEqual(env["PATH"],settings["PATH"])




class AtomLineBoundary(unittest.TestCase):
 def test_limit_applies_to_normalized_line(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp); source=root/'input.jsonl'
   row=json.dumps(dict(coordinate='1',signed_weight='1',family=''))
   for size,accepted in [(1<<20,False),((1<<20)-1,True)]:
    source.write_bytes((row[:-2]+'x'*(size-len(row))+row[-2:]).encode('ascii'))
    output=root/str(size)
    if accepted:
     result=prepare_atom_chunks.prepare(source,output,'band')
     self.assertEqual(result['maximum_input_bytes'],1<<20)
    else:
     with self.assertRaisesRegex(ValueError,'normalized atom line'):prepare_atom_chunks.prepare(source,output,'band')

if __name__=='__main__':unittest.main()
