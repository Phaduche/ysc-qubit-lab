import sys,json,tempfile,unittest,sqlite3,csv
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ysc
ROOT=Path(__file__).resolve().parents[1]

class LabTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.db=self.root/'test.sqlite'
  self.cfg=ysc.load(ROOT/'configs/baseline.json');self.data=ysc.load(ROOT/'data/supplied_smoke.json')
 def tearDown(self):self.tmp.cleanup()
 def initialize(self):
  (self.root/'c.json').write_text(json.dumps(self.cfg));(self.root/'d.json').write_text(json.dumps(self.data))
  return ysc.init_run(self.root/'c.json',self.root/'d.json',self.db)
 def test_numeric_and_name_boundaries(self):
  self.assertFalse(ysc.spans('112%와 RAGGED','12'));self.assertFalse(ysc.spans('RAGGED','RAG'))
  self.assertTrue(ysc.spans('비율 12%와 RAG를 사용','12%'));self.assertTrue(ysc.spans('RAG를','RAG'))
 def test_empty_gold_is_na(self):
  self.assertIsNone(ysc.evaluate('anything',{})['fact_retention']);self.assertIsNone(ysc.pair_jaccard([],[]))
 def test_fact_requires_all_terms(self):
  item={'id':'x','alternatives':[['비율','12%'],['rate','12%']]}
  self.assertFalse(ysc.item_evidence('비율 112%',item)['found']);self.assertTrue(ysc.item_evidence('rate 12%',item)['found'])
 def test_annotation_must_exist_in_source(self):
  self.data['documents'][0]['numbers'].append({'id':'bad','aliases':['9999kg']})
  with self.assertRaises(ValueError):ysc.validate_data(self.data)
 def test_plan_and_dependencies(self):
  self.assertEqual(self.initialize(),134);f,j=ysc.plan(self.cfg,self.data)
  self.assertEqual(sum(x['experiment']=='A' for x in j),6)
  self.assertTrue(all(not x['dependencies'] for x in j if x['experiment']=='B'))
  self.assertTrue(all(len(x['dependencies'])==1 for x in j if x['experiment']=='C' and x['stage']>1))
  self.assertTrue(all(len(x['dependencies'])==2 for x in j if x['experiment']=='RAG' and x['store']=='summary'))
  self.cfg['repeats']=10;self.assertEqual(len(ysc.plan(self.cfg,self.data)[1]),646)
 def test_changed_annotation_or_config_cannot_resume(self):
  self.initialize();self.cfg['seed']+=1
  with self.assertRaises(ValueError):self.initialize()
 def test_chunk_overlap_and_validation(self):
  chunks=ysc.chunk_text('a'*85,40,10);self.assertEqual([c['start'] for c in chunks],[0,30,60])
  with self.assertRaises(ValueError):ysc.chunk_text('abc',20,20)
 def test_real_baseline_resume_no_duplicate_jobs(self):
  self.initialize();ysc.execute(self.db,2);ysc.execute(self.db,1)
  db=ysc.connect(self.db)
  self.assertEqual(db.execute("select count(*) from jobs where status='done'").fetchone()[0],3)
  self.assertEqual(db.execute('select count(*) from attempts').fetchone()[0],3);db.close()
 def test_fail_then_resume_preserves_attempt(self):
  self.initialize()
  with patch.object(ysc.Backend,'generate',side_effect=RuntimeError('test fixture injected failure')):ysc.execute(self.db,1)
  ysc.execute(self.db,1);db=ysc.connect(self.db)
  states=[r[0] for r in db.execute('select status from attempts order by id')]
  self.assertEqual(states,['failed','done']);db.close()
 def test_runner_exclusion(self):
  self.initialize()
  with ysc.worker_lock(self.db):
   with self.assertRaises(RuntimeError):
    with ysc.worker_lock(self.db):pass
 def test_interrupted_attempt_recorded(self):
  self.initialize();db=ysc.connect(self.db);jid=db.execute('select id from jobs order by rowid').fetchone()[0]
  db.execute("update jobs set status='running',attempts=1 where id=?",(jid,))
  db.execute("insert into attempts(job_id,started,status) values (?,?,'running')",(jid,ysc.now()));db.commit();db.close()
  ysc.execute(self.db,1);db=ysc.connect(self.db)
  self.assertEqual([r[0] for r in db.execute('select status from attempts order by id')],['interrupted','done']);db.close()
 def test_retrieval_searches_whole_corpus(self):
  cfg=self.cfg['retrieval'];cfg['top_k']=1
  hits,_=ysc.retrieve(self.db,cfg,[{'id':'wrong','text':'우주의 블랙홀은 빛을 삼킨다.'},{'id':'right','text':'구기성 박사는 Qubit 팀에 자문했다.'}],'구기성 박사는 어느 팀에 자문했는가?')
  self.assertEqual(hits[0]['source_id'],'right')
 def test_api_usage_missing_is_not_fabricated(self):
  b=ysc.Backend({'kind':'openai_compatible','base_url':'http://localhost','model':'test-fixture','supports_seed':False})
  with patch.object(b,'request',return_value={'choices':[{'message':{'content':'simulated fixture'},'finish_reason':'stop'}]}):
   r=b.generate('fixture',1)
  self.assertIsNone(r['total_tokens']);self.assertFalse(r['seed_sent'])
 def test_export_only_successes_and_na_consistency(self):
  self.initialize();ysc.execute(self.db,1);ysc.export_csv(self.db,self.root/'csv')
  with (self.root/'csv/summary.csv').open(encoding='utf-8-sig') as h:rows=list(csv.DictReader(h))
  self.assertEqual(sum(int(r['n_successful_jobs']) for r in rows),1)
  self.assertEqual(rows[0]['consistency_fact_jaccard_mean'],'')
 def test_truncation_never_enters_summary(self):
  self.initialize();original=ysc.Backend.generate
  def truncated(obj,*a,**kw):
   r=original(obj,*a,**kw);r['finish_reason']='length';r['data_class']='simulation_fixture';return r
  with patch.object(ysc.Backend,'generate',new=truncated):ysc.execute(self.db,1)
  self.assertEqual(ysc.status_report(self.db)['jobs']['truncated'],1)
  self.assertEqual(ysc.status_report(self.db)['completed_by_class'],{})

if __name__=='__main__':unittest.main(verbosity=2)
