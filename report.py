"""Evidence exports beyond the core runner; never turns pending jobs into results."""
import argparse,csv,json,itertools
from pathlib import Path
from collections import defaultdict
import ysc

def create(dbpath,outpath):
 ysc.export_csv(dbpath,outpath);out=Path(outpath);db=ysc.connect(dbpath)
 frozen=json.loads(db.execute('select body from manifests').fetchone()[0]);rows=[]
 for row in db.execute('select * from jobs order by rowid'):
  rows.append(dict(json.loads(row['spec']),status=row['status'],result=json.loads(row['result']) if row['result'] else None))
 def write(name,records,fields):
  with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r in records:w.writerow({k:ysc.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
 sources=[dict(source_id=d['id'],stage='H0',source=d['source'],source_sha256=ysc.digest(d['text']),original_text=d['text'],
               facts=d.get('facts',[]),entities=d.get('entities',[]),numbers=d.get('numbers',[]),questions=d.get('questions',[])) for d in frozen['dataset']['documents']]
 write('sources_H0.csv',sources,['source_id','stage','source','source_sha256','original_text','facts','entities','numbers','questions'])
 status={r['id']:r['status'] for r in rows}
 blocked=[{'job_id':r['id'],'experiment':r['experiment'],'status':r['status'],'unmet_dependencies':[{d:status[d]} for d in r['dependencies'] if status[d]!='done']} for r in rows if r['status']=='pending']
 write('unexecuted_dependencies.csv',blocked,['job_id','experiment','status','unmet_dependencies'])
 qa=[{'job_id':r['id'],'source_id':r['source_id'],'repeat':r['repeat'],'store':r['store'],'prompt_id':r['prompt_id'],'stage':r['stage'],
      'answer':r['result']['output'],'reference_answer_groups':r['result']['answer_groups'],'retrieved_evidence':r['result']['retrieved'],
      'alias_proxy_correct':r['result']['qa_exact_alias_correct'],'citations_valid':r['result']['all_citations_valid'],
      'human_correct':'','human_grounded':'','reviewer':'','reason':''} for r in rows if r['status']=='done' and r['experiment']=='RAG']
 write('qa_review.csv',qa,['job_id','source_id','repeat','store','prompt_id','stage','answer','reference_answer_groups','retrieved_evidence','alias_proxy_correct','citations_valid','human_correct','human_grounded','reviewer','reason'])
 originals={(r['source_id'],r['question_id'],r['repeat']):r for r in rows if r['status']=='done' and r['experiment']=='RAG' and r['store']=='original'}
 paired=[]
 for r in rows:
  if r['status']!='done' or r['experiment']!='RAG' or r['store']!='summary':continue
  orig=originals.get((r['source_id'],r['question_id'],r['repeat']))
  if not orig:continue
  a,b=orig['result'],r['result']
  paired.append({'original_job_id':orig['id'],'summary_job_id':r['id'],'source_id':r['source_id'],'question_id':r['question_id'],
    'repeat':r['repeat'],'seed':r['seed'],'prompt_id':r['prompt_id'],'summary_stage':r['stage'],
    'original_qa_proxy':a['qa_exact_alias_correct'],'summary_qa_proxy':b['qa_exact_alias_correct'],
    'paired_qa_delta':b['qa_exact_alias_correct']-a['qa_exact_alias_correct'],
    'original_generation_tokens':a['total_tokens'],'summary_generation_tokens':b['total_tokens'],
    'original_fact_proxy':a['fact_retention'],'summary_fact_proxy':b['fact_retention'],
    'interpretation':'descriptive_only_small_supplied_excerpt_corpus'})
 write('paired_RAG_comparison.csv',paired,['original_job_id','summary_job_id','source_id','question_id','repeat','seed','prompt_id','summary_stage','original_qa_proxy','summary_qa_proxy','paired_qa_delta','original_generation_tokens','summary_generation_tokens','original_fact_proxy','summary_fact_proxy','interpretation'])
 costs=[]
 for st in sorted({r['status'] for r in rows}):
  group=[r['result'] for r in rows if r['status']==st and r['result']]
  costs.append({'category':'generation_'+st,'response_count':len(group),'known_total_tokens':sum(r['total_tokens'] or 0 for r in group),
    'missing_usage_count':sum(r['total_tokens'] is None for r in group),'cost_scope':'generation input+output; includes truncated responses in their own row'})
 embed=list(db.execute('select * from embed_calls'))
 costs.append({'category':'embedding_actual_calls','response_count':len(embed),'known_total_tokens':sum(r['tokens'] or 0 for r in embed),
 'missing_usage_count':sum(r['tokens'] is None for r in embed),'cost_scope':'actual cache-miss embedding input tokens, corpus+queries; cached retrieval reuses vectors'})
 write('token_costs.csv',costs,['category','response_count','known_total_tokens','missing_usage_count','cost_scope'])
 groups=defaultdict(list)
 for row in rows:
  if row['status']=='done':
   key=('C H5' if row['experiment']=='C' and row['stage']==5 else row['experiment'])
   if row['experiment']=='C' and row['stage']!=5:continue
   if row['experiment']=='RAG' and row['store']!='original':continue
   if row['experiment']=='RAG':key='RAG 원문'
   groups[key].append(row)
 metrics=[]
 for key,group in groups.items():
  result={'condition':key,'n':len(group)}
  for name in ['fact_retention','entity_retention','number_retention','output_tokens','total_tokens']:
   vals=[r['result'][name] for r in group if r['result'][name] is not None]
   result[name]=sum(vals)/len(vals) if vals else None
  # Same-stage, same-document, same-condition pairs only. RAG originals only here.
  cg=defaultdict(list)
  for r in group:cg[(r['source_id'],r['prompt_id'],r['stage'],r['question_id'],r['store'])].append(r['result']['fact_ids'])
  js=[ysc.pair_jaccard(a,b) for g in cg.values() for a,b in itertools.combinations(g,2)]
  js=[x for x in js if x is not None];result['consistency']=sum(js)/len(js) if js else None
  metrics.append(result)
 (out/'poster_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
 db.close();return {'completed_main_jobs':sum(r['status']=='done' for r in rows),'paired_RAG_rows':len(paired),'status':ysc.status_report(dbpath)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 print(json.dumps(create(a.db,a.out),ensure_ascii=False,indent=2))
