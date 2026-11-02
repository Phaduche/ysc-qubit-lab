"""Restartable RAG QA on the actual A/B/C summary stores. Exact-call reuse is labeled."""
import json,sqlite3,argparse
from pathlib import Path
from collections import defaultdict
import ysc

def run(source_db,dest):
    src=sqlite3.connect(Path(source_db).resolve().as_uri()+'?mode=ro',uri=True)
    frozen=json.loads(src.execute('select body from manifests').fetchone()[0]);cfg=frozen['config'];docs=frozen['dataset']['documents']
    db=ysc.connect(dest);cache={};groups=defaultdict(dict)
    for sp,res in src.execute("select spec,result from jobs where status='done'"):
        s=json.loads(sp);r=json.loads(res)
        if s['experiment'] in ['A','B','C']:groups[(s['experiment'],s['prompt_id'],s['repeat'],s['stage'])][s['source_id']]=r
        if s['experiment']=='RAG':cache[ysc.digest({'prompt':r['prompt'],'model':cfg['model'],'seed':s['seed']})]=r
    for row in src.execute('select id,vector,metadata from embeddings'):db.execute('insert or ignore into embeddings values (?,?,?)',row)
    manifest={'source_db':str(source_db),'source_code_hash':frozen['code_sha256'],'config':cfg,'dataset':frozen['dataset'],'method':'abc_summary_stores_same_questions_exact_call_cache_v1'}
    old=db.execute('select body from manifests').fetchone()
    if old:
        prior=json.loads(old[0])
        if any(prior[k]!=manifest[k] for k in ['source_code_hash','config','dataset','method']):raise ValueError('다른 설정이나 원문으로 만든 평가 DB입니다. 새 DB를 지정하세요.')
    else:db.execute('insert into manifests values (?,?)',(ysc.digest(manifest),ysc.dumps(manifest)))
    for (exp,pid,rep,stage),parts in groups.items():
        if len(parts)!=len(docs):continue # A missing corpus document makes this store unavailable.
        corpus=[{'id':d['id'],'text':parts[d['id']]['output']} for d in docs]
        for doc in docs:
            for q in doc.get('questions',[]):
                spec={'experiment':exp,'prompt_id':pid,'repeat':rep,'stage':stage,'source_id':doc['id'],'question':q,'seed':cfg['seed']+rep,'corpus':corpus}
                sid=ysc.digest(spec);db.execute('insert or ignore into jobs(id,experiment,spec,status) values (?,?,?,?)',(sid,exp,ysc.dumps(spec),'pending'))
    db.commit()
    # Completed destinations are eligible for exact-call caching on resume.
    for (body,) in db.execute("select result from jobs where status='done'"):
        r=json.loads(body);cache[r['call_key']]=r['generation']
    for row in db.execute("select id,spec from jobs where status in ('pending','failed','running') and attempts<2 order by rowid").fetchall():
        jid,spec=row;s=json.loads(spec);q=s['question'];aid=db.execute("insert into attempts(job_id,started,status) values (?,?,'running')",(jid,ysc.now())).lastrowid
        db.execute("update jobs set status='running',attempts=attempts+1 where id=?",(jid,));db.commit()
        try:
            chunks,index=ysc.retrieve(dest,cfg['retrieval'],s['corpus'],q['question'])
            evidence='\n\n'.join('['+ch['citation_id']+'] '+ch['text'] for ch in chunks)
            prompt=ysc.QA_PROMPT+'\n검색 근거:\n'+evidence+'\n질문:\n'+q['question']
            key=ysc.digest({'prompt':prompt,'model':cfg['model'],'seed':s['seed']});reused=key in cache
            res=cache[key] if reused else ysc.Backend(cfg['model']).generate(prompt,s['seed'],evidence,True)
            if res['finish_reason'] in ['length','max_tokens']:raise ValueError('출력 잘림')
            cache[key]=res
            groups_=q.get('answer_groups',[q['answers']]);correct=int(all(any(ysc.spans(res['output'],a) for a in alternatives) for alternatives in groups_))
            result={'output':res['output'],'qa_correct':correct,'generation':res,'call_key':key,'reused_actual_call':reused,'retrieved':chunks,'index_id':index,'prompt':prompt,'data_kind':'supplied_document_excerpt_smoke','metric':'answer_alias_proxy','completed_at':ysc.now()}
            db.execute("update jobs set status='done',result=?,error=null where id=?",(ysc.dumps(result),jid));db.execute("update attempts set ended=?,status='done' where id=?",(ysc.now(),aid))
        except Exception as e:
            db.execute("update jobs set status='failed',error=? where id=?",(str(e),jid));db.execute("update attempts set ended=?,status='failed',error=? where id=?",(ysc.now(),str(e),aid))
        db.commit()
        counts=dict(db.execute('select status,count(*) from jobs group by status').fetchall());print(ysc.dumps(counts),flush=True)
    db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='results/diagram_3repeats.sqlite');p.add_argument('--db',default='results/abc_rag_demo.sqlite');a=p.parse_args();run(a.source,a.db)
