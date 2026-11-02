"""Local Streamlit UI. Launch: streamlit run app.py"""
import io,json,re,subprocess,sys,zipfile,importlib
from pathlib import Path
import streamlit as st
import ysc
import report
import ragas_eval
import keyword_analyzer
importlib.reload(keyword_analyzer)

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'workspace';WORK.mkdir(exist_ok=True)
PROMPT_NAMES={'P1':'간결 요약','P2':'정보 보존 요약','P3':'항목별 요약'}
CONFIG_NAMES={'diagram_3repeats.json':'로컬 모델 · 3회 반복','local_smoke.json':'이전 검증 · 2회 반복','local_646.json':'로컬 모델 · 646건',
              'ollama.json':'Ollama','api.json':'API 모델','baseline.json':'추출식 기준선'}
CSV_NAMES={'results.csv':'전체 결과','summary.csv':'조건별 요약','sources_H0.csv':'원문과 정답표',
           'consistency_pairs.csv':'반복 일관성','paired_RAG_comparison.csv':'원문·요약본 RAG 비교',
           'annotation_review.csv':'사실 검토표','qa_review.csv':'답변 검토표','attempts.csv':'실행 기록',
           'unexecuted_dependencies.csv':'미실행 기록','embedding_calls.csv':'임베딩 사용량','token_costs.csv':'토큰 사용량'}

st.set_page_config(page_title='YSC 연구실',layout='wide')
st.markdown('<style>[data-testid="stToolbar"] {display: none;}</style>',unsafe_allow_html=True)
if st.query_params.get('view')=='analyzer':
    keyword_analyzer.render(ROOT)
    st.stop()
st.title('YSC 연구실')
tabs=st.tabs(['문서 준비','요약 실험','문서에 질문','결과 다운로드','키워드 분석'])
with tabs[4]:keyword_analyzer.render(ROOT)

with st.sidebar:
    st.header('실험')
    default_name='순서도 실험' if (WORK/'diagram_run'/'run.sqlite').exists() else '검증 결과' if (WORK/'verified_run'/'run.sqlite').exists() else '새 실험'
    display_name=st.text_input('실험 이름',default_name)
    run_name={'검증 결과':'verified_run','순서도 실험':'diagram_run'}.get(display_name,display_name)
    if not re.fullmatch(r'[\w -]{1,80}',run_name):
        st.error('이름에는 한글·영문·숫자·공백·밑줄·하이픈을 사용할 수 있습니다.');st.stop()
    run_dir=WORK/run_name;run_dir.mkdir(exist_ok=True)
    db=run_dir/'run.sqlite';data_path=run_dir/'dataset.json';config_path=run_dir/'config.json'
    if db.exists():
        try:
            counts=ysc.status_report(db)['jobs']
            st.metric('완료',counts.get('done',0))
            for key,label in [('running','진행 중'),('pending','미실행'),('failed','실패'),('truncated','출력 잘림')]:
                if counts.get(key,0):st.write(f'{label}  {counts[key]}')
            connection=ysc.connect(db)
            try:
                manifest=connection.execute('SELECT body FROM manifests').fetchone()
                if manifest:
                    kind=json.loads(manifest[0])['dataset'].get('kind','')
                    if kind=='supplied_document_excerpt_smoke':st.write('문서 발췌 검증')
            finally:connection.close()
        except Exception as e:st.error(str(e))

with tabs[0]:
    uploads=st.file_uploader('문서 선택',type=['txt','md','pdf','docx','json'],accept_multiple_files=True)
    left,right=st.columns(2)
    with left:import_clicked=st.button('불러오기')
    with right:example_clicked=st.button('예제 문서')
    if import_clicked:
        try:
            if not uploads:raise ValueError('문서를 선택하세요.')
            if len(uploads)==1 and uploads[0].name.endswith('.json'):
                data=json.loads(uploads[0].getvalue().decode('utf-8-sig'))
            else:
                docs=[]
                for u in uploads:
                    p=run_dir/('upload-'+ysc.digest(u.getvalue().hex())[:12]+Path(u.name).suffix)
                    p.write_bytes(u.getvalue());text=ysc.read_document(p)
                    docs.append({'id':'doc-'+ysc.digest(text)[:12],'source':Path(u.name).name,'text':text,
                                 'facts':[],'entities':[],'numbers':[],'questions':[]})
                data={'kind':'user_corpus_unannotated','documents':docs}
            st.session_state['draft_'+run_name]=json.dumps(data,ensure_ascii=False,indent=2)
        except Exception as e:st.error(str(e))
    if example_clicked:
        st.session_state['draft_'+run_name]=(ROOT/'data/supplied_smoke.json').read_text()
    initial=data_path.read_text() if data_path.exists() else '{"kind":"user_corpus","documents":[]}'
    with st.expander('원문·평가표 편집'):
        draft=st.text_area('원문·평가표',label_visibility='collapsed',
            value=st.session_state.get('draft_'+run_name,initial),height=400,
            key='editor_'+run_name+'_'+ysc.digest(st.session_state.get('draft_'+run_name,''))[:8])
        if st.button('저장'):
            try:
                if db.exists():raise ValueError('이 실험의 원문은 고정되어 있습니다. 새 실험 이름으로 저장하세요.')
                obj=ysc.validate_data(json.loads(draft))
                data_path.write_text(json.dumps(obj,ensure_ascii=False,indent=2));st.success('저장했습니다.')
            except Exception as e:st.error(str(e))

with tabs[1]:
    with st.expander('실험 흐름'):
        flow_tabs=st.tabs(['A · 1회 요약','B · 병렬 반복','C · 직렬 반복','RAG'])
        for flow_tab,kind in zip(flow_tabs,['A','B','C','RAG']):
            with flow_tab:
                flow_image=ROOT/'flowcharts'/(kind+'.png')
                if flow_image.exists():st.image(str(flow_image),width=700)
    config_names=list(CONFIG_NAMES)
    selected_index=0
    if db.exists() and config_path.exists():
        current_config=json.loads(config_path.read_text())
        for i,name in enumerate(config_names):
            if json.loads((ROOT/'configs'/name).read_text())==current_config:selected_index=i;break
    template=st.selectbox('모델 설정',config_names,index=selected_index,format_func=CONFIG_NAMES.get,disabled=db.exists())
    initial_config=config_path.read_text() if db.exists() and config_path.exists() else (ROOT/'configs'/template).read_text()
    with st.expander('세부 설정'):
        cfg_text=st.text_area('실험 설정',label_visibility='collapsed',value=initial_config,height=360,
                             key='cfg_'+run_name+'_'+template,disabled=db.exists())
    if st.button('계획 생성',disabled=db.exists()):
        try:
            if not data_path.exists():raise ValueError('먼저 문서 준비에서 원문과 평가표를 저장하세요.')
            cfg=json.loads(cfg_text);config_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
            n=ysc.init_run(config_path,data_path,db);st.success(f'{n}건을 계획했습니다.')
        except Exception as e:st.error(str(e))
    cap=st.number_input('이번 실행량',min_value=1,max_value=1000000,value=20)
    if st.button('실행 / 이어서 실행',disabled=not db.exists()):
        log=(run_dir/'batch.log').open('ab')
        subprocess.Popen([sys.executable,str(ROOT/'ysc.py'),'run','--db',str(db),'--limit',str(cap)],
                         stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        log.close();st.success('실행을 시작했습니다.')
    if st.button('새로고침'):st.rerun()
    if (run_dir/'batch.log').exists():
        with st.expander('실행 기록'):
            st.code((run_dir/'batch.log').read_text(errors='replace')[-7000:],language='text')

with tabs[2]:
    store=st.selectbox('검색 대상',['original','summary'],format_func=lambda x:{'original':'원문','summary':'요약본'}[x])
    pid,stage,rep='P2',5,0
    if store=='summary':
        pid=st.selectbox('요약 방식',list(PROMPT_NAMES),format_func=PROMPT_NAMES.get)
        stage=st.selectbox('요약 단계',[1,5],format_func=lambda x:{1:'첫 요약',5:'다섯 번째 요약'}[x])
        repeat_label=st.number_input('반복 회차',min_value=1,max_value=10001,value=1)
        rep=repeat_label-1
    question=st.text_input('질문')
    if st.button('답변 보기',disabled=not db.exists()):
        try:
            if not question.strip():raise ValueError('질문을 입력하세요.')
            with st.spinner('답변을 만들고 있습니다…'):r=ysc.ask(db,question,store,pid,stage,rep)
            st.write(r['output'])
            with st.expander('검색 근거'):
                for chunk in r['retrieved']:
                    st.write(chunk['text'])
        except Exception as e:st.error(str(e))

with tabs[3]:
    with st.expander('RAGAS 참고 평가'):
        eval_db=run_dir/'ragas.sqlite'
        eval_csv=run_dir/'ragas_csv'
        if eval_db.exists():
            eval_counts=ragas_eval.status_report(eval_db)['evaluation_tasks']
            st.write(f"평가 단계 완료 {eval_counts.get('done',0)} · 실패 {eval_counts.get('failed',0)} · 미실행 {eval_counts.get('pending',0)}")
        judge_default=json.loads(config_path.read_text())['model'] if config_path.exists() else {}
        if eval_db.exists():
            con=ragas_eval.connect(eval_db)
            try:
                row=con.execute('SELECT body FROM eval_manifest').fetchone()
                if row:judge_default=json.loads(row[0])['judge']
            finally:con.close()
        with st.expander('판정 모델 설정'):
            judge_text=st.text_area('판정 모델',value=json.dumps(judge_default,ensure_ascii=False,indent=2),height=250,
                                    key='judge_'+run_name,disabled=eval_db.exists())
        if st.button('평가 실행 / 이어서 실행',disabled=not db.exists()):
            try:
                if not eval_db.exists():ragas_eval.initialize(db,eval_db,pairs=2,judge=json.loads(judge_text))
                log=(run_dir/'ragas.log').open('ab')
                subprocess.Popen([sys.executable,str(ROOT/'ragas_eval.py'),'run','--db',str(eval_db),'--limit','20'],
                                 stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                log.close();st.success('평가를 시작했습니다.')
            except Exception as e:st.error(str(e))
        if st.button('평가 결과 갱신',disabled=not eval_db.exists()):
            try:ragas_eval.export_csv(eval_db,eval_csv);st.success('갱신했습니다.')
            except Exception as e:st.error(str(e))
        if (eval_csv/'ragas_metrics.csv').exists():
            import pandas as pd
            graded=pd.read_csv(eval_csv/'ragas_metrics.csv')
            display=pd.DataFrame({'검색 대상':graded['store'].map({'original':'원문','summary':'요약본'}),
                                  '근거 충실성':graded['faithfulness'],
                                  '질문 관련성':graded['answer_relevance'],
                                  '문맥 관련성':graded['context_relevance'],
                                  '상태':graded['review_status'].map({'review_pending':'검토 대기','needs_human_review':'수동 검토 필요'})})
            st.dataframe(display,hide_index=True,width='stretch')
            for fp in sorted(eval_csv.glob('*.csv')):
                st.download_button({'ragas_metrics.csv':'세 지표와 판정 근거','ragas_summary.csv':'평가 요약','ragas_evaluation_attempts.csv':'판정 실행 기록'}.get(fp.name,fp.stem),fp.read_bytes(),fp.name,'text/csv',key='ragas_'+fp.name)
    if st.button('결과 갱신',disabled=not db.exists()):
        try:report.create(db,run_dir/'csv');st.success('갱신했습니다.')
        except Exception as e:st.error(str(e))
    out=run_dir/'csv'
    if (out/'summary.csv').exists():
        import pandas as pd
        frame=pd.read_csv(out/'summary.csv')
        # The display omits identifiers and export metadata; CSVs retain the complete audit record.
        visible=pd.DataFrame(index=frame.index)
        visible['실험']=frame['experiment'].replace({'A':'1회 요약','B':'독립 반복','C':'연속 요약','RAG':'RAG'})
        visible['요약 방식']=frame['prompt_id'].map(PROMPT_NAMES).fillna('')
        visible['단계']=[f'{int(h)}회' if e=='C' or (e=='RAG' and store=='summary') else ''
                         for e,h,store in zip(frame['experiment'],frame['stage'],frame['store'])]
        visible['검색 대상']=frame['store'].map({'original':'원문','summary':'요약본'}).fillna('')
        visible['완료']=frame['n_successful_jobs']
        for source,label in [('fact_retention_mean','사실 보존율'),('entity_retention_mean','이름 유지율'),
                             ('number_retention_mean','숫자 유지율')]:
            visible[label]=frame[source].map(lambda x:'' if pd.isna(x) else f'{x*100:.1f}%')
        for source,label in [('output_tokens_mean','출력 토큰'),('total_tokens_mean','총 토큰')]:
            visible[label]=frame[source].map(lambda x:'' if pd.isna(x) else f'{x:.0f}')
        visible['일관성']=frame['consistency_fact_jaccard_mean'].map(lambda x:'' if pd.isna(x) else f'{x:.2f}')
        st.dataframe(visible,hide_index=True,width='stretch')
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():z.write(p,p.name)
        st.download_button('전체 다운로드',buf.getvalue(),'ysc_results.zip','application/zip')
        with st.expander('개별 CSV'):
            for p in sorted(out.glob('*.csv')):
                st.download_button(CSV_NAMES.get(p.name,p.stem),p.read_bytes(),p.name,'text/csv',key=p.name)
