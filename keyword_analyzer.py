"""Interactive keyword extraction and summary analysis, separate from batch results."""
import csv,io,json,os,re,html
from pathlib import Path
import streamlit as st
import ysc

EXAMPLE=('딥러닝은 인공신경망을 이용하여 데이터의 특징을 학습한다. 입력층은 데이터를 받아들이고, '
         '은닉층은 가중치를 적용하여 특징을 추출하며, 출력층은 예측 결과를 낸다. 학습 과정에서는 '
         '손실함수로 예측 오차를 계산하고 역전파와 경사하강법으로 가중치를 조정한다. '
         '과적합을 줄이기 위해 정규화를 적용하고, 검증데이터로 모델의 일반화 성능을 확인한다.')

def keyword_hits(summary,keywords):
    if len(keywords)!=10 or len({ysc.norm(k) for k in keywords})!=10:
        raise ValueError('서로 다른 핵심 키워드 10개를 입력하세요.')
    return [{'키워드':k,'보존':bool(ysc.spans(summary,k))} for k in keywords]

def extract(source,model,seed):
    candidates=[]
    for word in re.findall(r'[가-힣A-Za-z0-9][가-힣A-Za-z0-9·-]{1,15}',source):
        word=re.sub(r'(한다|했다|하며|하여|하고|이며|이다)$','',word)
        stem=re.sub(r'(으로|에서|에게|은|는|이|가|을|를|와|과|의|에|도|만|로)$','',word)
        for value in [stem]:
            if value in {'이용','적용','위해','통해','받아들이고','낸다','줄이기','있다'}:continue
            if len(value)>1 and value not in candidates and ysc.spans(source,value):candidates.append(value)
    if len(candidates)<10:raise ValueError('원문이 짧습니다. 서로 다른 키워드 후보가 10개 이상인 원문을 입력하세요.')
    schema={'type':'object','properties':{'keywords':{'type':'array','minItems':10,'maxItems':10,'items':{'type':'string','enum':candidates}}},'required':['keywords'],'additionalProperties':False}
    cfg={**model,'max_tokens':400,'temperature':0,
         'extra':{**model.get('extra',{}),'response_format':{'type':'json_schema','json_schema':{'name':'keywords','strict':True,'schema':schema}}}}
    res=ysc.Backend(cfg).generate('후보 목록에서 원문의 핵심 명사·전문용어 10개를 고르세요. 문장이나 서술어를 고르지 말고 서로 다른 개념을 고르세요. 동일한 단어나 그 조사 형태를 중복 선택하지 마세요. keywords 배열의 JSON만 출력하세요.\n후보:\n'+', '.join(candidates)+'\n원문:\n'+source,seed,source)
    if res['finish_reason'] in ['length','max_tokens']:raise ValueError('키워드 추출이 잘렸습니다. 다시 실행하세요.')
    keywords=json.loads(res['output'])['keywords'];keyword_hits('',keywords)
    if not all(ysc.spans(source,k) for k in keywords):raise ValueError('원문에 없는 키워드가 있어 검토가 필요합니다.')
    return keywords,res

def render(root):
    st.markdown('<style>[data-testid="stHeader"] {display:none;} [data-testid="stMainBlockContainer"] {padding-top: .2rem; padding-bottom: 1rem;} [data-testid="stVerticalBlock"] {gap: .55rem;}</style>',unsafe_allow_html=True)
    # Reload a real persisted interactive run for inspection, without model calls.
    if st.query_params.get('record')=='latest' and 'kw_result' not in st.session_state:
        for fp in sorted((root/'workspace'/'interactive_analysis').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            previous=json.loads(fp.read_text())
            if previous.get('status')=='done':
                st.session_state['kw_source']=previous['source'];st.session_state['kw_source_hash']=ysc.digest(previous['source'])
                st.session_state['kw_keywords']=', '.join(previous['keywords']);st.session_state['kw_summary']=previous['summary']['output'];st.session_state['kw_result']=previous
                st.session_state['kw_prompt']=previous['prompt_id'];break
    st.session_state.setdefault('kw_source',EXAMPLE)
    st.header('키워드·요약 자동 분석')
    with st.expander('모델 설정'):
        names=['local_smoke.json','api.json']
        name=st.selectbox('분석 모델',names,index=names.index(os.getenv('YSC_DEFAULT_CONFIG','local_smoke.json')),format_func=lambda x:'로컬 모델' if x.startswith('local') else 'API 모델',key='kw_model')
        cfg=st.text_area('모델 연결',json.dumps(ysc.load(root/'configs'/name)['model'],ensure_ascii=False,indent=2),height=180,key='kw_cfg_'+name)
    left,right=st.columns([1,1])
    with left:
        st.subheader('원문과 핵심 키워드')
        source=st.text_area('원문 · 예제 지문',height=140,key='kw_source')
        pid=st.selectbox('요약 프롬프트',list(ysc.PROMPTS),format_func=lambda p:{'P1':'P1 일반형','P2':'P2 사실 보존형','P3':'P3 구조화형'}[p],key='kw_prompt')
        if st.button('키워드 추출 + 요약',type='primary',key='kw_run'):
            run={'created_at':ysc.now(),'data_kind':'interactive_example','source':source,'prompt_id':pid,'seed':1729}
            try:
                model=json.loads(cfg);run['model']=model
                with st.spinner('키워드와 요약을 생성하고 있습니다.'):
                    keywords,ex=extract(source,model,1729)
                    sm=ysc.Backend(model).generate(ysc.PROMPTS[pid]+'\n자료:\n'+source,1729,source)
                    if sm['finish_reason'] in ['length','max_tokens']:raise ValueError('요약 출력이 잘렸습니다.')
                run.update(status='done',keywords=keywords,extraction=ex,summary=sm)
                st.session_state['kw_keywords']=', '.join(keywords)
                st.session_state['kw_summary']=sm['output']
                st.session_state['kw_source_hash']=ysc.digest(source)
                st.session_state['kw_result']=run
                st.session_state['kw_numbers']=', '.join(dict.fromkeys(re.findall(r'(?<![0-9])\d+(?:\.\d+)?(?:%|년|월|일|개|회)?',source)))
            except Exception as e:
                run.update(status='failed',error=str(e));st.error(str(e))
            out=root/'workspace'/'interactive_analysis';out.mkdir(parents=True,exist_ok=True)
            (out/(ysc.digest(run)[:20]+'.json')).write_text(json.dumps(run,ensure_ascii=False,indent=2))
        terms=st.text_area('핵심 키워드 10개',key='kw_keywords',height=75)
        with st.expander('숫자·고유명사 검토'):
            numbers=st.text_input('숫자 항목',key='kw_numbers')
            names=st.text_input('고유명사 항목',key='kw_names')
        st.write('핵심 사실 보존율 = 남은 항목 수 ÷ 전체 항목 수 × 100')
    with right:
        st.subheader('요약과 분석 결과')
        summary=st.text_area('생성된 요약',key='kw_summary',height=140)
        if terms.strip() and summary.strip():
            try:
                keys=[x.strip() for x in terms.replace('，',',').split(',') if x.strip()]
                if st.session_state.get('kw_source_hash')!=ysc.digest(source):raise ValueError('원문이 바뀌었습니다. 키워드와 요약을 다시 생성하세요.')
                if not all(ysc.spans(source,k) for k in keys):raise ValueError('원문에 없는 키워드를 확인하세요.')
                hits=[{'항목':h['키워드'],'유형':'키워드','보존':h['보존']} for h in keyword_hits(summary,keys)]
                for cat,body in [('숫자',numbers),('고유명사',names)]:
                    for value in dict.fromkeys(x.strip() for x in body.split(',') if x.strip()):
                        if not ysc.spans(source,value):raise ValueError('원문에 없는 항목을 확인하세요: '+value)
                        hits.append({'항목':value,'유형':cat,'보존':bool(ysc.spans(summary,value))})
                n=sum(h['보존'] for h in hits);denom=len(hits);score=n/denom
                a,b=st.columns(2);a.metric('핵심 사실 보존율',f'{score:.0%}');b.metric('남은 항목',f'{n} / {denom}')
                body=''.join('<tr>'+''.join('<td style="padding:6px;border-bottom:1px solid #e4e6eb">'+html.escape(v)+'</td>' for v in [h['항목'],h['유형'],'유지' if h['보존'] else '누락'])+'</tr>' for h in hits)
                st.markdown('<div style="height:200px;overflow:auto;border:1px solid #e4e6eb;border-radius:8px;background:white;color:#111"><table style="width:100%;border-collapse:collapse;text-align:left"><thead><tr>'+''.join('<th style="padding:6px;background:#f7f8fa">'+v+'</th>' for v in ['항목','유형','결과'])+'</tr></thead><tbody>'+body+'</tbody></table></div>',unsafe_allow_html=True)
                result=st.session_state.get('kw_result',{});sm=result.get('summary',{})
                st.write(f"출력 {sm.get('output_tokens','—')}토큰 · 모델 {sm.get('model_actual','—')}")
                f=io.StringIO();w=csv.DictWriter(f,fieldnames=['항목','유형','보존','보존율','자료종류','모델','프롬프트','원문해시']);w.writeheader()
                w.writerows({**v,'보존율':score,'자료종류':'interactive_example','모델':sm.get('model_actual'),'프롬프트':result.get('prompt_id'),'원문해시':ysc.digest(source)} for v in hits)
                st.download_button('분석 CSV 다운로드',f.getvalue().encode('utf-8-sig'),'keyword_analysis.csv','text/csv')
            except ValueError as e:st.error(str(e))
