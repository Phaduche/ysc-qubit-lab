"""Streamlit Community Cloud 진입점.

배포된 웹에서는 '키워드/요약 자동 분석' 화면만 보여 준다. 로컬 모델(llama-server)이 없는 환경이므로
기본 모델 설정은 configs/api.json(OpenAI API)이고, API 키는 Streamlit secrets의 OPENAI_API_KEY에서 읽는다.
전체 연구실 UI(실험 A/B/C, RAG, 결과 다운로드)는 app.py로 로컬에서 실행한다.
"""
import os
from pathlib import Path
import streamlit as st
import keyword_analyzer

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Qubit 요약 분석 웹', layout='wide')
for key in ('OPENAI_API_KEY',):
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass
os.environ.setdefault('YSC_DEFAULT_CONFIG', 'api.json')
keyword_analyzer.render(ROOT)
st.caption('2026 YSC 발표대회 시연용 · 프롬프트별 핵심 정보 보존율 연구')
