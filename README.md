# Qubit 요약 분석 웹 (YSC 연구실)

프롬프트에 따라 반복 요약에서 핵심 정보가 얼마나 남는지 보는 Qubit 팀(민족사관고 신민기, 이시우, 우승제)의 실험 도구입니다.
2026 YSC 발표대회 보조 포스터의 QR이 이 저장소의 웹 페이지로 연결됩니다.

- 웹(QR 연결): https://phaduche.github.io/ysc-qubit-lab/ → Streamlit 앱 https://ysc-qubit-lab.streamlit.app
- 웹 화면: `streamlit_app.py` — 원문을 넣으면 핵심 키워드 10개 추출 → P1/P2/P3 프롬프트로 요약 → 핵심 사실 보존율, 남은 항목, 출력 토큰 수를 보여 줌
- 전체 연구실(실험 A/B/C, RAG, 결과 다운로드): `app.py` — 로컬 llama-server 전용 (아래 원래 README 참고)

## 웹 배포 (Streamlit Community Cloud)

1. https://share.streamlit.io 에서 **New app** → Repository `Phaduche/ysc-qubit-lab`, Branch `main`, Main file `streamlit_app.py`
2. **App URL**(subdomain)을 `ysc-qubit-lab` 으로 지정 (QR의 이동 주소와 같아야 함; 다른 이름을 쓰면 `docs/index.html`의 주소를 바꿀 것)
3. **Advanced settings → Secrets** 에 `OPENAI_API_KEY = "sk-..."` 추가 (키는 저장소에 넣지 않음)
4. Deploy. 이후에는 `main`에 머지될 때마다 자동으로 다시 배포됨

## 고치는 방법 (PR 흐름)

```bash
git switch -c fix/설명
# 수정 후
git commit -am "무엇을 바꿨는지"
git push -u origin fix/설명
gh pr create --fill
```
PR을 `main`에 머지하면 웹이 자동으로 갱신됩니다. 로컬 확인: `pip install -r requirements.txt && OPENAI_API_KEY=... YSC_DEFAULT_CONFIG=api.json streamlit run streamlit_app.py`

---

# YSC 프롬프트 · 반복 요약 · RAG 연구실

실험 A/B/C와 실제 RAG 파이프라인을 실행하고, 원문·설정·평가 근거·실패 이력을 CSV로 내보내는 로컬 프로그램입니다. `app.py`는 한국어 화면, `ysc.py`는 배치 실행기, `report.py`는 상세 비교표 출력 도구입니다.

**실행 사실:** 본 패키지의 `results/validation_csv`는 사용자 제공 문서 2개에서 발췌·정리한 기능 검증 코퍼스와 Qwen2.5-0.5B-Instruct Q4_K_M / Nomic 임베딩을 실제 사용했습니다. 계획 134건 중 정상 완료 120건(A 6, B 11, C 55, RAG 48), 출력 길이 제한 2건, 의존 단계 미완료로 미실행 12건입니다. 500회 수행한 결과가 아닙니다. 본래 연구의 모의고사 코퍼스·LoRA 가중치·기존 프로그램 소스·원시 결과 CSV는 제공되지 않았습니다. 따라서 기존 H0→H5 방법을 독립 실행 가능한 도구로 구현했으며, 기존 코드나 LoRA 가중치를 복구했다고 주장하지 않습니다.

`results/preflight_csv`는 본 검증 전에 수행한 실제 요약 2건과 당시 계획 미실행 행입니다. 실행기 설계 점검용이며 본 검증 120건과 합산하거나 실험 A 결과에 편입하지 않았습니다. 주요 집계는 `validation_csv`만 사용하세요. 단위 테스트의 모의 응답은 `simulation_fixture`로 표시되며 연구 CSV에 포함하지 않습니다.

## 처음 실행

Python 3.10 이상이 필요합니다. 터미널에서 이 폴더로 이동한 뒤:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

화면을 먼저 열려면:

```sh
streamlit run app.py --server.address 127.0.0.1
```

화면 순서: ① 문서 업로드/정답표 저장 → ② 모델·검색 설정 고정 및 계획 → 배치 실행 → ③ 원문/요약본 저장소 질의 → ④ CSV 다운로드. 설정과 원문을 고정한 실행은 수정할 수 없습니다. 다른 조건은 새로운 실행 이름/DB로 만드세요. 화면을 닫아도 시작된 배치는 계속됩니다.

## 실제 로컬 모델 실행

이번 검증과 동일한 Apple Silicon Mac용 예제를 제공합니다. 모델·실행기는 약 590 MB를 다운로드하며 전역 설치 없이 프로젝트의 `.runtime/` 안에 저장합니다. 공식 모델 저장소와 llama.cpp 공식 릴리스에서 내려받고 SHA-256을 검사합니다.

```sh
sh scripts/download_local.sh
sh scripts/start_local.sh
```

위 터미널을 유지한 상태에서 다른 터미널로 앱/배치를 실행하세요. 서버 주소는 생성 `127.0.0.1:8765`, 임베딩 `127.0.0.1:8766`입니다. 이미 해당 포트를 사용하는 서버가 있으면 로그를 확인하고 포트와 설정을 함께 바꾸세요. 종료는 서버 터미널에서 Ctrl+C입니다.

다른 운영체제나 기존 모델은 `configs/ollama.json`의 생성 모델·임베딩 모델 이름을 설치된 모델 이름으로 바꾸세요. 생성과 임베딩은 별도 모델입니다. 원래 연구의 LoRA 모델이 준비되면 호환 로컬 서버에 올리고 모델명·해시를 새 설정에 기록하면 됩니다.

API 모델은 `configs/api.json`의 모델 ID를 계정에서 사용할 수 있는 정확한 ID로 바꾸고 `OPENAI_API_KEY` 환경변수를 설정합니다. 키를 JSON·CSV에 넣지 마세요. API 모델 인터페이스의 요청/사용량 처리 단위 테스트는 했지만 **외부 API 실호출은 키가 없어 수행하지 않았습니다.** 모델에 따라 temperature, seed, max_tokens 지원이 달라집니다. `supports_seed`, `max_tokens_field`, `extra`를 해당 서버 사양에 맞추고 새 DB를 사용하세요. `seed_sent`는 요청 전송 여부이며 완전한 재현성 보장이 아닙니다.

## 배치와 500회 이상 계획

주요 검증과 같은 134작업 계획:

```sh
python ysc.py init --config configs/local_smoke.json --data data/supplied_smoke.json --db workspace/new_smoke.sqlite
python ysc.py run --db workspace/new_smoke.sqlite
python report.py --db workspace/new_smoke.sqlite --out workspace/new_smoke_csv
```

646작업 확장 계획(이번에 실행하지 않음):

```sh
python ysc.py init --config configs/local_646.json --data data/supplied_smoke.json --db workspace/batch646.sqlite
python ysc.py run --db workspace/batch646.sqlite --limit 50
python ysc.py status --db workspace/batch646.sqlite
python ysc.py run --db workspace/batch646.sqlite
python report.py --db workspace/batch646.sqlite --out workspace/batch646_csv
```

646 = A 6 + B 60 + C 300 + RAG 280. 한 작업은 생성 모델 요청 1건입니다. H0 원문, 청크, 임베딩 호출 수를 생성 시행 수에 더하지 않습니다. C의 300개 출력은 60개 체인×5단계이며 독립 표본 300개를 뜻하지 않습니다. A는 원문×프롬프트별 1회, B와 C는 반복 10회, RAG는 4질문×7저장소×10회입니다. 원문 2개를 646번 생성해도 서로 다른 원문 646개가 되지는 않습니다. 본 실험에서는 충분한 수의 서로 다른 원문과 독립 질문을 준비해야 합니다.

무제한 수의 원문·반복을 지정할 수 있고 `workers`로 독립 작업의 동시 실행 수를 설정할 수 있습니다. B의 ‘병렬’은 입력이 독립이라는 실험 설계입니다. 처리 자원이 부족하면 `workers: 1`로 순차 처리해도 설계상 병렬 반복입니다. C 체인 내부는 순서를 지키며 서로 독립인 체인만 동시 실행할 수 있습니다.

중단 후 같은 DB로 `run`을 다시 호출하면 완료 작업은 재사용하고, 중단/실패한 작업은 `max_attempts` 한도까지 재시도합니다. 실행기 충돌 방지 잠금은 프로세스 종료 시 풀립니다. 재시작은 기존 `running` 시도를 `interrupted`로 남깁니다. `--limit`은 이번 프로세스의 시도 수 한도이며 성공 수 한도가 아닙니다.

`truncated`는 응답이 길이 제한에 도달한 상태로 자동 재시도하지 않습니다. 그 출력은 CSV에 보관하되 후속 C/RAG 입력이나 성공 평균에 쓰지 않습니다. 출력 예산이나 프롬프트를 바꾸려면 새 DB를 만들어 비교하세요. 최종 `complete: false`는 12개 미실행 의존 작업이 남아 있다는 뜻이며 은폐된 성공이 아닙니다. 오류·실행 중 상태에서도 CSV를 내보낼 수 있습니다. API 응답이 서버에서 생성된 직후 로컬 저장 전에 프로세스가 죽으면 재시도로 원격 요청이 중복될 수 있으므로, 외부 과금 내역과 로컬 완료 건수는 별도로 대조해야 합니다.

## 원문과 정답표

TXT·MD·PDF·DOCX를 화면에서 업로드하거나:

```sh
python ysc.py import my_article.pdf my_notes.docx --out workspace/my_dataset.json
```

PDF는 텍스트 추출형을 지원합니다. 스캔 PDF는 OCR 후 가져오세요. 가져온 원문은 그대로 저장하고 `source`와 `id`를 보존합니다. 아직 정답이 없는 항목은 0점이 아니라 N/A로 계산합니다. 본 연구 전 `data/supplied_smoke.json` 형식을 따라 정답과 질문을 사전 작성·고정하세요.

- `facts`: 한 개의 명제에 ID, `statement`, `alternatives`를 작성합니다. alternatives는 OR, 그 안의 표현 목록은 AND입니다. 예: `[["병렬", "원문", "독립"], ["같은 입력", "독립 요약"]]`.
- `entities`: 원문 기준 고유명사의 ID와 허용 별칭 `aliases`입니다. 같은 이름을 여러 번 반복해도 한 항목으로 셉니다.
- `numbers`: 숫자·단위·날짜 등 묶음의 ID와 허용 표기입니다. `12%`를 정답으로 주면 `112%`를 정답으로 세지 않습니다. 의미가 다른 `12명`과 `12%`를 같은 별칭에 넣지 마세요.
- `questions`: 질문 ID, 질문, 정답 별칭 `answers`, 필요하면 AND 조건의 `answer_groups`를 지정합니다. facts/entities/numbers에는 해당 질문에서 요구하는 정답 ID만 넣습니다. RAG 답변에 원문 전체 사실을 재현하도록 요구하지 않습니다.
- 정답표를 실험 결과에 맞춰 사후 수정하지 마세요. 변경 시 별도 버전과 새 DB로 실행합니다.

원문 전체, 정답표, 세 프롬프트 전문, 시스템 지시, QA 지시, 생성·검색 조건, 코드 SHA-256은 `manifest.json`에 동결됩니다. 모든 조건은 동일한 정답표를 씁니다. 문서를 입력 데이터로 취급하도록 시스템 지시를 넣었지만 생성 모델의 지시 준수나 근거 충실성을 자동 보장하지는 않습니다.

## 실험 설계

| 실험 | 입력과 반복 | 평가 기준 |
|---|---|---|
| A | H0를 P1/P2/P3로 각각 1회 요약 | 첫 요약의 보존·길이·토큰 |
| B | 매 요청에 동일 H0, 프롬프트 고정, 반복마다 다른 seed | 독립 반복 변동과 일관성 |
| C | H0→H1→H2→H3→H4→H5, 체인마다 seed 고정 | 매 단계 H0 대비 보존, 같은 단계 독립 체인 간 일관성 |
| RAG | 원문 / P1·P2·P3 H1 / P1·P2·P3 H5의 7종 저장소 | 같은 질문·seed·검색 조건의 QA와 보존 비교 |

RAG는 전체 코퍼스를 검색합니다. 정답 원문만 미리 골라 주지 않습니다. 청킹은 유니코드 문자 수 기준 고정 길이+겹침, 임베딩은 지정 모델, 검색은 벡터 코사인 유사도 top-k입니다. 검색 청크의 원문 ID, 범위, 텍스트, 해시, 점수와 인용 ID를 보존합니다. 요약 저장소는 모든 원문에 대해 해당 프롬프트·H단계·반복의 요약이 완성되어야 구성됩니다. 청크와 벡터, 저장소 스냅샷은 SQLite에 보관됩니다. 원문 저장소는 반복 간 고정되며 요약 저장소는 해당 체인의 결과를 사용합니다. 벡터 캐시에는 모델/설정과 텍스트 해시가 포함됩니다. 모델 이름이 같아도 가중치가 바뀌면 revision을 바꾸고 새 DB를 쓰세요.

이번 임베딩 모델 Nomic v1.5는 기능 검증용 선택입니다. **한국어 검색 성능을 검증한 모델이라는 결론은 내리지 않았습니다.** 본 연구에서는 한국어/다국어 임베딩 모델을 고정하고 별도의 검색 정답셋으로 평가하세요. 서로 다른 모델 간 토큰 수는 각 모델의 토크나이저 단위가 다르므로 동일 척도의 압축 비율로 바로 비교하지 마세요.

자유 질문(정답 없는 탐색)은 다음과 같습니다. 배치 결과와 섞지 않고 `.adhoc.jsonl`에 따로 저장합니다.

```sh
python ysc.py ask --db results/validation.sqlite --store original --question '기존 핵심 키워드는 몇 개인가?'
python ysc.py ask --db results/validation.sqlite --store summary --prompt-id P2 --stage 5 --repeat 0 --question '기존 핵심 키워드는 몇 개인가?'
```

## 다섯 지표의 정확한 의미

| 지표 | 구현 | 해석 한계 |
|---|---|---|
| 핵심 사실 보존율 | 사전 고정 명제의 필요 표현 조건을 만족한 수 / 기준 명제 수 | **표현 일치 자동 대리지표**. 부정·인과 역전·모순·의미적 동치를 확정하지 못함 |
| 고유명사 유지율 | 사전 지정 이름의 별칭이 나온 수 / 기준 이름 수 | 이름이 나와도 올바른 관계에 쓰였는지 별도 검토 |
| 숫자 정보 유지율 | 숫자·단위·날짜 표기 묶음 일치 수 / 기준 묶음 수 | 단위 변환·표기 변형은 사전 별칭 없으면 놓칠 수 있음 |
| 출력·총 토큰 | 모델 응답 usage의 출력 토큰과 입력+출력 합 | usage 없으면 N/A. 문자 수를 토큰이라고 부르지 않음 |
| 반복별 일관성 | 같은 원문·프롬프트·단계·질문·저장소 내 독립 반복의 보존 사실 집합 Jaccard 평균 | 양쪽이 모두 빈 집합이면 N/A. 일관성이 높아도 정답성 보장은 아님 |

A는 원문·조건당 1회이므로 일관성 N/A입니다. C의 H1–H2 인접 단계 비교는 `adjacent_step_fact_jaccard`로 별도 기록하며 독립 반복 일관성과 구분합니다. RAG의 ①~③은 질문이 요구하는 정답 항목을 분모로 씁니다. 없는 분모, 한 개뿐인 반복 집단은 공란/N/A입니다. 집계 평균은 각 완료 출력 비율의 단순 평균이며 문서별 분모를 모두 합한 micro 평균과 다릅니다. 독립 표본 수를 늘리지 않는 반복 때문에 신뢰구간이나 유의성 검정을 자동으로 주장하지 않습니다.

RAG QA의 `qa_exact_alias_correct`는 필요한 정답 표현이 포함되었는지의 대리지표입니다. `all_citations_valid`는 청크 ID가 실제 검색 목록에 있는지만 확인합니다. 이것만으로 답변이 근거에 의해 지지된다고 확정할 수 없습니다. `qa_review.csv`와 `annotation_review.csv`에 사람의 정답성·근거성·모순 검토를 기입하세요. 자동 점수는 수동 점수로 둔갑시키지 않습니다. 이 패키지는 검토표 입력을 자동 집계해 공식 점수로 병합하지 않습니다.

토큰의 `total_tokens`는 **그 생성 요청의 입력+출력**입니다. 요약 저장소를 만드는 생성 비용은 C 작업, 검색 임베딩 비용은 `embedding_calls.csv`에 따로 기록되어 있습니다. `token_costs.csv`는 잘린 응답까지 포함한 실제 기록 비용을 상태별로 구분합니다. 저장소 구축비를 질의마다 다시 더하거나, 캐시된 벡터를 새 임베딩 호출로 세지 마세요. 원문/요약본의 end-to-end 비용은 구축 1회+질의 횟수의 비용으로 별도 비교해야 합니다. 제공된 `paired_RAG_comparison.csv`의 토큰 열은 질의 생성 비용만 비교합니다.

## 출력 파일

`results/validation_csv`의 주요 파일:

- `results.csv`: 계획 134행, 각 행에 상태·실행 조건·모델명·seed·원문 ID·입출력·사용량·자동 점수·근거·raw 응답 포함. 정상 완료만 120행입니다.
- `sources_H0.csv`: H0 원문과 출처, 해시, 동결 정답표.
- `summary.csv`: 완료 출력만 집계한 조건별 평균·유효 분모·원문 수.
- `consistency_pairs.csv`: 반복 쌍 ID, 사실 집합, Jaccard 원자료.
- `paired_RAG_comparison.csv`: 같은 질문/반복의 원문 vs 요약본 비교 40쌍.
- `annotation_review.csv`, `qa_review.csv`: 자동 검출 근거와 미기입 수동 검토 칸.
- `attempts.csv`: 시작·종료·중단·실패·출력 잘림 이력.
- `unexecuted_dependencies.csv`: 미실행 12건과 아직 충족되지 않은 선행 단계.
- `embedding_calls.csv`, `token_costs.csv`: 실제 임베딩 호출과 사용량, 상태별 토큰 집계.
- `manifest.json`, `status.json`: 설정·원문·프롬프트·코드 해시와 실제 집계.

CSV는 UTF-8 BOM으로 저장되어 한국어 Excel에서 열 수 있습니다. 중첩된 평가 근거는 JSON 문자열로 한 셀에 보존합니다. `results/validation.sqlite`에는 재시작에 필요한 작업·청크·벡터도 저장되어 있습니다. 결과 DB는 원문을 포함하므로 다른 사람에게 공유할 때 원문 이용 권한을 먼저 확인하세요.

## 테스트와 포스터

```sh
python -m unittest discover -s tests -v
```

15개 테스트: 숫자/이름 경계, 명제의 AND 조건, 정답 검증, A/B/C/RAG 계획과 의존 순서, 646건 계획, 설정 변경 차단, 재시작·실패 이력, 실행기 중복 방지, 전체 코퍼스 검색, usage 누락, N/A 집계, 잘린 출력 제외를 검증했습니다. 모의 응답을 쓰는 테스트는 연구 데이터와 분리됩니다.

상위 폴더의 `YSC_A0_editable.pptx`는 사용자 제공 코드페어 포스터의 첫 슬라이드에 있는 남색 제목 띠·큰 외곽 프레임·화살표 소제목·내부 박스를 재사용한 편집 가능한 중앙 포스터 1장입니다. 원본의 A0 세로 치수(약 841×1189mm)를 유지했습니다. 실제 완료 수와 자동 점수의 기술통계, 미실행 항목을 구분해 넣었습니다. 학교/학생 이름은 넣지 않았습니다. 이름 기재 가능 여부나 최신 대회 제출 규정을 이번 구현 작업에서 새로 확인하지 않았으므로 제출 전 담당자 확인이 필요합니다. 본 실험에서 사용할 원문·모델의 새 결과가 나오면 표와 상태를 갱신하세요.

공식 기술 참고는 `SOURCES.md`, 모델과 실행기 다운로드의 정확한 출처·해시는 `MODEL_PROVENANCE.json`에 있습니다.
