# RAGAS를 참고한 RAG 평가

`ragas_eval.py`는 Es 외(2024)의 **RAGAS 논문을 참고한 별도 구현**이다. 공식 `ragas` 패키지를 설치하거나 호출하지 않으며, 원 논문의 정확한 재현이나 동일한 성능을 주장하지 않는다. 기존 실험의 다섯 지표와 별도로 저장한다.

원문: [Es, James, Espinosa-Anke, Schockaert (2024), RAGAS: Automated Evaluation of Retrieval Augmented Generation, EACL System Demonstrations, 150–158](https://aclanthology.org/2024.eacl-demo.16/).

## 평가 흐름

기존 RAG에서 완료된 답변과 당시 실제 검색된 청크를 고정한다. 답변을 다시 생성하거나 검색을 다시 수행하지 않는다.

`완료된 RAG 답변·검색 근거 → 주장 분해·역질문 생성·필요 문장 선택 → 3축 자동 평가 → 판정 근거·실패 기록·CSV`

| 지표 | 이 구현의 계산 | 해석 |
|---|---|---|
| Faithfulness | 답변을 짧은 사실 주장으로 분해 → 주장마다 검색 근거의 지지를 판정 → 지지된 주장 수 / 전체 주장 수 | 문맥에 근거하는가. 외부 세계에서의 진실성을 직접 확인하지 않는다. |
| Answer Relevance | 답변만 보고 역질문 3개 생성 → 원질문과 각 역질문의 임베딩 코사인 유사도 평균 | 질문과 대응하는가. 사실성 지표가 아니며 코사인은 −1~1 범위다. |
| Context Relevance | 질문에 필요한 검색 문장 ID 선택 → 선택 문장 수 / 검색 문맥의 전체 문장 수 | 필요한 내용에 집중되어 있는가. 검색 재현율이나 정답률과 다르다. |

분모 0, 실패한 판정, 잘린 출력, JSON 형식 오류, 존재하지 않는 문장 ID, 중복 주장 등은 해당 점수를 결측으로 둔다. 유효한 필요 문장 선택 결과가 빈 집합이고 `insufficient=true`이면 Context Relevance 0은 정의된 값이다. 판정 실패를 0으로 바꾸지 않는다.

## 원 논문과 다른 점

- 논문은 `gpt-3.5-turbo-16k`와 `text-embedding-ada-002`를 사용했다. 이번 검증은 로컬 Qwen2.5-0.5B-Instruct Q4_K_M과 nomic-embed-text-v1.5 Q4_K_M을 사용했다. 모델·언어·양자화·지시문 차이 때문에 논문 수치와 직접 비교할 수 없다.
- 사용자 자료의 한국어 문맥을 평가하고, 서버의 JSON Schema 출력 강제와 별도 구조 검증을 적용한다. 문장 선택은 원문 재작성 대신 번호 선택을 사용한다.
- 문장은 각 검색 청크 안에서 문장부호 뒤 공백 또는 줄바꿈으로 나눈다. 청크 끝의 불완전한 문장도 한 단위로 세며, 겹침 청크의 중복 문장을 제거하지 않는다. 분모와 실제 문장 목록을 CSV에 보존한다.
- 원 논문의 Answer Relevance 수식에는 명시적인 거절 패널티가 없다. 이 구현은 `noncommittal` 판정을 추가로 보관하지만, 거절로 판정된 답변의 기본 점수는 **결측·검토 대기**로 둔다. 원래 코사인 평균은 `answer_relevance_raw_cosine`에 남긴다. 작은 모델이 정상 답변을 거절로 오판할 수 있어 임의의 0점을 확정하지 않는다.
- 비어 있지 않은 답변에서 주장 목록이 비면 Faithfulness를 계산하지 않고 의미 검토 경고를 붙인다. 빈 주장 목록을 100%로 계산하지 않는다.
- 최초 실행 기록의 `noncommittal_policy`는 당시 설정을 그대로 보존했다. 내보내기 시의 보수적인 결측 처리 방침과 코드 해시는 `ragas_manifest.json`의 `reporting_policy`, `reporting_code_hash`에 따로 기록한다.

## 이번 실제 검증

원래 `results/validation.sqlite`의 실제 완료 RAG 답변 중 **같은 원문·질문·반복 번호·연구 실행 ID**인 원문 저장소와 P2 H1 요약 저장소를 짝지었다. 2쌍, 답변 4개가 대상이다. 데이터 종류는 `supplied_document_excerpt_smoke`이며 본 연구 코퍼스의 대규모 검증이 아니다.

| 항목 | 실제 기록 |
|---|---:|
| 평가 대상 답변 | 4 |
| 실제 판정 모델 호출 | 13 |
| 평가용 임베딩 호출 | 4 |
| 주장 없음에 따른 결정적 후처리 | 2 |
| 성공한 평가 단계 | 14 |
| 실패한 평가 단계 | 5 |
| 선행 단계 실패로 미실행인 단계 | 1 |
| 세 지표 평가 절차가 모두 완료된 답변 | 0 |
| 추가된 요약·RAG 생성 시행 | 0 |

JSON Schema 지원 확인을 위한 별도 사전 점검 호출 1회는 위 13회에 포함하지 않는다. 평가 단계 20개는 `주장 분해·주장 지지 판정·역질문 생성·임베딩 유사도·문장 선택` × 답변 4개다. 판정 모델 호출과 요약·답변 생성 시행을 혼합 집계하지 않는다.

실패는 문장 선택 응답 잘림 4개와 중복 주장 목록 1개다. 추가로, 사실을 포함한 답변에서 빈 주장 목록을 생성하거나 정상 답변을 거절로 판정하는 의미 오류를 관찰했다. 따라서 이 판정 모델의 수치는 **신뢰할 수 있는 본 연구 성능 결과로 제시하지 않는다**. 원문 저장소와 요약 저장소 중 어느 쪽이 우수하다는 결론을 내리지 않는다. CSV에 남은 숫자도 자동 판정의 검토 대상이다. 포스터에는 “RAGAS 참고 3축 평가 구현 / 판정 모델 검증 필요”로 표시한다.

## 실행

프로젝트 폴더에서 실행한다. 모델·임베딩 서버가 준비되어 있어야 한다. 외부 API는 이번 검증에서 호출하지 않았다.

```sh
# 새 평가 계획: 원문 저장소와 P2 H1의 짝 2개
python3 ragas_eval.py init --source-db results/validation.sqlite --db results/ragas_new.sqlite --pairs 2

# 실제 평가 / 실패 기록을 유지하며 이어서 실행
python3 ragas_eval.py run --db results/ragas_new.sqlite

# 상태 확인 및 CSV 내보내기
python3 ragas_eval.py status --db results/ragas_new.sqlite
python3 ragas_eval.py export --db results/ragas_new.sqlite --output results/ragas_new_csv
```

`--pairs 0`이면 원본 DB에서 실제 모델로 완료된 모든 RAG 답변을 대상으로 한다. `run --limit 5`는 이번에 시도할 평가 단계 수를 제한한다. 각 단계는 최대 2번 시도할 수 있다. 실패는 같은 실행에서 즉시 반복하지 않으며, 다음 `run`에서 이어진다. 실행이 중단되면 이전 시도는 `interrupted`로 보존된다. 운영체제 파일 잠금으로 같은 평가 DB의 동시 실행을 막는다. 완료된 단계를 재실행하거나 중복 합산하지 않는다.

기존 검증 기록은 `results/ragas_validation.sqlite`, CSV는 `results/ragas_csv/`에 있다. 이 DB를 초기화하지 않고 그대로 보존한다. 더 강한 판정 모델·다른 지시문·변경된 코드로 새 검증을 할 때는 새 평가 DB를 만든다.

### 판정 모델 바꾸기

`init`에 JSON 설정 파일을 지정한다. 파일 내용은 기존 `ysc.Backend`와 `ysc.Embedder` 인터페이스를 따른다. 실제 모델명과 실제 리비전을 기록하고, API 키는 환경변수에 둔다.

```sh
python3 ragas_eval.py init --source-db results/validation.sqlite --db results/ragas_stronger.sqlite --pairs 2 --judge-config configs/my_judge.json --embedding-config configs/my_embedding.json
```

판정 설정에는 `kind: openai_compatible`, `base_url`, `model`, `revision`, `supports_seed`를 넣는다. API가 키를 요구하면 `api_key_env`에 환경변수 이름을 추가한다. 서버는 `/chat/completions`의 JSON Schema `response_format`을 지원해야 한다. 임베딩 설정은 `openai_compatible` 또는 `ollama`와 학습된 임베딩 모델을 사용한다. 질문 비교에서는 원질문·역질문 모두 같은 `query_prefix`를 적용한다. 임베딩 모델을 바꾸면 유사도 척도도 달라지므로 이전 실행과 직접 합치지 않는다.

## 저장 파일과 검증

- `ragas_metrics.csv`: 답변별 세 지표, 각 지표 상태, 원래 코사인 평균, 원문·질문·시드·저장소·프롬프트·단계·모델·리비전·토큰·평가 근거·문장 목록·의미 검토 경고.
- `ragas_evaluation_attempts.csv`: 성공·실패·중단을 포함한 모든 평가 시도. 실제 사용자 지시문, JSON Schema, 시드, 요청 설정, 모델의 원본 응답, 토큰 사용량, 오류와 시간이 남는다. 서버가 반환한 모델명과 시스템 fingerprint도 원본 응답에서 확인한다.
- `ragas_summary.csv`: 조건별 각 지표의 유효 개수와 평균. 결측값은 분모에 넣지 않는다. 자동 판정의 신뢰도는 별도 검토가 필요하다.
- `ragas_manifest.json`: 고정된 입력 ID·해시·모델 설정·지시문·스키마·집계 정책. 판정 시스템 지시문도 기록한다.

```sh
python3 -m unittest discover -s tests -p test_ragas_eval.py -v
```

전용 테스트 8개가 수식, 0과 결측의 구분, 잘못된 판정 파싱, 중복 검색 문장 분모, 재시작, 실패 응답 보존, 원본 DB 보존, 설정 변경 차단을 확인한다. 테스트의 임시 모의 응답은 `simulation_unit_test`로 구분하며 연구 결과에 포함하지 않는다.

최종 연구에서는 충분한 성능의 별도 판정 모델과 사람이 검토한 표본으로 일치도를 확인한 뒤, 그 판정기·설정·프롬프트를 고정하여 전체 평가를 수행해야 한다. 답변 생성기와 판정기가 같은 작은 모델이면 상관된 오류가 생길 수 있고, RAGAS 참고 점수 자체가 정답의 증명은 아니다.
