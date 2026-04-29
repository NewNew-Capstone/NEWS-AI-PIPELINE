# Emotion A/B Evaluation

## Dataset Format (`jsonl`)

한 줄에 한 문장씩 아래 필드를 넣습니다.

- `content_sentence_id` (int, optional): 문장 ID. 없으면 자동 부여.
- `sentence_text` (str, required): 평가 문장.
- `gold_emotion` (bool/int/str, required): 감정 표현 정답 라벨.
  - 허용값 예시: `true/false`, `1/0`, `"yes"/"no"`

예시:

```json
{"content_sentence_id": 101, "sentence_text": "이번 정책은 최악이다.", "gold_emotion": true}
{"content_sentence_id": 102, "sentence_text": "국회는 오늘 예산안을 처리했다.", "gold_emotion": false}
```

## Run

```bash
.venv/bin/python scripts/eval_emotion_ab.py \
  --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl
```

운영 로그에서 평가셋 생성(신규):

```bash
.venv/bin/python scripts/build_emotion_eval_from_logs.py \
  --input logs/analyze_requests.jsonl \
  --output analysis_bc/data/eval/emotion_eval_operational_sample.jsonl \
  --sample-size 400 \
  --recent-days 7
```

DB에서 운영 로그 JSONL export:

```bash
.venv/bin/python scripts/export_analysis_request_logs.py \
  --output logs/analyze_requests.jsonl \
  --days 7 \
  --limit 5000
```

DB 대신 로컬 JSONL 로그를 바로 사용할 수도 있습니다.

1) `.env`에 추가:
```env
ANALYSIS_REQUEST_LOG_LOCAL_PATH=logs/analysis_requests_local.jsonl
ANALYSIS_REQUEST_LOG_LOCAL_DAILY_SPLIT=true
ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS=30
```
2) API(`/analyze`, `/analyze/raw`) 호출 시 로컬 파일에 자동 누적
   - 기본값 기준 일자별 파일로 저장됩니다. 예: `logs/analysis_requests_local-20260425.jsonl`
   - 보존 기간(`ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS`) 지난 일자 파일은 자동 정리됩니다.
3) 평가셋 생성 입력을 로컬 파일로 지정:
```bash
.venv/bin/python scripts/build_emotion_eval_from_logs.py \
  --input logs/analysis_requests_local.jsonl \
  --output analysis_bc/data/eval/emotion_eval_operational_sample.jsonl \
  --sample-size 400 \
  --recent-days 7
```

임시 자동 라벨 채움(수작업 교정 전제):

```bash
.venv/bin/python scripts/build_emotion_eval_from_logs.py \
  --input logs/analyze_requests.jsonl \
  --output analysis_bc/data/eval/emotion_eval_operational_sample.jsonl \
  --sample-size 400 \
  --recent-days 7 \
  --autolabel
```

중복 문장 압축 + 오차 로그 저장:

```bash
.venv/bin/python scripts/eval_emotion_ab.py \
  --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl \
  --dedupe-by-text \
  --export-errors analysis_bc/data/eval/reports/error_cases.jsonl
```

패턴 확정 리포트까지 생성:

```bash
.venv/bin/python scripts/eval_emotion_ab.py \
  --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl \
  --confirm-patterns
```

`--confirm-patterns`는 항상 2단계로 실행됩니다.
- 1단계: raw(비 dedupe)
- 2단계: deduped(중복 sentence_text 압축)

생성 파일:
- `analysis_bc/data/eval/reports/error_cases_raw.jsonl`
- `analysis_bc/data/eval/reports/error_cases_deduped.jsonl`
- `analysis_bc/data/eval/reports/pattern_report_raw.json`
- `analysis_bc/data/eval/reports/pattern_report_deduped.json`
- `analysis_bc/data/eval/reports/eval_summary.json`
- `analysis_bc/data/eval/reports/metrics_history.jsonl`

기준선 비교(승인 게이트):

```bash
.venv/bin/python scripts/eval_emotion_ab.py \
  --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl \
  --confirm-patterns \
  --baseline-summary analysis_bc/data/eval/reports/baseline_eval_summary.json \
  --change-reason \"fn boost trial #1\"
```

승인 게이트 기준:
- FP non-increase
- precision 하락 `<= 0.02`
- f1 non-decrease
- recall 하락 `<= 0.02`

## 운영 루프

1. 운영 로그 샘플링: `build_emotion_eval_from_logs.py`로 최근 7일 300~500문장 추출
   - DB 기반일 경우 `export_analysis_request_logs.py`로 먼저 JSONL export
2. 라벨 보정: `gold_emotion`을 수작업 교정(자동 라벨 사용 시 필수)
3. 기준선 생성: `--confirm-patterns` 실행 후 `eval_summary.json`를 `baseline_eval_summary.json`로 보관
4. 소규모 반영(1~3개): 확정 FN 패턴만 `init_qdrant.py`에 반영 후 재인덱싱
5. 재평가: `--baseline-summary`와 함께 실행해 승인 게이트 확인
6. 통과 시 유지, 미통과 시 마지막 반영 롤백

그리드서치 포함:

```bash
.venv/bin/python scripts/eval_emotion_ab.py \
  --dataset analysis_bc/data/eval/emotion_eval_sample.jsonl \
  --run-grid
```
