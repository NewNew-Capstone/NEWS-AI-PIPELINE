"""익명출처/추측 표현 마이닝 스크립트.

AI Hub 원천데이터(mrc_news/*.json)에서 OPINION 문장을 추출하고
Claude API로 익명출처/추측 표현을 검증하여 anonymous_patterns.json을 생성한다.

실행:
    python scripts/mine_anonymous_patterns.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock
from kiwipiepy import Kiwi  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.classifier import FactOpinionClassifier
from analysis_bc.config import ANTHROPIC_API_KEY

DATA_DIR = "analysis_bc/data/mrc_news"
OUTPUT_PATH = "analysis_bc/data/anonymous_patterns.json"
MIN_SENTENCE_LEN = 10
MIN_PHRASE_FREQ = 10
TOP_N_CANDIDATES = 500
CLAUDE_BATCH_SIZE = 50
CLAUDE_MODEL = "claude-opus-4-6"


def load_contexts(data_dir: str) -> list[str]:
    contexts: list[str] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f, strict=False)
        except json.JSONDecodeError as e:
            print(f"  [경고] {path.name} 파싱 실패 — 스킵 ({e})", flush=True)
            continue
        for doc in data.get("data", []):
            for para in doc.get("paragraphs", []):
                ctx = para.get("context", "").strip()
                if ctx:
                    contexts.append(ctx)
    return contexts


def split_sentences(context: str, kiwi: Kiwi) -> list[str]:
    results = kiwi.split_into_sents(context)
    return [s.text for s in results if len(s.text) >= MIN_SENTENCE_LEN]


def filter_opinion_sentences(
    sentences: list[str],
    classifier: FactOpinionClassifier,
) -> list[str]:
    opinions: list[str] = []
    for sentence in sentences:
        label, confidence = classifier.predict(sentence)
        if label == "opinion_like" and confidence >= 0.7:
            opinions.append(sentence)
    return opinions


def extract_phrases(sentence: str) -> list[str]:
    words = sentence.split()
    phrases: list[str] = []
    for window_size in range(2, 5):
        for i in range(len(words) - window_size + 1):
            phrases.append(" ".join(words[i : i + window_size]))
    return phrases


def verify_with_claude(phrases: list[str], client: Anthropic) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    total_batches = (len(phrases) + CLAUDE_BATCH_SIZE - 1) // CLAUDE_BATCH_SIZE

    for batch_idx, batch_start in enumerate(range(0, len(phrases), CLAUDE_BATCH_SIZE)):
        batch = phrases[batch_start : batch_start + CLAUDE_BATCH_SIZE]
        phrase_list = "\n".join(f"- {p}" for p in batch)

        prompt = f"""아래 한국어 구(句) 목록에서 뉴스 기사의 익명출처 표현이나 추측성 표현만 골라줘.

익명출처 표현 기준 (ANONYMOUS_SOURCE):
- 출처를 명확히 밝히지 않은 표현
- 예) "관계자에 따르면", "소식통에 의하면", "전문가들은", "일각에서는"

추측성 표현 기준 (SPECULATIVE):
- 사실이 아닌 추측이나 전망을 나타내는 표현
- 예) "것으로 알려졌다", "전망이다", "것으로 예상된다", "카더라"

구 목록:
{phrase_list}

아래 JSON 형식으로만 응답해줘.
익명출처/추측 표현이 아닌 것은 반드시 제외해줘:
[
  {{"phrase": "표현", "label_type": "ANONYMOUS_SOURCE 또는 SPECULATIVE"}},
  ...
]"""

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            block = response.content[0]
            if not isinstance(block, TextBlock):
                continue
            text = block.text.strip()
            # JSON 배열 부분만 추출
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                batch_results: list[dict[str, str]] = json.loads(text[start:end])
                results.extend(batch_results)
        except (json.JSONDecodeError, IndexError, Exception):
            pass

        print(
            f"  배치 {batch_idx + 1}/{total_batches} 완료",
            flush=True,
        )

    return results


def main() -> None:
    print("[1/4] 원천데이터 로딩 중...", flush=True)
    contexts = load_contexts(DATA_DIR)
    print(f"  context 수: {len(contexts)}개", flush=True)

    print("[2/4] 문장 분리 + OPINION 분류 중...", flush=True)
    kiwi = Kiwi()
    classifier = FactOpinionClassifier()

    all_sentences: list[str] = []
    for ctx in contexts:
        all_sentences.extend(split_sentences(ctx, kiwi))
    print(f"  전체 문장: {len(all_sentences)}개", flush=True)

    opinion_sentences = filter_opinion_sentences(all_sentences, classifier)
    print(f"  OPINION 문장: {len(opinion_sentences)}개", flush=True)

    print("[3/4] 구(句) 빈도 계산 중...", flush=True)
    counter: Counter[str] = Counter()
    for sentence in opinion_sentences:
        for phrase in extract_phrases(sentence):
            counter[phrase] += 1

    candidates = [
        phrase
        for phrase, freq in counter.most_common(TOP_N_CANDIDATES)
        if freq >= MIN_PHRASE_FREQ
    ]
    print(f"  후보 구: {len(candidates)}개", flush=True)

    print("[4/4] Claude API 검증 중...", flush=True)
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    raw_results = verify_with_claude(candidates, anthropic_client)

    # 중복 제거 (phrase 기준)
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in raw_results:
        phrase = item.get("phrase", "").strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            deduped.append({"phrase": phrase, "label_type": item.get("label_type", "")})

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUTPUT_PATH} ({len(deduped)}개 표현)", flush=True)


if __name__ == "__main__":
    main()
