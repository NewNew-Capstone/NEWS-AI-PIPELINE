"""A/B 극단 기사로 편향 점수 가중치를 튜닝하는 스크립트.

실행 예시:
    python scripts/tune_bias_weights_ab.py full
    python scripts/tune_bias_weights_ab.py snapshot
    python scripts/tune_bias_weights_ab.py snapshot --grid-divisions 20
    python scripts/tune_bias_weights_ab.py full --limit-per-dataset 5 --snapshot-path /tmp/ab_smoke.json

`full`은 실제 A/B 원문을 서비스와 같은 점수 계산 전 파이프라인에 태운 뒤,
재사용 가능한 중간 결과를 snapshot 파일로 저장한다.
`snapshot`은 저장된 중간 결과를 다시 읽어 가중치 조합별 scorer 결과만
빠르게 다시 계산한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "analysis_bc" / "data"
SNAPSHOT_PATH = DATA_DIR / "eval" / "ab_pipeline_snapshot.json"


# 가중치 후보 하나를 표현한다. ScorerWeights는 실행 시점에 import해서
# --help처럼 가벼운 명령은 torch 같은 ML 의존성 없이도 실행되게 한다.
@dataclass(frozen=True)
class WeightVariant:
    name: str
    weights: Any


# 한 기사와 한 가중치 조합을 scorer에 넣어 얻은 결과 행이다.
@dataclass(frozen=True)
class RunResult:
    dataset: str
    target_id: int
    title: str
    variant: str
    overall: float
    opinion: float
    emotion: float
    fact_ratio: float
    headline_body_gap: float
    sentence_count: int


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _gap_score_value(gap: Any) -> float:
    """TitleBodyGapCalculator 반환값 호환 처리.

    과거에는 float, 현재는 GapResult 객체를 반환할 수 있다.
    """
    if isinstance(gap, (int, float)):
        return float(gap)
    if hasattr(gap, "gap_score"):
        return float(gap.gap_score)
    raise TypeError(f"Unsupported gap value type: {type(gap)!r}")


def _load_articles(limit_per_dataset: int | None = None) -> list[dict[str, Any]]:
    # A는 "편향 점수가 높아야 하는" 사설/의견성 극단 케이스,
    # B는 "편향 점수가 낮아야 하는" 정책/공적 자막 극단 케이스로 본다.
    articles: list[dict[str, Any]] = []
    for dataset, filename in (("A", "A.json"), ("B", "B.json")):
        dataset_articles = _load_json(DATA_DIR / filename)
        if limit_per_dataset is not None:
            dataset_articles = dataset_articles[:limit_per_dataset]
        for article in dataset_articles:
            articles.append({"dataset": dataset, **article})
    return articles


def _ensure_qdrant_ready(span_tagger: Any) -> None:
    """Fail fast so full snapshots are not silently built with emotion=0."""
    if span_tagger._is_qdrant_healthy():
        return

    raise RuntimeError(
        "Qdrant health check failed. `full` mode would create a snapshot with "
        "empty emotion spans, so it was stopped. Start Qdrant and initialize "
        "`emotion_words` first: `docker compose up -d qdrant` then "
        "`.venv/bin/python scripts/init_qdrant.py`."
    )


def _record_emotion_score(record: dict[str, Any]) -> float:
    from analysis_bc.classifier import ClassifiedSentenceDto
    from analysis_bc.schemas import SpanLabelDto
    from analysis_bc.scorer import BiasScorer

    classified = [ClassifiedSentenceDto(**item) for item in record["classified"]]
    span_labels = [SpanLabelDto(**item) for item in record["span_labels"]]
    return BiasScorer().calculate(
        classified=classified,
        span_labels=span_labels,
        headline_body_gap=record["headline_body_gap"],
    )["emotion_score"]


def _print_emotion_diagnostics(snapshot: dict[str, Any]) -> None:
    records = snapshot.get("records", [])
    total_spans = sum(len(record.get("span_labels", [])) for record in records)

    print("\nemotion diagnostics")
    print("-" * 72)
    print(f"records={len(records)} total_spans={total_spans}")

    for dataset in ("A", "B"):
        dataset_records = [
            record for record in records
            if record.get("dataset") == dataset
        ]
        if not dataset_records:
            continue

        span_count = sum(
            len(record.get("span_labels", []))
            for record in dataset_records
        )
        avg_emotion = sum(
            _record_emotion_score(record)
            for record in dataset_records
        ) / len(dataset_records)
        print(
            f"{dataset}: records={len(dataset_records)} "
            f"spans={span_count} avg_emotion_score={avg_emotion:.4f}"
        )


def _build_snapshot(
    *,
    snapshot_path: Path = SNAPSHOT_PATH,
    limit_per_dataset: int | None = None,
) -> dict[str, Any]:
    """비용이 큰 점수 계산 전 파이프라인을 한 번 실행하고 중간 결과를 저장한다."""
    # full 모드는 실제 서비스와 같은 계산 경로를 탄다.
    # 모델/Qdrant/임베딩 로딩 비용이 크므로, 결과를 snapshot으로 남겨
    # 이후 가중치 실험에서는 같은 중간 결과를 재사용한다.
    from analysis_bc.classifier import FactOpinionClassifier
    from analysis_bc.preprocessor import SentencePreprocessor, split_into_sentences
    from analysis_bc.tagger.span_tagger import SpanTagger
    from analysis_bc.tagger.title_body_gap import TitleBodyGapCalculator

    classifier = FactOpinionClassifier(model_path="analysis_bc/models/best_model")
    span_tagger = SpanTagger()
    _ensure_qdrant_ready(span_tagger)
    gap_calculator = TitleBodyGapCalculator()

    records: list[dict[str, Any]] = []
    for article in _load_articles(limit_per_dataset):
        language = article["language"]

        # 1. raw_text를 서비스의 /analyze/raw 경로처럼 문장 단위로 나눈다.
        raw_sentences = split_into_sentences(article["raw_text"], language)

        # 2. 실제 AnalysisService.prepare_sentences()와 같은 전처리를 적용한다.
        sentences = SentencePreprocessor(expected_language=language).preprocess(raw_sentences)

        # 3. classifier로 fact_like / opinion_like를 분리한다.
        classified = classifier.classify(sentences)
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]

        # 4. 서비스와 동일하게 opinion 문장에 대해서만 감정 span을 태깅한다.
        span_labels = span_tagger.tag(opinion_sentences)

        # 5. 제목-본문 gap을 계산한다. 현재 scorer 최종 산식에는 직접 들어가지 않지만,
        #    결과 응답과 score_evidence 생성에 쓰이므로 snapshot에 함께 저장한다.
        headline_body_gap_raw = gap_calculator.calculate(
            title=article["title"],
            sentences=sentences,
        )
        headline_body_gap = _gap_score_value(headline_body_gap_raw)

        # 6. 가중치가 바뀌어도 변하지 않는 중간 결과만 저장한다.
        #    이후 snapshot 모드에서는 이 값들로 scorer만 다시 돌린다.
        records.append({
            "dataset": article["dataset"],
            "target_id": article["target_id"],
            "title": article.get("title", ""),
            "language": language,
            "sentence_count": len(sentences),
            "headline_body_gap": headline_body_gap,
            "classified": [s.model_dump(mode="json") for s in classified],
            "span_labels": [s.model_dump(mode="json") for s in span_labels],
        })

        print(
            f"[full] {article['dataset']}#{article['target_id']} "
            f"sentences={len(sentences)} opinion={len(opinion_sentences)} "
            f"spans={len(span_labels)} gap={headline_body_gap:.4f}"
        )

    snapshot = {
        "source": {
            "a_path": str(DATA_DIR / "A.json"),
            "b_path": str(DATA_DIR / "B.json"),
            "limit_per_dataset": limit_per_dataset,
        },
        "records": records,
    }
    _write_json(snapshot_path, snapshot)
    print(f"\n[snapshot] wrote {snapshot_path}")
    _print_emotion_diagnostics(snapshot)
    return snapshot


def _load_snapshot(snapshot_path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    # snapshot 모드는 full 모드가 저장한 중간 결과를 전제로 한다.
    # 파일이 없으면 먼저 full을 실행해야 한다.
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot not found: {snapshot_path}\n"
            "먼저 `python scripts/tune_bias_weights_ab.py full`을 실행해 주세요."
        )
    return _load_json(snapshot_path)


def _default_variants() -> list[WeightVariant]:
    # 빠르게 눈으로 비교하기 위한 대표 후보들이다.
    # 모든 조합은 w_opinion + w_emotion + w_fact = 1.0이어야 한다.
    from analysis_bc.scorer import ScorerWeights

    return [
        WeightVariant("default_40_30_30", ScorerWeights()),
        WeightVariant("opinion_heavy_50_25_25", ScorerWeights(0.50, 0.25, 0.25)),
        WeightVariant("emotion_heavy_35_45_20", ScorerWeights(0.35, 0.45, 0.20)),
        WeightVariant("fact_guard_35_25_40", ScorerWeights(0.35, 0.25, 0.40)),
    ]


def _grid_variants(divisions: int) -> Iterable[WeightVariant]:
    # grid 탐색은 세 가중치가 0 이상이고 합이 1인 모든 조합을 만든다.
    # divisions=10이면 0.0, 0.1, ..., 1.0 단위로 탐색한다.
    from analysis_bc.scorer import ScorerWeights

    if divisions <= 0:
        raise ValueError("--grid-divisions must be positive")
    for opinion_units in range(divisions + 1):
        for emotion_units in range(divisions + 1 - opinion_units):
            fact_units = divisions - opinion_units - emotion_units
            w_opinion = opinion_units / divisions
            w_emotion = emotion_units / divisions
            w_fact = fact_units / divisions
            name = f"grid_{w_opinion:.2f}_{w_emotion:.2f}_{w_fact:.2f}"
            yield WeightVariant(name, ScorerWeights(w_opinion, w_emotion, w_fact))


def _variants(args: argparse.Namespace) -> list[WeightVariant]:
    if args.grid:
        return list(_grid_variants(args.grid_divisions))
    return _default_variants()


def _score_record(record: dict[str, Any], variant: WeightVariant) -> RunResult:
    # snapshot에 저장된 dict를 원래 DTO로 복원한 뒤 BiasScorer에 넣는다.
    # 여기서는 classifier/tagger를 다시 호출하지 않고 가중치 산식만 바뀐다.
    from analysis_bc.classifier import ClassifiedSentenceDto
    from analysis_bc.schemas import SpanLabelDto
    from analysis_bc.scorer import BiasScorer

    classified = [ClassifiedSentenceDto(**item) for item in record["classified"]]
    span_labels = [SpanLabelDto(**item) for item in record["span_labels"]]
    scores = BiasScorer(variant.weights).calculate(
        classified=classified,
        span_labels=span_labels,
        headline_body_gap=record["headline_body_gap"],
    )
    return RunResult(
        dataset=record["dataset"],
        target_id=record["target_id"],
        title=record["title"][:24],
        variant=variant.name,
        overall=scores["overall_bias_score"],
        opinion=scores["opinion_score"],
        emotion=scores["emotion_score"],
        fact_ratio=scores["fact_ratio"],
        headline_body_gap=record["headline_body_gap"],
        sentence_count=record["sentence_count"],
    )


def _evaluate(snapshot: dict[str, Any], variants: list[WeightVariant]) -> list[RunResult]:
    # 모든 기사 x 모든 가중치 조합의 점수를 계산한다.
    return [
        _score_record(record, variant)
        for variant in variants
        for record in snapshot["records"]
    ]


def _print_detail(results: list[RunResult]) -> None:
    header = (
        f"{'가중치':<25} {'셋':<2} {'id':>4} {'최종':>7} "
        f"{'의견':>6} {'감정':>6} {'사실':>6} {'gap':>6} {'문장':>4} 제목"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.variant:<25} {r.dataset:<2} {r.target_id:>4} "
            f"{r.overall:>7.4f} {r.opinion:>6.4f} {r.emotion:>6.4f} "
            f"{r.fact_ratio:>6.4f} {r.headline_body_gap:>6.4f} "
            f"{r.sentence_count:>4} {r.title}"
        )


def _print_summary(results: list[RunResult]) -> None:
    print("\n요약")
    print("-" * 94)
    header = (
        f"{'가중치':<25} {'A평균':>7} {'B평균':>7} {'A최소':>7} "
        f"{'B최대':>7} {'마진':>7} {'손실':>9} {'분리':>3}"
    )
    print(header)
    print("-" * len(header))

    variants = dict.fromkeys(r.variant for r in results)
    rows: list[tuple[float, str]] = []
    for variant in variants:
        # A는 1에 가까울수록, B는 0에 가까울수록 좋다.
        # A_min - B_max가 양수이면 현재 snapshot 기준으로 두 그룹이 완전히 분리된다.
        a_scores = [r.overall for r in results if r.variant == variant and r.dataset == "A"]
        b_scores = [r.overall for r in results if r.variant == variant and r.dataset == "B"]
        if not a_scores or not b_scores:
            continue
        a_avg = sum(a_scores) / len(a_scores)
        b_avg = sum(b_scores) / len(b_scores)
        a_min = min(a_scores)
        b_max = max(b_scores)
        margin = a_min - b_max
        # 극단 테스트 목적의 단순 손실 함수:
        # A는 1에서 멀수록 벌점, B는 0에서 멀수록 벌점이다.
        loss = (
            sum((score - 1.0) ** 2 for score in a_scores) / len(a_scores)
            + sum(score ** 2 for score in b_scores) / len(b_scores)
        )
        ok = "예" if margin > 0 else "아니오"
        row = (
            f"{variant:<25} {a_avg:>7.4f} {b_avg:>7.4f} "
            f"{a_min:>7.4f} {b_max:>7.4f} {margin:>7.4f} "
            f"{loss:>9.6f} {ok:>3}"
        )
        rows.append((loss, row))

    for _, row in sorted(rows, key=lambda item: item[0]):
        print(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B 극단 케이스로 편향 점수 가중치를 튜닝합니다."
    )
    parser.add_argument(
        "mode",
        choices=("full", "snapshot"),
        help="full은 새 snapshot을 만들고, snapshot은 저장된 결과로 가중치만 다시 계산합니다.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="0 이상이고 합이 1인 모든 가중치 조합을 격자 단위로 평가합니다.",
    )
    parser.add_argument(
        "--grid-divisions",
        type=int,
        default=10,
        help="격자 해상도입니다. 10은 0.1 단위, 20은 0.05 단위입니다.",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="기사별 상세 표를 생략하고 요약 표만 출력합니다.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=SNAPSHOT_PATH,
        help="읽거나 쓸 snapshot 경로입니다. 기본값은 analysis_bc/data/eval/ab_pipeline_snapshot.json입니다.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="full 모드에서 A/B 각 데이터셋의 앞 N건만 실행합니다. 로컬 smoke test용입니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # full: 실제 파이프라인을 돌려 snapshot을 새로 만든다.
    # snapshot: 저장된 중간 결과를 읽어 scorer 가중치만 다시 계산한다.
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        raise ValueError("--limit-per-dataset must be positive")

    snapshot = (
        _build_snapshot(
            snapshot_path=args.snapshot_path,
            limit_per_dataset=args.limit_per_dataset,
        )
        if args.mode == "full"
        else _load_snapshot(args.snapshot_path)
    )
    if args.mode == "snapshot":
        _print_emotion_diagnostics(snapshot)

    # --grid가 있으면 촘촘한 모든 조합을, 없으면 대표 후보들만 비교한다.
    results = _evaluate(snapshot, _variants(args))

    # 상세 표는 기사별 진단용, 요약 표는 가중치 후보 선택용이다.
    if not args.no_detail:
        _print_detail(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
