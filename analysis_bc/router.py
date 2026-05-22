from functools import lru_cache
import logging
import threading
import time

from fastapi import APIRouter

from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.request_log_repository import AnalysisRequestLogRepository
from analysis_bc.schemas import (
    AnalyzeRequestDto,
    AnalyzeRawTextRequestDto,
    BiasAnalysisResultDto,
    RawAnalysisResultDto,
    ScoreReasonRequestDto,
    ScoreReasonResponseDto,
    SentenceResultDto,
    SummaryRequestDto,
    SummaryResponseDto,
)
from analysis_bc.service import AnalysisService

router = APIRouter(prefix="/analyze", tags=["analysis"])
logger = logging.getLogger(__name__)

# 배치 분석은 1개씩만 실행, priority 요청은 바로 통과
_BG_ANALYSIS_SEMAPHORE = threading.Semaphore(1)
# priority 분석이 실행 중일 때 True — 배치 분석은 이 플래그가 내려갈 때까지 대기
_PRIORITY_ACTIVE = threading.Event()
_request_log_repo = AnalysisRequestLogRepository()


@lru_cache
def get_analysis_service() -> AnalysisService:
    """모델 로딩이 무거운 AnalysisService를 프로세스 내 1회만 초기화."""
    return AnalysisService()


def _safe_log_request(
    *,
    source_endpoint: str,
    target_id: int | None,
    transcript_id: int | None,
    language: str | None,
    target_type: str | None,
    country: str | None,
    sentences: list[str],
) -> None:
    try:
        _request_log_repo.insert_request_log(
            source_endpoint=source_endpoint,
            target_id=target_id,
            transcript_id=transcript_id,
            language=language,
            target_type=target_type,
            country=country,
            sentences=sentences,
        )
    except Exception:
        logger.warning("analysis request log insert failed", exc_info=True)


@router.post("", response_model=BiasAnalysisResultDto)
def analyze(request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
    print("=== 받은 요청 ===")
    print(request.model_dump())
    print("=================")
    _safe_log_request(
        source_endpoint="/analyze",
        target_id=request.target_id,
        transcript_id=request.transcript_id,
        language=request.language,
        target_type=str(request.target_type) if request.target_type is not None else None,
        country=request.country,
        sentences=[s.sentence_text for s in request.sentences],
    )
    return get_analysis_service().analyze(request)


@router.post("/raw", response_model=RawAnalysisResultDto)
def analyze_raw(request: AnalyzeRawTextRequestDto) -> RawAnalysisResultDto:
    print("=== [/analyze/raw] 받은 요청 ===")
    print(
        f"target_id={request.target_id}, transcript_id={request.transcript_id}, "
        f"title={request.title}, language={request.language}"
    )
    print(f"raw_text 길이={len(request.raw_text)}, 앞 100자: {request.raw_text[:100]}")
    sentences = split_into_sentences(request.raw_text, request.language)
    print(f"분리된 문장 수: {len(sentences)}")
    for s in sentences[:5]:
        print(f"  [{s.content_sentence_id}] {s.sentence_text[:60]}")
    print("================================")
    _safe_log_request(
        source_endpoint="/analyze/raw",
        target_id=request.target_id,
        transcript_id=request.transcript_id,
        language=request.language,
        target_type=str(request.target_type) if request.target_type is not None else None,
        country=request.country,
        sentences=[s.sentence_text for s in sentences],
    )
    analyze_request = AnalyzeRequestDto(
        target_id=request.target_id,
        title=request.title,
        language=request.language,
        target_type=request.target_type,
        transcript_id=request.transcript_id,
        country=request.country,
        sentences=sentences,
        priority=request.priority,
    )
    if request.priority:
        # priority: 세마포어 없이 즉시 실행 + 플래그 ON → 신규 배치 분석 블로킹
        logger.info("[분석] priority 요청 — 즉시 실행, 신규 배치 분석 대기 처리")
        _PRIORITY_ACTIVE.set()
        try:
            result = get_analysis_service().analyze(analyze_request)
        finally:
            _PRIORITY_ACTIVE.clear()
            logger.info("[분석] priority 완료 — 배치 분석 재개 허용")
    else:
        # 배치: priority 분석 중이면 완료될 때까지 대기 후 세마포어 획득
        wait_count = 0
        while _PRIORITY_ACTIVE.is_set():
            if wait_count == 0:
                logger.info("[분석] 배치 대기 — priority 분석 완료 후 시작 예정")
            time.sleep(1)
            wait_count += 1

        logger.info("[분석] 배치 요청 — BG_ANALYSIS_SEMAPHORE 대기 중")
        with _BG_ANALYSIS_SEMAPHORE:
            # 세마포어 획득 후 priority가 새로 들어왔으면 다시 대기
            while _PRIORITY_ACTIVE.is_set():
                logger.info("[분석] 배치 세마포어 보유 중 priority 감지 — 완료 대기")
                time.sleep(1)
            logger.info("[분석] 배치 요청 — 세마포어 획득, 분석 시작")
            result = get_analysis_service().analyze(analyze_request)
        logger.info("[분석] 배치 요청 — 세마포어 반환 완료")
    sentence_results = [
        SentenceResultDto(
            content_sentence_id=s.content_sentence_id,
            sentence_text=s.sentence_text,
            sentence_order=s.sentence_order,
        )
        for s in sentences
    ]
    return RawAnalysisResultDto(**result.model_dump(), sentences=sentence_results)


@router.post("/score-reason", response_model=ScoreReasonResponseDto)
def summarize_score_reason(request: ScoreReasonRequestDto) -> ScoreReasonResponseDto:
    print("=== [/analyze/score-reason] 받은 요청 ===")
    print(
        f"target_id={request.target_id}, language={request.language}, "
        f"overall={request.overall_bias_score:.4f}, opinion={request.opinion_score:.4f}, "
        f"emotion={request.emotion_score:.4f}"
    )
    print("======================================")
    summary = get_analysis_service().summarize_score_reason_only(request)
    return ScoreReasonResponseDto(score_reason_summary=summary)


@router.post("/summary", response_model=SummaryResponseDto)
def summarize_text(request: SummaryRequestDto) -> SummaryResponseDto:
    print("=== [/analyze/summary] 받은 요청 ===")
    print(
        f"target_id={request.target_id}, title={request.title}, "
        f"language={request.language}, raw_text_len={len(request.raw_text)}"
    )
    print("===================================")
    summary_text = get_analysis_service().summarize_text_only(request)
    return SummaryResponseDto(summary_text=summary_text)
