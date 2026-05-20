from functools import lru_cache
import logging
import threading

from fastapi import APIRouter

from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.request_log_repository import AnalysisRequestLogRepository
from analysis_bc.schemas import AnalyzeRequestDto, AnalyzeRawTextRequestDto, BiasAnalysisResultDto, RawAnalysisResultDto, SentenceResultDto
from analysis_bc.service import AnalysisService

router = APIRouter(prefix="/analyze", tags=["analysis"])
logger = logging.getLogger(__name__)

# 배치 분석은 1개씩만 실행, priority 요청은 바로 통과
_BG_ANALYSIS_SEMAPHORE = threading.Semaphore(1)
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
        print("[분석] priority 요청 — 세마포어 없이 즉시 실행")
        result = get_analysis_service().analyze(analyze_request)
    else:
        print("[분석] 배치 요청 — BG_ANALYSIS_SEMAPHORE 대기 중")
        with _BG_ANALYSIS_SEMAPHORE:
            print("[분석] 배치 요청 — 세마포어 획득, 분석 시작")
            result = get_analysis_service().analyze(analyze_request)
        print("[분석] 배치 요청 — 세마포어 반환 완료")
    sentence_results = [
        SentenceResultDto(
            content_sentence_id=s.content_sentence_id,
            sentence_text=s.sentence_text,
            sentence_order=s.sentence_order,
        )
        for s in sentences
    ]
    return RawAnalysisResultDto(**result.model_dump(), sentences=sentence_results)
