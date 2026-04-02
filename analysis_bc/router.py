from fastapi import APIRouter, Request
import json

from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.schemas import AnalyzeRequestDto, AnalyzeRawTextRequestDto, BiasAnalysisResultDto, RawAnalysisResultDto, SentenceResultDto
from analysis_bc.service import AnalysisService

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("", response_model=BiasAnalysisResultDto)
def analyze(request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
    print("=== 받은 요청 ===")
    print(request.model_dump())
    print("=================")
    return AnalysisService().analyze(request)


@router.post("/raw", response_model=RawAnalysisResultDto)
def analyze_raw(request: AnalyzeRawTextRequestDto) -> RawAnalysisResultDto:
    print("=== [/analyze/raw] 받은 요청 ===")
    print(f"target_id={request.target_id}, title={request.title}, language={request.language}")
    print(f"raw_text 길이={len(request.raw_text)}, 앞 100자: {request.raw_text[:100]}")
    sentences = split_into_sentences(request.raw_text, request.language)
    print(f"분리된 문장 수: {len(sentences)}")
    for s in sentences[:5]:
        print(f"  [{s.content_sentence_id}] {s.sentence_text[:60]}")
    print("================================")
    analyze_request = AnalyzeRequestDto(
        target_id=request.target_id,
        title=request.title,
        language=request.language,
        target_type=request.target_type,
        transcript_id=request.transcript_id,
        country=request.country,
        sentences=sentences,
    )
    result = AnalysisService().analyze(analyze_request)
    sentence_results = [
        SentenceResultDto(
            content_sentence_id=s.content_sentence_id,
            sentence_text=s.sentence_text,
            sentence_order=s.sentence_order,
        )
        for s in sentences
    ]
    return RawAnalysisResultDto(**result.model_dump(), sentences=sentence_results)