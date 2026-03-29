from fastapi import APIRouter, Request
import json

from analysis_bc.schemas import AnalyzeRequestDto, BiasAnalysisResultDto
from analysis_bc.service import AnalysisService

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("", response_model=BiasAnalysisResultDto)
def analyze(request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
    print("=== 받은 요청 ===")
    print(request.model_dump())
    print("=================")
    return AnalysisService().analyze(request)