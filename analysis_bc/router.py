from fastapi import APIRouter

from analysis_bc.schemas import AnalyzeRequestDto, BiasAnalysisResultDto
from analysis_bc.service import AnalysisService

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("", response_model=BiasAnalysisResultDto)
def analyze(request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
    return AnalysisService().analyze(request)
