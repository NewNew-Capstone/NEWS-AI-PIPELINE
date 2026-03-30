from __future__ import annotations

from fastapi import APIRouter

from issue_comparison_bc.schemas import ClusterSimilarityRequest, ClusterSimilarityResponse
from issue_comparison_bc.service import IssueComparisonService

router = APIRouter(prefix="/issue", tags=["issue_comparison"])


@router.post("/cluster-similarity", response_model=ClusterSimilarityResponse)
def cluster_similarity(request: ClusterSimilarityRequest) -> ClusterSimilarityResponse:
    """두 이슈의 제목/요약 임베딩 코사인 유사도를 반환한다."""
    return IssueComparisonService().cluster_similarity(request)
