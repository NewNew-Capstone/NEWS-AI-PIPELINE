from __future__ import annotations

import logging

from fastapi import APIRouter

from issue_comparison_bc.schemas import ClusterSimilarityRequest, ClusterSimilarityResponse
from issue_comparison_bc.service import IssueComparisonService

router = APIRouter(prefix="/issue", tags=["issue_comparison"])
logger = logging.getLogger(__name__)


@router.post("/cluster-similarity", response_model=ClusterSimilarityResponse)
def cluster_similarity(request: ClusterSimilarityRequest) -> ClusterSimilarityResponse:
    """두 이슈의 제목/요약 임베딩 코사인 유사도를 반환한다."""
    response = IssueComparisonService().cluster_similarity(request)
    logger.info(
        "issue_cluster_similarity feature_mode=SEARCH cluster_type=%s",
        response.applied_cluster_type,
    )
    return response
