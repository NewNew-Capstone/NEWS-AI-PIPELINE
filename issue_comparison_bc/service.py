from __future__ import annotations

import numpy as np 
from sentence_transformers import SentenceTransformer 

from issue_comparison_bc.schemas import ClusterSimilarityRequest, ClusterSimilarityResponse

# 프로세스당 모델을 한 번만 로드하는 싱글톤
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _MODEL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 반환한다."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class IssueComparisonService:
    def cluster_similarity(self, req: ClusterSimilarityRequest) -> ClusterSimilarityResponse:
        """제목과 요약 각각의 임베딩 코사인 유사도를 계산한다."""
        model = _get_model()
        title_sim = _cosine(model.encode(req.title_a), model.encode(req.title_b))
        summary_sim = _cosine(model.encode(req.summary_a), model.encode(req.summary_b))
        return ClusterSimilarityResponse(
            title_similarity=title_sim,
            summary_similarity=summary_sim,
            applied_cluster_type="SEARCH_AUTO",
        )
