from __future__ import annotations

from pydantic import BaseModel


class ClusterSimilarityRequest(BaseModel):
    # 비교할 이슈 A의 제목과 요약
    title_a: str
    summary_a: str
    # 비교할 이슈 B의 제목과 요약
    title_b: str
    summary_b: str


class ClusterSimilarityResponse(BaseModel):
    # 제목 임베딩 간 코사인 유사도 (0.0 ~ 1.0)
    title_similarity: float
    # 요약 임베딩 간 코사인 유사도 (0.0 ~ 1.0)
    summary_similarity: float
