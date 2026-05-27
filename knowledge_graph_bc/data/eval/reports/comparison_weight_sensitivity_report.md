# 비교 추천 가중치 민감도 분석 리포트

- 생성시각(UTC): `2026-05-26T05:10:40.348710+00:00`
- 평가셋: `knowledge_graph_bc/data/eval/comparison_weight_eval.jsonl`
- 평가 row 수: `50`, 그룹 수: `6`
- 관련 후보 기준: `expected_relevance >= 3`
- Baseline 가중치: `{'issue_overlap': 5.0, 'shared_keyword': 2.0, 'shared_entity': 3.0, 'semantic_similarity': 4.0, 'analysis_success': 1.0, 'view_score': 1.0, 'published_at': 0.3}`

## 1) 반박 질문 대응
- Q. 이 가중치는 임의인가요?
  - A. KG 신호(issue/entity)를 강하게 두고 텍스트/메타 신호를 보조로 둔 baseline을 수작업 라벨 평가셋에서 검증했습니다.
- Q. 가중치를 조금 바꾸면 결과가 크게 바뀌나요?
  - A. grid/ablation sweep에서 nDCG@3, Recall@3, Spearman, Top3 Jaccard를 함께 측정해 안정 구간과 취약 구간을 분리했습니다.
- Q. 조회수나 최신성이 추천을 왜곡하지 않나요?
  - A. view/published 가중치는 작게 두었고, ablation에서 해당 신호 제거 시 Top3 변화가 제한적인지 확인합니다.

## 2) Baseline 결과
- mean nDCG@3=1.0000, Recall@3=1.0000, MRR=1.0000, Top3 relevant count=3.00

## 3) 전체 Sweep 요약
- nDCG@3: min=0.7654, median=1.0000, max=1.0000
- Recall@3: min=0.6667, median=1.0000, max=1.0000
- Baseline 대비 순위 안정성: Spearman median=0.9373, Top3 Jaccard median=1.0000

## 4) 안정 상위 조합
| variant | nDCG@3 | Recall@3 | MRR | Top3 relevant | Top3 Jaccard | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 1.0000 |
| `ablate_analysis_success` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 1.0000 |
| `double_analysis_success` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 1.0000 |
| `ablate_published_at` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 1.0000 |
| `double_published_at` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 1.0000 |
| `double_semantic_similarity` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 0.9944 |
| `double_view_score` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 0.9944 |
| `grid__issue_overlapx0.50_shared_keywordx0.50_semantic_similarityx1.50` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 0.9944 |
| `grid__issue_overlapx0.50_shared_keywordx0.50_semantic_similarityx2.00` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 0.9944 |
| `grid__issue_overlapx0.50_shared_keywordx1.00_semantic_similarityx0.50` | 1.0000 | 1.0000 | 1.0000 | 3.00 | 1.0000 | 0.9944 |

## 5) 취약 하위 조합
| variant | nDCG@3 | Recall@3 | MRR | Top3 relevant | Top3 Jaccard | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| `grid__issue_overlapx0.00_shared_keywordx1.50_semantic_similarityx0.00` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9254 |
| `grid__issue_overlapx0.00_shared_keywordx2.00_semantic_similarityx0.00` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9254 |
| `ablate_issue_overlap` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx0.50_semantic_similarityx1.50` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx0.50_semantic_similarityx2.00` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx1.00_semantic_similarityx0.50` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx1.00_semantic_similarityx1.50` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx1.00_semantic_similarityx2.00` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx1.50_semantic_similarityx0.50` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |
| `grid__issue_overlapx0.00_shared_keywordx1.50_semantic_similarityx1.00` | 0.7654 | 0.6667 | 1.0000 | 2.00 | 0.5000 | 0.9349 |

## 6) 해석
- `ablate_issue_overlap`에서 품질이 하락하면 같은 이슈 노드를 강하게 둔 이유를 설명할 수 있습니다.
- `ablate_view_score`, `ablate_published_at` 변화가 작으면 조회수/최신성은 보조 신호라는 방어 논리가 됩니다.
- hard negative가 취약 조합에서 Top3에 진입하면, 키워드/semantic만으로는 부족하고 KG 신호가 필요하다는 근거입니다.
