from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from analysis_bc.classifier import ClassifiedSentenceDto, FactOpinionClassifier
from analysis_bc.schemas import SentenceInputDto


@pytest.fixture
def mock_classifier() -> FactOpinionClassifier:
    """실제 모델 로드 없이 FactOpinionClassifier 인스턴스 반환."""
    with (
        patch("analysis_bc.classifier.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "analysis_bc.classifier.ElectraForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_tok_cls.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "fact_like", 1: "opinion_like"}
        mock_model_cls.return_value = mock_model
        clf = FactOpinionClassifier(model_path="dummy")
    return clf


def _set_model_output(clf: FactOpinionClassifier, label_idx: int) -> None:
    """model forward의 logits를 label_idx 기준으로 설정."""
    logits = (
        torch.tensor([[5.0, -5.0]]) if label_idx == 0 else torch.tensor([[-5.0, 5.0]])
    )
    mock_out = MagicMock()
    mock_out.logits = logits
    clf.model.return_value = mock_out
    clf.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("정부는 오늘 새로운 부동산 정책을 발표했다.", "fact_like"),
        ("이번 부동산 정책은 정말 최악의 선택이다.", "opinion_like"),
        ("집값이 너무 올라서 서민들이 너무 힘들다.", "opinion_like"),
        ("서울 아파트 평균 가격이 10억을 넘었다.", "fact_like"),
    ],
)
def test_predict_label(
    mock_classifier: FactOpinionClassifier,
    text: str,
    expected_label: str,
) -> None:
    label_idx = 0 if expected_label == "fact_like" else 1
    _set_model_output(mock_classifier, label_idx)

    label, confidence = mock_classifier.predict(text)

    assert label == expected_label
    assert 0.0 <= confidence <= 1.0


def test_classify_splits_fact_and_opinion(mock_classifier: FactOpinionClassifier) -> None:
    """SentenceInputDto 5개 입력 → fact 3개, opinion 2개로 분리."""
    sentences = [
        SentenceInputDto(content_sentence_id=1, sentence_text="s1", sentence_order=1),
        SentenceInputDto(content_sentence_id=2, sentence_text="s2", sentence_order=2),
        SentenceInputDto(content_sentence_id=3, sentence_text="s3", sentence_order=3),
        SentenceInputDto(content_sentence_id=4, sentence_text="s4", sentence_order=4),
        SentenceInputDto(content_sentence_id=5, sentence_text="s5", sentence_order=5),
    ]
    predict_results = [
        ("fact_like", 0.99),
        ("opinion_like", 0.87),
        ("fact_like", 0.95),
        ("opinion_like", 0.82),
        ("fact_like", 0.91),
    ]

    with patch.object(mock_classifier, "predict", side_effect=predict_results):
        classified = mock_classifier.classify(sentences)

    fact = [s for s in classified if s.label == "fact_like"]
    opinion = [s for s in classified if s.label == "opinion_like"]

    assert len(classified) == 5
    assert len(fact) == 3
    assert len(opinion) == 2


def test_classified_sentence_dto_fields(mock_classifier: FactOpinionClassifier) -> None:
    """ClassifiedSentenceDto 필드가 SentenceInputDto에서 올바르게 복사되는지 검증."""
    sentence = SentenceInputDto(
        content_sentence_id=42,
        sentence_text="테스트 문장",
        sentence_order=3,
        start_time_ms=1000,
        end_time_ms=2000,
    )

    with patch.object(mock_classifier, "predict", return_value=("fact_like", 0.98)):
        result = mock_classifier.classify([sentence])

    assert len(result) == 1
    dto = result[0]
    assert dto.content_sentence_id == 42
    assert dto.sentence_text == "테스트 문장"
    assert dto.sentence_order == 3
    assert dto.start_time_ms == 1000
    assert dto.end_time_ms == 2000
    assert dto.label == "fact_like"
    assert dto.confidence == pytest.approx(0.98)
