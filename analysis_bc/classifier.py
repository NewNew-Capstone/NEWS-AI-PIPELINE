from __future__ import annotations

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, ElectraForSequenceClassification

from analysis_bc.schemas import SentenceInputDto

_MAX_LENGTH = 128


class ClassifiedSentenceDto(BaseModel):
    content_sentence_id: int
    sentence_text: str
    sentence_order: int
    start_time_ms: int | None
    end_time_ms: int | None
    label: str
    confidence: float


class FactOpinionClassifier:
    def __init__(self, model_path: str = "analysis_bc/models/best_model") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ElectraForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(-1).item())
        label: str = self.model.config.id2label[pred]
        confidence: float = probs[pred].item()
        return label, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        preds_tensor = logits.argmax(-1)
        probs = torch.softmax(logits, dim=-1)
        confidences = torch.gather(probs, 1, preds_tensor.unsqueeze(1)).squeeze(1)
        return [
            (self.model.config.id2label[pred], conf.item())
            for pred, conf in zip(preds_tensor.tolist(), confidences)
        ]

    def classify(
        self,
        sentences: list[SentenceInputDto],
    ) -> list[ClassifiedSentenceDto]:
        if not sentences:
            return []
        texts = [s.sentence_text for s in sentences]
        predictions = self.predict_batch(texts)
        return [
            ClassifiedSentenceDto(
                content_sentence_id=s.content_sentence_id,
                sentence_text=s.sentence_text,
                sentence_order=s.sentence_order,
                start_time_ms=s.start_time_ms,
                end_time_ms=s.end_time_ms,
                label=label,
                confidence=confidence,
            )
            for s, (label, confidence) in zip(sentences, predictions)
        ]
