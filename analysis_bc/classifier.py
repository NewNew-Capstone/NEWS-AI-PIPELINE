from __future__ import annotations

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, ElectraForSequenceClassification

from analysis_bc.schemas import SentenceInputDto


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
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(logits.argmax(-1).item())
        label: str = self.model.config.id2label[pred]
        confidence: float = probs[pred].item()
        return label, confidence

    def classify(
        self,
        sentences: list[SentenceInputDto],
    ) -> list[ClassifiedSentenceDto]:
        results: list[ClassifiedSentenceDto] = []
        for s in sentences:
            label, confidence = self.predict(s.sentence_text)
            results.append(
                ClassifiedSentenceDto(
                    content_sentence_id=s.content_sentence_id,
                    sentence_text=s.sentence_text,
                    sentence_order=s.sentence_order,
                    start_time_ms=s.start_time_ms,
                    end_time_ms=s.end_time_ms,
                    label=label,
                    confidence=confidence,
                )
            )
        return results
