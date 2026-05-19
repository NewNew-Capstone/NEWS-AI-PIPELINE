from __future__ import annotations

import os
from dataclasses import dataclass, field

import fasttext
import httpx
import numpy as np
import torch
from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    EMOTION_SCORE_MARGIN,
    EMOTION_SIMILARITY_THRESHOLD,
    EMOTION_TOP_K,
    FASTTEXT_MODEL_PATH,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

_FASTTEXT_SERVER_URL = os.getenv("FASTTEXT_SERVER_URL", "").strip().rstrip("/")
_DEFAULT_KOTE_MODEL = os.getenv(
    "KOTE_MODEL_NAME",
    "searle-j/kote_for_easygoing_people",
)
_MAX_LENGTH = 192

_VALID_POS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}

KOTE_LABELS = [
    "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔", "화남/분노", "존경", "기대감",
    "우쭐댐/무시함", "안타까움/실망", "비장함", "의심/불신", "뿌듯함", "편안/쾌적", "신기함/관심",
    "아껴주는", "부끄러움", "공포/무서움", "절망", "한심함", "역겨움/징그러움", "짜증", "어이없음",
    "없음", "패배/자기혐오", "귀찮음", "힘듦/지침", "즐거움/신남", "깨달음", "죄책감", "증오/혐오",
    "흐뭇함(귀여움/예쁨)", "당황/난처", "경악", "부담/안_내킴", "서러움", "재미없음", "불쌍함/연민",
    "놀람", "행복", "불안/걱정", "기쁨", "안심/신뢰",
]

NEGATIVE_LABELS = {
    "불평/불만", "지긋지긋", "슬픔", "화남/분노", "우쭐댐/무시함", "안타까움/실망", "의심/불신",
    "공포/무서움", "절망", "한심함", "역겨움/징그러움", "짜증", "어이없음", "패배/자기혐오",
    "귀찮음", "힘듦/지침", "죄책감", "증오/혐오", "당황/난처", "경악", "부담/안_내킴", "서러움",
    "재미없음", "불안/걱정",
}
POSITIVE_LABELS = {
    "환영/호의", "감동/감탄", "고마움", "존경", "기대감", "뿌듯함", "편안/쾌적", "신기함/관심",
    "아껴주는", "즐거움/신남", "깨달음", "흐뭇함(귀여움/예쁨)", "놀람", "행복", "기쁨", "안심/신뢰",
}
STOPWORDS = {
    "중국", "미국", "결국", "과연", "오히려", "어차피", "부리",
}


@dataclass
class GateResult:
    content_sentence_id: int
    sentence_text: str
    gate_score: float
    gate_passed: bool
    top_labels: list[str]
    label_scores: dict[str, float]
    skip_reason: str | None = None


@dataclass
class GateTraceToken:
    surface: str
    form: str
    tag: str
    start: int
    length: int
    accepted: bool
    reject_reason: str | None
    qdrant_scores: list[float] = field(default_factory=list)
    payload_polarities: list[int] = field(default_factory=list)


@dataclass
class GateTraceSentence:
    content_sentence_id: int
    sentence_text: str
    gate_score: float
    gate_passed: bool
    top_labels: list[str]
    skip_reason: str | None
    tokens: list[GateTraceToken] = field(default_factory=list)


class EmotionGateClassifier:
    def __init__(self, model_name: str = _DEFAULT_KOTE_MODEL) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        label_count = self.model.config.num_labels
        if label_count != len(KOTE_LABELS):
            raise ValueError(
                f"KOTE label mismatch: model={label_count}, expected={len(KOTE_LABELS)}"
            )

    def predict_batch(
        self,
        texts: list[str],
        gate_threshold: float,
        label_threshold: float,
        top_k: int,
    ) -> list[GateResult]:
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
        probs = torch.sigmoid(logits).cpu().numpy()

        out: list[GateResult] = []
        for idx, row in enumerate(probs, start=1):
            label_scores = {KOTE_LABELS[i]: float(row[i]) for i in range(len(KOTE_LABELS))}
            sorted_items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            gate_score = float(sorted_items[0][1])
            top_labels = [name for name, score in sorted_items if score >= label_threshold][:top_k]
            gate_passed = gate_score >= gate_threshold and len(top_labels) > 0
            out.append(
                GateResult(
                    content_sentence_id=idx,
                    sentence_text=texts[idx - 1],
                    gate_score=gate_score,
                    gate_passed=gate_passed,
                    top_labels=top_labels,
                    label_scores=label_scores,
                    skip_reason=None if gate_passed else "low_gate_score",
                )
            )
        return out


class ConditionalSpanRunner:
    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        if _FASTTEXT_SERVER_URL:
            self.ft_model = None
            self._ft_server_url = _FASTTEXT_SERVER_URL
        else:
            self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            self._ft_server_url = ""

    def _embed_words(self, words: list[str]) -> list[np.ndarray]:
        if self.ft_model is not None:
            return [self.ft_model.get_word_vector(w) for w in words]
        resp = httpx.post(
            f"{self._ft_server_url}/embed",
            json={"words": words},
            timeout=10.0,
        )
        resp.raise_for_status()
        return [np.array(v) for v in resp.json()["vectors"]]

    @staticmethod
    def _allowed_polarities(top_labels: list[str]) -> set[int]:
        has_neg = any(label in NEGATIVE_LABELS for label in top_labels)
        has_pos = any(label in POSITIVE_LABELS for label in top_labels)
        if has_neg and has_pos:
            return {-2, -1, 1, 2}
        if has_neg:
            return {-2, -1}
        if has_pos:
            return {1, 2}
        return {-2, -1, 1, 2}

    def run(
        self,
        sentences: list[ClassifiedSentenceDto],
        gate_results: list[GateResult],
    ) -> tuple[list[SpanLabelDto], list[GateTraceSentence]]:
        by_id = {g.content_sentence_id: g for g in gate_results}
        spans: list[SpanLabelDto] = []
        trace_sentences: list[GateTraceSentence] = []

        for sent in sentences:
            gate = by_id[sent.content_sentence_id]
            sent_trace = GateTraceSentence(
                content_sentence_id=sent.content_sentence_id,
                sentence_text=sent.sentence_text,
                gate_score=gate.gate_score,
                gate_passed=gate.gate_passed,
                top_labels=gate.top_labels,
                skip_reason=gate.skip_reason,
            )
            if not gate.gate_passed:
                trace_sentences.append(sent_trace)
                continue

            allowed_polarities = self._allowed_polarities(gate.top_labels)
            tokens = self.kiwi.tokenize(sent.sentence_text)
            valid_tokens = []
            for t in tokens:
                surface = sent.sentence_text[t.start:t.start + t.len]
                if t.tag not in _VALID_POS:
                    sent_trace.tokens.append(
                        GateTraceToken(surface, t.form, t.tag, t.start, t.len, False, "invalid_pos")
                    )
                    continue
                if len(t.form) < 2:
                    sent_trace.tokens.append(
                        GateTraceToken(surface, t.form, t.tag, t.start, t.len, False, "too_short")
                    )
                    continue
                if surface in STOPWORDS:
                    sent_trace.tokens.append(
                        GateTraceToken(surface, t.form, t.tag, t.start, t.len, False, "blocked_stopword")
                    )
                    continue
                valid_tokens.append(t)
                sent_trace.tokens.append(
                    GateTraceToken(surface, t.form, t.tag, t.start, t.len, False, None)
                )

            if not valid_tokens:
                trace_sentences.append(sent_trace)
                continue

            surfaces = [sent.sentence_text[t.start:t.start + t.len] for t in valid_tokens]
            vectors = self._embed_words(surfaces)
            queries = []
            for vec in vectors:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                queries.append(
                    QueryRequest(
                        query=vec.tolist(),
                        limit=EMOTION_TOP_K,
                        score_threshold=None,
                        with_payload=True,
                    )
                )
            results = self.qdrant.query_batch_points(
                collection_name=QDRANT_EMOTION_COLLECTION,
                requests=queries,
            )

            trace_idx = 0
            for token, result in zip(valid_tokens, results):
                while trace_idx < len(sent_trace.tokens):
                    cur = sent_trace.tokens[trace_idx]
                    trace_idx += 1
                    if cur.reject_reason is None:
                        trace_token = cur
                        break
                else:
                    continue

                hits = result.points
                trace_token.qdrant_scores = [float(h.score) for h in hits]
                hit_rows = []
                for h in hits:
                    pol = int(str((h.payload or {}).get("polarity", "0")))
                    hit_rows.append((h, pol))
                trace_token.payload_polarities = [pol for _, pol in hit_rows]

                filtered_hits = [h for h, pol in hit_rows if pol in allowed_polarities]
                if not filtered_hits:
                    trace_token.reject_reason = "polarity_mismatch"
                    continue
                top_hit = filtered_hits[0]
                if top_hit.score < EMOTION_SIMILARITY_THRESHOLD:
                    trace_token.reject_reason = "below_threshold"
                    continue
                if len(filtered_hits) >= 2:
                    margin = filtered_hits[0].score - filtered_hits[1].score
                    if margin < EMOTION_SCORE_MARGIN:
                        trace_token.reject_reason = "low_margin"
                        continue

                surface = sent.sentence_text[token.start:token.start + token.len]
                span = SpanLabelDto(
                    content_sentence_id=sent.content_sentence_id,
                    start_offset=token.start,
                    end_offset=token.start + token.len,
                    label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                    score=float(min(max(top_hit.score, 0.0), 1.0)),
                    matched_word=surface,
                )
                spans.append(span)
                trace_token.accepted = True
                trace_token.reject_reason = None

            trace_sentences.append(sent_trace)

        return spans, trace_sentences

