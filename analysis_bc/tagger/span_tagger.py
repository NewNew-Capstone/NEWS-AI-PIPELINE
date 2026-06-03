from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

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
    EMOTION_GATE_ENABLED,
    EMOTION_GATE_LABEL_THRESHOLD,
    EMOTION_GATE_MODEL_NAME,
    EMOTION_GATE_THRESHOLD,
    EMOTION_GATE_TOP_K,
    EMOTION_POLARITY_FILTER_ENABLED,
    EMOTION_SCORE_MARGIN,
    EMOTION_SIMILARITY_THRESHOLD,
    EMOTION_TOP_K,
    FASTTEXT_MODEL_PATH,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.score_utils import normalize_score
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.tagger.emotion_stopwords import is_blocked_emotion_stopword

_FASTTEXT_SERVER_URL: str = os.getenv("FASTTEXT_SERVER_URL", "").strip().rstrip("/")

logger = logging.getLogger(__name__)

_HEALTH_CHECK_TTL = 30.0
_MAX_LENGTH = 192

_VALID_POS: frozenset[str] = frozenset({"NNG", "XR"})
_PROPER_NOUN_POS: frozenset[str] = frozenset({"NNP"})
_SPAN_BOUNDARY_CHARS: frozenset[str] = frozenset(
    ".,!?;:()[]{}\"'“”‘’…·，。！？；：、"
)
_SEMANTIC_FALLBACK_MIN_SCORE = max(EMOTION_SIMILARITY_THRESHOLD, 0.93)
_SEMANTIC_FALLBACK_NOUN_MARKERS: frozenset[str] = frozenset(
    {
        "분노", "화", "짜증", "혐오", "증오", "경악", "불안", "걱정", "공포", "두려",
        "슬픔", "절망", "실망", "불만", "의심", "불신", "불쾌", "기쁨", "행복",
        "감동", "감탄", "고마움", "안심", "신뢰", "즐거움", "연민", "죄책감",
        "당황", "부끄러움", "귀찮음", "피로", "우려",
    }
)
_SEMANTIC_FALLBACK_NOUN_SUFFIXES: frozenset[str] = frozenset(
    {"감", "감정", "심", "증", "움", "픔"}
)

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


def _hit_clean_seed(hit: object) -> str:
    payload = getattr(hit, "payload", None) or {}
    for key in ("clean_seed", "embed_text", "word_root", "word"):
        value = payload.get(key)
        if value:
            return str(value)
    return str(id(hit))


def _hit_match_seed(hit: object) -> str | None:
    seeds = _hit_match_seeds(hit)
    return next(iter(seeds), None)


def _hit_match_seeds(hit: object) -> frozenset[str]:
    payload = getattr(hit, "payload", None) or {}
    seeds: set[str] = set()
    for key in ("clean_seed", "embed_text", "canonical_seed", "word_root", "word"):
        value = payload.get(key)
        if value:
            seed = str(value).strip()
            if seed:
                seeds.add(seed)
    return frozenset(seeds)


def _dedupe_hits_by_clean_seed(hits: list) -> list:
    deduped = []
    seen: set[str] = set()
    for hit in hits:
        key = _hit_clean_seed(hit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


RejectReason = Literal[
    "invalid_pos",
    "too_short",
    "no_hit",
    "low_margin",
    "below_threshold",
    "polarity_mismatch",
    "low_gate_score",
    "blocked_stopword",
    "proper_noun",
    "seed_mismatch",
]


@dataclass
class TokenTrace:
    form: str
    surface: str
    tag: str
    start: int
    length: int
    accepted: bool = False
    reject_reason: RejectReason | None = None
    qdrant_scores: list[float] = field(default_factory=list)
    payload_polarities: list[int] = field(default_factory=list)
    created_span: SpanLabelDto | None = None


@dataclass(frozen=True)
class EmotionTokenCandidate:
    token: object
    query_text: str
    keyword_text: str
    start_offset: int
    end_offset: int
    matched_word: str


@dataclass
class SentenceTrace:
    content_sentence_id: int
    sentence_text: str
    gate_score: float = 1.0
    gate_passed: bool = True
    top_labels: list[str] = field(default_factory=list)
    skip_reason: str | None = None
    tokens: list[TokenTrace] = field(default_factory=list)


@dataclass
class TagTrace:
    qdrant_healthy: bool
    qdrant_health_reason: str
    gate_enabled: bool
    gate_model_loaded: bool
    sentences: list[SentenceTrace] = field(default_factory=list)


@dataclass
class GateResult:
    content_sentence_id: int
    gate_score: float
    gate_passed: bool
    top_labels: list[str]
    skip_reason: str | None = None


class EmotionGateClassifier:
    def __init__(self, model_name: str = EMOTION_GATE_MODEL_NAME) -> None:
        self.model_name = model_name
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
        sentences: list[ClassifiedSentenceDto],
        gate_threshold: float,
        label_threshold: float,
        top_k: int,
    ) -> dict[int, GateResult]:
        if not sentences:
            return {}

        texts = [s.sentence_text for s in sentences]
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

        out: dict[int, GateResult] = {}
        for idx, row in enumerate(probs):
            sorted_items = sorted(
                ((KOTE_LABELS[i], float(row[i])) for i in range(len(KOTE_LABELS))),
                key=lambda x: x[1],
                reverse=True,
            )
            gate_score = float(sorted_items[0][1])
            top_labels = [name for name, score in sorted_items if score >= label_threshold][:top_k]
            gate_passed = gate_score >= gate_threshold and len(top_labels) > 0
            content_sentence_id = sentences[idx].content_sentence_id
            out[content_sentence_id] = GateResult(
                content_sentence_id=content_sentence_id,
                gate_score=gate_score,
                gate_passed=gate_passed,
                top_labels=top_labels,
                skip_reason=None if gate_passed else "low_gate_score",
            )
        return out


class SpanTagger:
    """OPINION 문장의 span에 감정 태그를 붙여 반환한다."""

    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        self._qdrant_healthy: bool = False
        self._health_checked_at: float = 0.0

        if _FASTTEXT_SERVER_URL:
            self.ft_model = None
            self._ft_server_url = _FASTTEXT_SERVER_URL
        else:
            self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            self._ft_server_url = ""

        self._gate: EmotionGateClassifier | None = None
        self._gate_load_attempted = False
        self.last_debug_trace: TagTrace | None = None

    def _is_qdrant_healthy(self) -> bool:
        now = time.monotonic()
        if now - self._health_checked_at < _HEALTH_CHECK_TTL:
            return self._qdrant_healthy

        try:
            self.qdrant.get_collections()
            self._qdrant_healthy = True
            print("[SpanTagger] Qdrant health check passed")
        except Exception:
            self._qdrant_healthy = False
            print("[SpanTagger] Qdrant health check failed — skipping SpanTagger")
            logger.warning("Qdrant health check failed, skipping SpanTagger")

        self._health_checked_at = now
        return self._qdrant_healthy

    def _get_gate(self) -> EmotionGateClassifier | None:
        if not EMOTION_GATE_ENABLED:
            return None
        if self._gate is not None:
            return self._gate
        if self._gate_load_attempted:
            return None

        self._gate_load_attempted = True
        try:
            self._gate = EmotionGateClassifier()
            return self._gate
        except Exception:
            logger.warning("Emotion gate model load failed; fallback to ungated tagging", exc_info=True)
            return None

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

    def _gate_sentences(self, sentences: list[ClassifiedSentenceDto]) -> dict[int, GateResult]:
        gate = self._get_gate()
        if gate is None:
            return {
                s.content_sentence_id: GateResult(
                    content_sentence_id=s.content_sentence_id,
                    gate_score=1.0,
                    gate_passed=True,
                    top_labels=[],
                )
                for s in sentences
            }
        return gate.predict_batch(
            sentences=sentences,
            gate_threshold=EMOTION_GATE_THRESHOLD,
            label_threshold=EMOTION_GATE_LABEL_THRESHOLD,
            top_k=EMOTION_GATE_TOP_K,
        )

    @staticmethod
    def _is_predicate_tag(tag: str) -> bool:
        return tag == "VV" or tag.startswith("VA")

    @classmethod
    def _is_valid_pos(cls, tag: str) -> bool:
        return tag in _VALID_POS or cls._is_predicate_tag(tag)

    @staticmethod
    def _is_semantic_fallback_hit(hit: object) -> bool:
        payload = getattr(hit, "payload", None) or {}
        category = str(payload.get("category", "")).strip().upper()
        return category == "DIRECT_EMOTION"

    @staticmethod
    def _is_emotion_like_noun(keyword_text: str) -> bool:
        if any(marker in keyword_text for marker in _SEMANTIC_FALLBACK_NOUN_MARKERS):
            return True
        return any(keyword_text.endswith(suffix) for suffix in _SEMANTIC_FALLBACK_NOUN_SUFFIXES)

    @classmethod
    def _allows_semantic_fallback(cls, candidate: EmotionTokenCandidate) -> bool:
        tag = str(candidate.token.tag)
        if tag == "VV":
            return False
        if tag.startswith("VA") or tag == "XR":
            return True
        if tag == "NNG":
            return cls._is_emotion_like_noun(candidate.keyword_text)
        return False

    @classmethod
    def _semantic_fallback_hits(cls, candidate: EmotionTokenCandidate, hits: list) -> list:
        if not cls._allows_semantic_fallback(candidate):
            return []
        return [
            hit for hit in hits
            if (
                float(getattr(hit, "score", 0.0)) >= _SEMANTIC_FALLBACK_MIN_SCORE
                and cls._is_semantic_fallback_hit(hit)
            )
        ]

    @staticmethod
    def _predicate_query_text(form: str) -> str:
        return form if form.endswith("다") else f"{form}다"

    @staticmethod
    def _expand_to_word_end(text: str, default_end: int) -> int:
        end = default_end
        while end < len(text):
            char = text[end]
            if char.isspace() or char in _SPAN_BOUNDARY_CHARS:
                break
            end += 1
        return end

    @staticmethod
    def _is_xr_hada_candidate(tokens: list, token_index: int, token_end: int) -> bool:
        token = tokens[token_index]
        if token.tag != "XR":
            return False

        for next_token in tokens[token_index + 1:]:
            if next_token.start < token_end:
                continue
            if next_token.start > token_end:
                return False
            return next_token.tag == "XSA" and next_token.form == "하"
        return False

    def _build_candidate(
        self,
        sentence_text: str,
        tokens: list,
        token_index: int,
    ) -> EmotionTokenCandidate | None:
        token = tokens[token_index]
        start_offset = token.start
        end_offset = token.start + token.len
        surface = sentence_text[start_offset:end_offset]

        if token.tag == "NNG":
            return EmotionTokenCandidate(
                token=token,
                query_text=surface,
                keyword_text=surface,
                start_offset=start_offset,
                end_offset=end_offset,
                matched_word=surface,
            )

        if self._is_predicate_tag(token.tag):
            keyword_text = self._predicate_query_text(token.form)
            end_offset = self._expand_to_word_end(sentence_text, end_offset)
            return EmotionTokenCandidate(
                token=token,
                query_text=keyword_text,
                keyword_text=keyword_text,
                start_offset=start_offset,
                end_offset=end_offset,
                matched_word=sentence_text[start_offset:end_offset],
            )

        if self._is_xr_hada_candidate(tokens, token_index, end_offset):
            keyword_text = self._predicate_query_text(f"{token.form}하")
            end_offset = self._expand_to_word_end(sentence_text, end_offset)
            return EmotionTokenCandidate(
                token=token,
                query_text=keyword_text,
                keyword_text=keyword_text,
                start_offset=start_offset,
                end_offset=end_offset,
                matched_word=sentence_text[start_offset:end_offset],
            )

        return None

    def tag(
        self,
        sentences: list[ClassifiedSentenceDto],
        debug: bool = False,
    ) -> list[SpanLabelDto]:
        print(f"[SpanTagger] tag() 시작 — opinion 문장 수: {len(sentences)}")
        self.last_debug_trace = None

        is_healthy = self._is_qdrant_healthy()
        if debug:
            self.last_debug_trace = TagTrace(
                qdrant_healthy=is_healthy,
                qdrant_health_reason="ok" if is_healthy else "health_check_failed",
                gate_enabled=EMOTION_GATE_ENABLED,
                gate_model_loaded=self._get_gate() is not None,
            )

        if not is_healthy:
            return []

        gate_results = self._gate_sentences(sentences)
        results: list[SpanLabelDto] = []

        for sentence in sentences:
            gate = gate_results.get(
                sentence.content_sentence_id,
                GateResult(sentence.content_sentence_id, 1.0, True, []),
            )
            sentence_trace = None
            if debug and self.last_debug_trace is not None:
                sentence_trace = SentenceTrace(
                    content_sentence_id=sentence.content_sentence_id,
                    sentence_text=sentence.sentence_text,
                    gate_score=gate.gate_score,
                    gate_passed=gate.gate_passed,
                    top_labels=gate.top_labels,
                    skip_reason=gate.skip_reason,
                )

            if not gate.gate_passed:
                if debug and self.last_debug_trace is not None and sentence_trace is not None:
                    self.last_debug_trace.sentences.append(sentence_trace)
                continue

            tokens = self.kiwi.tokenize(sentence.sentence_text)
            emotion_spans = self._search_emotion(
                sentence=sentence,
                tokens=tokens,
                top_labels=gate.top_labels,
                debug=debug,
                sentence_trace=sentence_trace,
            )
            if debug and self.last_debug_trace is not None and sentence_trace is not None:
                self.last_debug_trace.sentences.append(sentence_trace)
            results.extend(emotion_spans)

        print(f"[SpanTagger] tag() 완료 — 총 라벨 수: {len(results)}")
        return results

    def _search_emotion(
        self,
        sentence: ClassifiedSentenceDto,
        tokens: list,
        top_labels: list[str],
        debug: bool = False,
        sentence_trace: SentenceTrace | None = None,
    ) -> list[SpanLabelDto]:
        spans: list[SpanLabelDto] = []

        valid_candidates: list[EmotionTokenCandidate] = []
        token_traces: list[TokenTrace] = []
        for token_index, t in enumerate(tokens):
            surface = sentence.sentence_text[t.start:t.start + t.len]
            trace = TokenTrace(
                form=t.form,
                surface=surface,
                tag=t.tag,
                start=t.start,
                length=t.len,
            )
            if t.tag in _PROPER_NOUN_POS:
                trace.reject_reason = "proper_noun"
            elif not self._is_valid_pos(t.tag):
                trace.reject_reason = "invalid_pos"
            elif len(t.form) < 2:
                trace.reject_reason = "too_short"
            elif is_blocked_emotion_stopword(surface, t.form):
                trace.reject_reason = "blocked_stopword"
            else:
                candidate = self._build_candidate(
                    sentence_text=sentence.sentence_text,
                    tokens=tokens,
                    token_index=token_index,
                )
                if candidate is None:
                    trace.reject_reason = "invalid_pos"
                else:
                    valid_candidates.append(candidate)
            token_traces.append(trace)

        if debug and sentence_trace is not None:
            sentence_trace.tokens.extend(token_traces)

        if not valid_candidates:
            return spans

        query_texts = [candidate.query_text for candidate in valid_candidates]

        if self.ft_model is not None:
            raw_vecs = [self.ft_model.get_word_vector(s) for s in query_texts]
        else:
            resp = httpx.post(
                f"{self._ft_server_url}/embed",
                json={"words": query_texts},
                timeout=10.0,
            )
            resp.raise_for_status()
            raw_vecs = [np.array(v) for v in resp.json()["vectors"]]

        embeddings = []
        for vec in raw_vecs:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        batch_results = self.qdrant.query_batch_points(
            collection_name=QDRANT_EMOTION_COLLECTION,
            requests=[
                QueryRequest(
                    query=vec.tolist(),
                    limit=EMOTION_TOP_K,
                    score_threshold=None if debug else EMOTION_SIMILARITY_THRESHOLD,
                    with_payload=True,
                )
                for vec in embeddings
            ],
        )

        allowed_polarities = self._allowed_polarities(top_labels)
        valid_index = 0
        for candidate, result in zip(valid_candidates, batch_results):
            hits = result.points
            trace = None
            if debug and sentence_trace is not None:
                while valid_index < len(sentence_trace.tokens):
                    trace_candidate = sentence_trace.tokens[valid_index]
                    valid_index += 1
                    if trace_candidate.reject_reason is None:
                        trace = trace_candidate
                        break

            if debug and trace is not None:
                trace.qdrant_scores = [float(h.score) for h in hits]

            if EMOTION_POLARITY_FILTER_ENABLED:
                hit_rows = []
                for h in hits:
                    try:
                        pol = int(str((h.payload or {}).get("polarity", "0")))
                    except ValueError:
                        pol = 0
                    hit_rows.append((h, pol))
                filtered_hits = [h for h, pol in hit_rows if pol == 0 or pol in allowed_polarities]
                if debug and trace is not None:
                    trace.payload_polarities = [pol for _, pol in hit_rows]
            else:
                filtered_hits = hits

            seed_matched_hits = [
                h for h in filtered_hits
                if candidate.keyword_text in _hit_match_seeds(h)
            ]
            fallback_hits: list = []
            if not seed_matched_hits:
                fallback_hits = self._semantic_fallback_hits(candidate, filtered_hits)
            if filtered_hits and not seed_matched_hits and not fallback_hits:
                if debug and trace is not None:
                    trace.reject_reason = "seed_mismatch"
                continue
            filtered_hits = _dedupe_hits_by_clean_seed(seed_matched_hits or fallback_hits)
            if not filtered_hits:
                if debug and trace is not None:
                    if not hits:
                        trace.reject_reason = "no_hit"
                    else:
                        trace.reject_reason = "polarity_mismatch" if EMOTION_POLARITY_FILTER_ENABLED else "no_hit"
                continue

            top_hit = filtered_hits[0]
            if debug and top_hit.score < EMOTION_SIMILARITY_THRESHOLD:
                if trace is not None:
                    trace.reject_reason = "below_threshold"
                continue

            if len(filtered_hits) >= 2:
                margin = filtered_hits[0].score - filtered_hits[1].score
                if margin < EMOTION_SCORE_MARGIN:
                    if debug and trace is not None:
                        trace.reject_reason = "low_margin"
                    continue

            span = SpanLabelDto(
                content_sentence_id=sentence.content_sentence_id,
                start_offset=candidate.start_offset,
                end_offset=candidate.end_offset,
                label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                score=normalize_score(top_hit.score),
                matched_word=candidate.matched_word,
            )
            spans.append(span)
            if debug and trace is not None:
                trace.accepted = True
                trace.created_span = span
        return spans
