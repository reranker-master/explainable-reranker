from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.topa.adapter import parse_topa_page_response

# plan §1.5: v1 근거 코퍼스는 topa.page 응답에 동봉된 후보 문장 그 자체이고, sentence_index는
# "추출"이 아니라 받은 문장에 ID/offset만 부여한다. book_chunks/Qdrant review·synopsis 컬렉션은
# JSON에 문장이 부족하거나 보강이 필요할 때만 fallback으로 조회한다. 이 모듈이 그 fallback seam.
#
# 소싱(Protocol + Static + Qdrant 스켈레톤)만 담당하고, 가져온 문장은 topa 후보와 동일한 evidence
# 형태로 payload에 주입되어 sentence_index가 다른 문장과 똑같이 ID/offset을 부여한다.

FALLBACK_SOURCE_TYPE = "fallback"


@runtime_checkable
class EvidenceFallbackSource(Protocol):
    """External boundary for supplementary per-book evidence sentences.

    Mirrors the topa/hard-negative seams: offline pipelines use
    :class:`StaticEvidenceFallback`, production uses :class:`QdrantEvidenceFallback`.
    """

    def fetch(self, book_id: str, query: str, *, need: int) -> list[str]:
        """Return up to ``need`` extra evidence sentence texts for ``book_id``."""


@dataclass
class StaticEvidenceFallback:
    """Offline fallback that replays canned sentences per book id."""

    by_book: dict[str, list[str]] = field(default_factory=dict)

    def fetch(self, book_id: str, query: str, *, need: int) -> list[str]:
        return [str(text) for text in self.by_book.get(book_id, [])][: max(need, 0)]

    @classmethod
    def from_file(cls, path: str | Path) -> StaticEvidenceFallback:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(by_book={str(book): [str(t) for t in texts] for book, texts in raw.items()})


@dataclass
class QdrantEvidenceFallback:
    """Production skeleton for plan §1.5 book_chunks/Qdrant evidence fallback.

    The vector-search boundary is injected (``query_executor: (collection, params)
    -> list[row dict]``) so the policy stays unit-testable without a live Qdrant,
    matching the other adapter skeletons. No executor → a clear error rather than a
    silent empty result (which would read as "enough evidence" and skip the fallback).
    """

    query_executor: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None
    collection: str = "book_chunks"

    def fetch(self, book_id: str, query: str, *, need: int) -> list[str]:
        if need <= 0:
            return []
        if self.query_executor is None:
            raise RuntimeError(
                "QdrantEvidenceFallback needs a query_executor (collection, params) -> rows; "
                "inject one or use StaticEvidenceFallback for offline runs"
            )
        rows = self.query_executor(
            self.collection, {"book_id": book_id, "query": query, "limit": need}
        )
        texts = [str(row.get("text", "")).strip() for row in rows]
        return [text for text in texts if text][:need]


def augment_with_fallback(
    payload: dict[str, Any],
    source: EvidenceFallbackSource,
    *,
    min_sentences: int,
) -> dict[str, Any]:
    """Return a copy of the payload with fallback evidence added to thin candidates.

    A candidate is "thin" when sentence_index yields fewer than ``min_sentences``
    sentences for it; only those trigger a fallback query (plan §1.5: fallback is
    the exception, not the default path). Fetched sentences are appended as ordinary
    evidence so they get IDs/offsets like any other sentence.
    """

    if min_sentences <= 0:
        return payload

    response = parse_topa_page_response(payload)
    counts: dict[str, int] = {}
    for sentence in build_sentence_index(response):
        counts[sentence.book_id] = counts.get(sentence.book_id, 0) + 1

    candidates_key = _candidates_key(payload)
    if candidates_key is None:
        return payload
    augmented = copy.deepcopy(payload)

    for raw_candidate in augmented[candidates_key]:
        if not isinstance(raw_candidate, dict):
            continue
        book_id = _candidate_book_id(raw_candidate)
        if not book_id:  # adapter would synthesize an id we can't match here; skip
            continue
        need = min_sentences - counts.get(book_id, 0)
        if need <= 0:
            continue
        texts = source.fetch(book_id, response.query, need=need)
        if not texts:
            continue
        evidence = raw_candidate.setdefault("evidence", [])
        if not isinstance(evidence, list):
            evidence = list(evidence) if isinstance(evidence, (tuple, list)) else []
            raw_candidate["evidence"] = evidence
        for idx, text in enumerate(texts, start=1):
            evidence.append(
                {
                    "text": text,
                    "source_type": FALLBACK_SOURCE_TYPE,
                    "source_id": f"fallback:{book_id}:{idx}",
                }
            )
    return augmented


def _candidates_key(payload: dict[str, Any]) -> str | None:
    for key in ("candidates", "books", "items", "results"):
        if isinstance(payload.get(key), list):
            return key
    return None


def _candidate_book_id(raw_candidate: dict[str, Any]) -> str:
    book_obj = raw_candidate.get("book") if isinstance(raw_candidate.get("book"), dict) else {}
    for source in (raw_candidate, book_obj):
        for field_name in ("book_id", "id", "isbn"):
            value = source.get(field_name)
            if value is not None:
                return str(value)
    return ""
