from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    """A sentence or paragraph-level evidence item bundled in a topa.page result."""

    source_type: str
    source_id: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopaBookCandidate:
    """A normalized book candidate with evidence text."""

    book_id: str
    title: str
    rank: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceItem, ...] = ()
    # Set when this candidate was mined as a hard negative and mixed into the pool
    # (teacher.hard_negatives). Persisted in the snapshot so the §2 anchor loss can
    # recover a hard label of 0 at train time without re-querying Memgraph.
    is_hard_negative: bool = False
    hard_negative_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopaPageResponse:
    """Normalized topa.page response used as the immutable training snapshot input."""

    response_id: str
    query_id: str
    query: str
    topa_pipeline_version: str
    schema_version: str
    retrieval_params: dict[str, Any]
    candidates: tuple[TopaBookCandidate, ...]
    raw: dict[str, Any]


def parse_topa_page_response(payload: dict[str, Any]) -> TopaPageResponse:
    """Normalize a topa.page-like JSON payload.

    The production topa schema may drift, so this parser intentionally accepts a
    small set of equivalent field names while still failing on missing core data.
    """

    response_id = _first_text(payload, "response_id", "id", default="")
    query_id = _first_text(payload, "query_id", "request_id", default=response_id)
    query = _first_text(payload, "query", "question", default="")
    if not response_id:
        response_id = _stable_fallback_id(query_id, query)
    if not query_id:
        query_id = response_id

    raw_candidates = _first_list(payload, "candidates", "books", "items", "results")
    if not raw_candidates:
        raise ValueError("topa.page response has no candidates/books/items/results")

    candidates: list[TopaBookCandidate] = []
    for idx, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"candidate at index {idx} is not an object")
        # The live /api/search/search-candidates schema nests the book fields under
        # `book` and the retrieval scores under `retrieval_debug`; older/compat
        # payloads keep them flat. Accept both.
        book_obj = raw_candidate.get("book") if isinstance(raw_candidate.get("book"), dict) else {}
        debug_obj = (
            raw_candidate.get("retrieval_debug")
            if isinstance(raw_candidate.get("retrieval_debug"), dict)
            else {}
        )
        book_id = (
            _first_text(raw_candidate, "book_id", "id", "isbn")
            or _first_text(book_obj, "isbn", "id", "book_id")
            or f"book_{idx + 1:04d}"
        )
        title = (
            _first_text(raw_candidate, "title", "name")
            or _first_text(book_obj, "title", "name")
            or book_id
        )
        score = _first_number(raw_candidate, "score", "retrieval_score", "rrf_score")
        if score is None:
            score = _first_number(debug_obj, "rrf_score", "score")
        rank = _first_int(
            raw_candidate, "rank", default=_first_int(raw_candidate, "pre_rerank_rank", default=idx + 1)
        )
        evidence = tuple(_parse_evidence(raw_candidate))
        is_hard_negative = bool(raw_candidate.get("hard_negative") or book_obj.get("hard_negative"))
        hard_negative_reason = (
            _first_text(raw_candidate, "hard_negative_reason") if is_hard_negative else ""
        ) or None
        candidates.append(
            TopaBookCandidate(
                book_id=book_id,
                title=title,
                rank=rank,
                score=score,
                metadata=_metadata_without(raw_candidate, {"evidence", "sentences", "chunks", "reviews"}),
                evidence=evidence,
                is_hard_negative=is_hard_negative,
                hard_negative_reason=hard_negative_reason,
                raw=raw_candidate,
            )
        )

    return TopaPageResponse(
        response_id=response_id,
        query_id=query_id,
        query=query,
        topa_pipeline_version=_first_text(
            payload, "topa_pipeline_version", "pipeline_version", default="unknown"
        ),
        schema_version=_first_text(payload, "schema_version", default="topa.page.compat.v1"),
        retrieval_params=_first_dict(payload, "retrieval_params", "params"),
        candidates=tuple(candidates),
        raw=payload,
    )


def _parse_evidence(raw_candidate: dict[str, Any]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    # Live schema: `chunks` is a mapping {source_type: text}, e.g. {synopsis, review}.
    # These are paragraph-level; sentence splitting/ID/offset assignment happens in
    # data.sentence_index (split_if_needed handles the long-text exception path).
    chunks = raw_candidate.get("chunks")
    if isinstance(chunks, dict):
        for source_type, value in chunks.items():
            if isinstance(value, str):
                text, raw = value.strip(), {"source_type": source_type, "text": value}
            elif isinstance(value, dict):
                text = _first_text(value, "text", "sentence", "content", "body").strip()
                raw = value
            else:
                continue
            if text:
                evidence.append(
                    EvidenceItem(
                        source_type=str(source_type), source_id=str(source_type), text=text, raw=raw
                    )
                )

    # Compat schema: list-based evidence/sentences/chunks.
    raw_evidence = _first_list(raw_candidate, "evidence", "evidence_sentences", "sentences", "chunks")

    for idx, item in enumerate(raw_evidence):
        if isinstance(item, str):
            text = item.strip()
            source_type = "unknown"
            source_id = f"inline_{idx + 1}"
            raw: dict[str, Any] = {"text": item}
        elif isinstance(item, dict):
            text = _first_text(item, "text", "sentence", "content", "body", default="").strip()
            source_type = _first_text(item, "source_type", "type", "collection", default="unknown")
            source_id = _first_text(item, "source_id", "id", "chunk_id", default=f"inline_{idx + 1}")
            raw = item
        else:
            continue

        if text:
            evidence.append(EvidenceItem(source_type=source_type, source_id=source_id, text=text, raw=raw))

    return evidence


def _first_text(mapping: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return str(value)
    return default


def _first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _first_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_list(mapping: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return value
    return []


def _first_dict(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _metadata_without(mapping: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def _stable_fallback_id(query_id: str, query: str) -> str:
    import hashlib

    raw = f"{query_id}\n{query}".encode("utf-8")
    return "resp_" + hashlib.sha256(raw).hexdigest()[:12]
