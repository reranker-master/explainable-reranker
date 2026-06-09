from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from explainable_reranker.topa.adapter import TopaPageResponse

# plan.md §3 "하드 네거티브": Memgraph theme/mood/trope("동일 장르·다른 무드") and
# books 제목정규화(세트책/개정판) are mixed into the teacher candidate pool to raise
# difficulty. Random negatives are too easy; these plausible-but-wrong distractors are
# what stop the reranker from learning surface genre/title similarity (plan §5.1.3).
#
# This module is the *sourcing* seam only. Marking lives in topa.adapter
# (`TopaBookCandidate.is_hard_negative`) and the learning signal in
# distill.losses.hard_anchor_loss / neural_training._hard_anchor.

# Canonical reasons so downstream code (and analysis) can group by strategy.
REASON_SAME_GENRE_DIFFERENT_MOOD = "same_genre_diff_mood"
REASON_TITLE_VARIANT = "title_variant"  # 세트책/개정판 등 제목정규화로 잡히는 중복


@dataclass(frozen=True)
class HardNegative:
    """A plausible-but-wrong candidate to mix into the teacher pool.

    ``evidence`` carries the same kind of synopsis/review sentences a real topa
    candidate would, so the teacher (and the sentence index) treat it like any
    other book rather than an obviously empty distractor.
    """

    book_id: str
    title: str
    reason: str
    evidence: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HardNegativeSource(Protocol):
    """External boundary for hard-negative mining.

    Mirrors the ``TopaPageClient`` seam: offline pipelines use
    :class:`StaticHardNegativeSource`, production uses
    :class:`MemgraphHardNegativeSource`. ``existing_candidates`` is the raw topa
    payload so a source can avoid proposing books already in the pool.
    """

    def fetch(self, query: str, payload: dict[str, Any]) -> list[HardNegative]:
        ...


@dataclass
class StaticHardNegativeSource:
    """Offline source that replays canned hard negatives per query.

    ``by_query`` maps a query string to its hard negatives; ``default`` is used
    for any unmapped query. Used in tests and for ``--hard-negatives <file>``
    runs so the whole pipeline exercises injection without Memgraph.
    """

    by_query: dict[str, list[HardNegative]] = field(default_factory=dict)
    default: list[HardNegative] = field(default_factory=list)

    def fetch(self, query: str, payload: dict[str, Any]) -> list[HardNegative]:
        return list(self.by_query.get(query, self.default))

    @classmethod
    def from_file(cls, path: str | Path) -> StaticHardNegativeSource:
        """Load `{query: [neg, ...]}` (or a bare `[neg, ...]` applied to all)."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return cls(default=[_negative_from_dict(item) for item in raw])
        by_query = {
            str(query): [_negative_from_dict(item) for item in items]
            for query, items in raw.items()
        }
        return cls(by_query=by_query)


@dataclass
class MemgraphHardNegativeSource:
    """Production skeleton for plan §3 Memgraph-mined hard negatives.

    Two strategies, both intentionally *hard*:
      1. same genre / opposite mood — `theme`/`mood`/`trope` neighbours that share
         the query book's genre but flip the mood (the most confusable case).
      2. title normalization — 세트책/개정판 variants of an in-pool book.

    The Cypher boundary is injected (`query_executor: cypher -> list[row dict]`)
    so the strategies stay unit-testable without a live Memgraph, matching the
    ``HttpTopaPageClient`` opener seam. No executor → a clear error rather than a
    silent empty result (which would read as "no hard negatives" and disable the
    anchor loss without warning).
    """

    query_executor: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None
    max_per_strategy: int = 5

    def fetch(self, query: str, payload: dict[str, Any]) -> list[HardNegative]:
        if self.query_executor is None:
            raise RuntimeError(
                "MemgraphHardNegativeSource needs a query_executor (cypher -> rows); "
                "inject one or use StaticHardNegativeSource for offline runs"
            )
        existing = _existing_book_ids(payload)
        negatives: list[HardNegative] = []
        negatives.extend(self._same_genre_other_mood(query, existing))
        negatives.extend(self._title_variants(existing))
        return negatives

    def _same_genre_other_mood(self, query: str, existing: set[str]) -> list[HardNegative]:
        rows = self.query_executor(_CYPHER_SAME_GENRE_OTHER_MOOD, {"query": query, "limit": self.max_per_strategy})  # type: ignore[misc]
        return [
            _negative_from_row(row, REASON_SAME_GENRE_DIFFERENT_MOOD)
            for row in rows
            if str(row.get("book_id", "")) and str(row.get("book_id")) not in existing
        ]

    def _title_variants(self, existing: set[str]) -> list[HardNegative]:
        rows = self.query_executor(_CYPHER_TITLE_VARIANTS, {"book_ids": sorted(existing), "limit": self.max_per_strategy})  # type: ignore[misc]
        return [
            _negative_from_row(row, REASON_TITLE_VARIANT)
            for row in rows
            if str(row.get("book_id", "")) and str(row.get("book_id")) not in existing
        ]


# Cypher kept here as documentation of the production mining strategy; the live
# schema (Book/Theme/Mood/Trope labels) is owned by topa, so these are templates.
_CYPHER_SAME_GENRE_OTHER_MOOD = (
    "MATCH (q:Book {query_ref:$query})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(b:Book) "
    "MATCH (q)-[:HAS_MOOD]->(qm:Mood) MATCH (b)-[:HAS_MOOD]->(bm:Mood) "
    "WHERE bm <> qm RETURN b.isbn AS book_id, b.title AS title, b.synopsis AS synopsis "
    "LIMIT $limit"
)
_CYPHER_TITLE_VARIANTS = (
    "MATCH (b:Book) WHERE b.isbn IN $book_ids "
    "MATCH (v:Book) WHERE v.title_normalized = b.title_normalized AND v.isbn <> b.isbn "
    "RETURN v.isbn AS book_id, v.title AS title, v.synopsis AS synopsis LIMIT $limit"
)


def inject_hard_negatives(
    payload: dict[str, Any],
    negatives: list[HardNegative],
    *,
    max_negatives: int | None = None,
) -> dict[str, Any]:
    """Return a copy of the raw topa payload with hard negatives appended.

    Each injected candidate is marked ``hard_negative: true`` (parsed by
    :mod:`topa.adapter` into ``TopaBookCandidate.is_hard_negative``), so the
    marking is persisted in the immutable snapshot and recovered at train time
    without re-querying Memgraph. Negatives whose ``book_id`` already appears in
    the pool are skipped so we never relabel a real candidate as negative.
    """

    candidates_key = _candidates_key(payload)
    if candidates_key is None:
        raise ValueError("topa payload has no candidates/books/items/results list to extend")

    augmented = copy.deepcopy(payload)
    pool = augmented[candidates_key]
    existing = _existing_book_ids(payload)

    added = 0
    seen: set[str] = set()
    for negative in negatives:
        if max_negatives is not None and added >= max_negatives:
            break
        book_id = str(negative.book_id)
        if not book_id or book_id in existing or book_id in seen:
            continue
        seen.add(book_id)
        pool.append(_candidate_dict(negative))
        added += 1
    return augmented


def hard_label_map(response: TopaPageResponse) -> dict[str, int]:
    """Map each injected hard-negative book to a hard label of 0 (known irrelevant).

    Feeds ``build_training_batch(..., hard_labels=...)`` so the §2 anchor loss
    actually fires. Empty dict if a snapshot has no injected negatives.
    """

    return {c.book_id: 0 for c in response.candidates if c.is_hard_negative}


def _candidate_dict(negative: HardNegative) -> dict[str, Any]:
    # Flat shape accepted by topa.adapter.parse_topa_page_response; evidence is the
    # compat list form so each sentence gets its own ID/offset in sentence_index.
    candidate: dict[str, Any] = dict(negative.metadata)
    candidate.update(
        {
            "book_id": negative.book_id,
            "title": negative.title,
            "hard_negative": True,
            "hard_negative_reason": negative.reason,
            "evidence": [
                {
                    "text": text,
                    "source_type": "synopsis",
                    "source_id": f"hn:{negative.book_id}:{idx}",
                }
                for idx, text in enumerate(negative.evidence, start=1)
            ],
        }
    )
    return candidate


def _negative_from_dict(item: dict[str, Any]) -> HardNegative:
    evidence = item.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    return HardNegative(
        book_id=str(item["book_id"]),
        title=str(item.get("title", item["book_id"])),
        reason=str(item.get("reason", REASON_SAME_GENRE_DIFFERENT_MOOD)),
        evidence=tuple(str(text) for text in evidence),
        metadata=dict(item.get("metadata", {})),
    )


def _negative_from_row(row: dict[str, Any], reason: str) -> HardNegative:
    synopsis = row.get("synopsis")
    evidence = (str(synopsis),) if synopsis else ()
    return HardNegative(
        book_id=str(row["book_id"]),
        title=str(row.get("title", row["book_id"])),
        reason=reason,
        evidence=evidence,
    )


def _candidates_key(payload: dict[str, Any]) -> str | None:
    for key in ("candidates", "books", "items", "results"):
        if isinstance(payload.get(key), list):
            return key
    return None


def _existing_book_ids(payload: dict[str, Any]) -> set[str]:
    key = _candidates_key(payload)
    if key is None:
        return set()
    ids: set[str] = set()
    for raw in payload[key]:
        if not isinstance(raw, dict):
            continue
        book_obj = raw.get("book") if isinstance(raw.get("book"), dict) else {}
        for source in (raw, book_obj):
            for field_name in ("book_id", "id", "isbn"):
                value = source.get(field_name)
                if value is not None:
                    ids.add(str(value))
                    break
    return ids
