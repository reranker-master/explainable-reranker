from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical hard-negative reasons (mirrors teacher.hard_negatives), so the teacher's
# in-pool trap flags group by the same strategy vocabulary as injected negatives.
HARD_NEGATIVE_REASONS = {"same_genre_diff_mood", "title_variant", "other"}


@dataclass(frozen=True)
class TeacherRankingItem:
    book_id: str
    score: float


@dataclass(frozen=True)
class TeacherRationale:
    book_id: str
    sentence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class TeacherHardNegative:
    """A candidate the teacher judged a plausible-but-wrong trap within the pool.

    These are *in-pool* hard negatives: books retrieval actually surfaced, so the
    label drives the §2 anchor loss on the exact decision boundary the reranker
    faces at inference (no out-of-pool injection / train-serve mismatch).
    """

    book_id: str
    reason: str
    note: str = ""


@dataclass(frozen=True)
class TeacherLabel:
    query_id: str
    response_id: str
    ranking: tuple[TeacherRankingItem, ...]
    rationales: dict[str, TeacherRationale]
    raw: dict[str, Any]
    hard_negatives: dict[str, TeacherHardNegative] = field(default_factory=dict)

    def score_by_book(self) -> dict[str, float]:
        return {item.book_id: item.score for item in self.ranking}

    def ranked_book_ids(self) -> list[str]:
        return [item.book_id for item in sorted(self.ranking, key=lambda item: item.score, reverse=True)]


def parse_teacher_label(
    payload: dict[str, Any],
    *,
    query_id: str,
    response_id: str,
) -> TeacherLabel:
    ranking_payload = payload.get("ranking")
    if not isinstance(ranking_payload, list) or not ranking_payload:
        raise ValueError("teacher output must include a non-empty ranking list")

    ranking: list[TeacherRankingItem] = []
    for idx, item in enumerate(ranking_payload):
        if not isinstance(item, dict):
            raise ValueError(f"ranking item {idx} is not an object")
        book_id = str(item.get("book") or item.get("book_id") or "")
        if not book_id:
            raise ValueError(f"ranking item {idx} has no book/book_id")
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ranking item {idx} has invalid score") from exc
        ranking.append(TeacherRankingItem(book_id=book_id, score=score))

    rationales_payload = payload.get("rationales", {})
    if not isinstance(rationales_payload, dict):
        raise ValueError("teacher output rationales must be an object keyed by book id")

    rationales: dict[str, TeacherRationale] = {}
    for book_id, rationale_payload in rationales_payload.items():
        if not isinstance(rationale_payload, dict):
            raise ValueError(f"rationale for {book_id} is not an object")
        sentence_ids = rationale_payload.get("sentence_ids", [])
        if not isinstance(sentence_ids, list):
            raise ValueError(f"rationale sentence_ids for {book_id} must be a list")
        rationales[str(book_id)] = TeacherRationale(
            book_id=str(book_id),
            sentence_ids=tuple(str(sentence_id) for sentence_id in sentence_ids),
            reason=str(rationale_payload.get("reason", "")),
        )

    hard_negatives_payload = payload.get("hard_negatives", {})
    if not isinstance(hard_negatives_payload, dict):
        raise ValueError("teacher output hard_negatives must be an object keyed by book id")
    hard_negatives: dict[str, TeacherHardNegative] = {}
    for book_id, hn_payload in hard_negatives_payload.items():
        if isinstance(hn_payload, dict):
            reason = str(hn_payload.get("reason", "other") or "other")
            note = str(hn_payload.get("note", ""))
        else:
            # Tolerate a bare reason string, e.g. {"book_id": "title_variant"}.
            reason, note = str(hn_payload or "other"), ""
        hard_negatives[str(book_id)] = TeacherHardNegative(
            book_id=str(book_id), reason=reason, note=note
        )

    return TeacherLabel(
        query_id=query_id,
        response_id=response_id,
        ranking=tuple(ranking),
        rationales=rationales,
        raw=payload,
        hard_negatives=hard_negatives,
    )


def validate_teacher_label(
    label: TeacherLabel,
    *,
    candidate_book_ids: set[str],
    sentence_ids_by_book: dict[str, set[str]],
    require_rationales_for_top_k: int = 10,
) -> list[str]:
    errors: list[str] = []
    seen_books: set[str] = set()

    for item in label.ranking:
        if item.book_id not in candidate_book_ids:
            errors.append(f"ranking references unknown book_id={item.book_id}")
        if item.book_id in seen_books:
            errors.append(f"ranking repeats book_id={item.book_id}")
        seen_books.add(item.book_id)
        if not 0.0 <= item.score <= 3.0:
            errors.append(f"score for book_id={item.book_id} is outside supported 0..3 range")

    sorted_scores = [item.score for item in label.ranking]
    if sorted_scores != sorted(sorted_scores, reverse=True):
        errors.append("ranking is not sorted by descending score")

    for book_id, hard_negative in label.hard_negatives.items():
        if book_id not in candidate_book_ids:
            errors.append(f"hard_negative references unknown book_id={book_id}")
        if hard_negative.reason not in HARD_NEGATIVE_REASONS:
            errors.append(
                f"hard_negative for book_id={book_id} has unknown reason={hard_negative.reason!r}"
            )

    top_books = label.ranked_book_ids()[:require_rationales_for_top_k]
    for book_id in top_books:
        rationale = label.rationales.get(book_id)
        if rationale is None:
            errors.append(f"missing rationale for top book_id={book_id}")
            continue
        if not rationale.sentence_ids:
            errors.append(f"empty rationale sentence_ids for book_id={book_id}")
        known_sentence_ids = sentence_ids_by_book.get(book_id, set())
        for sentence_id in rationale.sentence_ids:
            if sentence_id not in known_sentence_ids:
                errors.append(f"rationale for book_id={book_id} references unknown sentence_id={sentence_id}")

    return errors
