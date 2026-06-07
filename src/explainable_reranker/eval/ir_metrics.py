from __future__ import annotations

from math import log2


def dcg(gains: list[float]) -> float:
    return sum((2**gain - 1) / log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(relevance_by_book: dict[str, float], ranked_book_ids: list[str], *, k: int) -> float:
    gains = [relevance_by_book.get(book_id, 0.0) for book_id in ranked_book_ids[:k]]
    ideal = sorted(relevance_by_book.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 1.0
    return dcg(gains) / ideal_dcg


def mean_reciprocal_rank(relevance_by_book: dict[str, float], ranked_book_ids: list[str], *, threshold: float = 1.0) -> float:
    for rank, book_id in enumerate(ranked_book_ids, start=1):
        if relevance_by_book.get(book_id, 0.0) >= threshold:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevance_by_book: dict[str, float], ranked_book_ids: list[str], *, k: int, threshold: float = 1.0) -> float:
    relevant = {book_id for book_id, relevance in relevance_by_book.items() if relevance >= threshold}
    if not relevant:
        return 1.0
    retrieved = set(ranked_book_ids[:k])
    return len(relevant & retrieved) / len(relevant)
