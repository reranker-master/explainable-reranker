from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.topa.adapter import TopaPageResponse


@dataclass(frozen=True)
class BaselineScore:
    book_id: str
    score: float


class LexicalBaselineReranker:
    """Off-the-shelf reranker stand-in for local tests and metric plumbing."""

    def score(self, response: TopaPageResponse, sentence_index: list[IndexedSentence]) -> list[BaselineScore]:
        evidence_by_book: dict[str, list[IndexedSentence]] = {}
        for sentence in sentence_index:
            evidence_by_book.setdefault(sentence.book_id, []).append(sentence)
        scores = []
        for candidate in response.candidates:
            text = f"{candidate.title} " + " ".join(
                sentence.text for sentence in evidence_by_book.get(candidate.book_id, [])
            )
            scores.append(BaselineScore(book_id=candidate.book_id, score=_char_bigram_jaccard(response.query, text)))
        return sorted(scores, key=lambda item: item.score, reverse=True)


def _char_bigram_jaccard(left: str, right: str) -> float:
    left_set = _char_bigrams(left)
    right_set = _char_bigrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _char_bigrams(text: str) -> set[str]:
    compact = "".join(char for char in text.lower() if not char.isspace())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[idx : idx + 2] for idx in range(len(compact) - 1)}
