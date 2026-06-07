from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.topa.adapter import TopaPageResponse


@dataclass(frozen=True)
class RankingLabelerConfig:
    score_scale: float = 3.0


class HeuristicRankingTeacher:
    """Local dummy teacher used for schema tests and offline smoke runs."""

    def __init__(self, config: RankingLabelerConfig | None = None):
        self.config = config or RankingLabelerConfig()

    def label(self, response: TopaPageResponse, sentence_index: list[IndexedSentence]) -> dict:
        evidence_by_book: dict[str, list[IndexedSentence]] = {}
        for sentence in sentence_index:
            evidence_by_book.setdefault(sentence.book_id, []).append(sentence)

        ranking = []
        for candidate in response.candidates:
            evidence_text = " ".join(sentence.text for sentence in evidence_by_book.get(candidate.book_id, []))
            overlap = _char_bigram_jaccard(response.query, f"{candidate.title} {evidence_text}")
            retrieval_bonus = min(max(candidate.score or 0.0, 0.0), 1.0) * 0.15
            score = min(self.config.score_scale, self.config.score_scale * overlap + retrieval_bonus)
            ranking.append({"book": candidate.book_id, "score": round(score, 4)})
        ranking.sort(key=lambda item: item["score"], reverse=True)
        return {"ranking": ranking, "rationales": {}}


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
