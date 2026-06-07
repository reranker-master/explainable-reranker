from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.topa.adapter import TopaPageResponse


@dataclass(frozen=True)
class RationaleLabelerConfig:
    max_sentences_per_book: int = 2


class HeuristicRationaleTeacher:
    """Local rationale selector that mimics the grounded sentence-ID teacher contract."""

    def __init__(self, config: RationaleLabelerConfig | None = None):
        self.config = config or RationaleLabelerConfig()

    def label(
        self,
        response: TopaPageResponse,
        sentence_index: list[IndexedSentence],
        ranked_book_ids: list[str],
    ) -> dict:
        evidence_by_book: dict[str, list[IndexedSentence]] = {}
        for sentence in sentence_index:
            evidence_by_book.setdefault(sentence.book_id, []).append(sentence)

        rationales = {}
        for book_id in ranked_book_ids:
            scored = [
                (_char_bigram_jaccard(response.query, sentence.text), sentence)
                for sentence in evidence_by_book.get(book_id, [])
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = [sentence for _score, sentence in scored[: self.config.max_sentences_per_book]]
            if not selected and evidence_by_book.get(book_id):
                selected = [evidence_by_book[book_id][0]]
            if selected:
                rationales[book_id] = {
                    "sentence_ids": [sentence.sentence_id for sentence in selected],
                    "reason": _reason_from_sentences(response.query, selected),
                }
        return {"ranking": [], "rationales": rationales}


def merge_ranking_and_rationales(ranking_payload: dict, rationale_payload: dict) -> dict:
    return {
        "ranking": ranking_payload.get("ranking", []),
        "rationales": rationale_payload.get("rationales", {}),
    }


def _reason_from_sentences(query: str, sentences: list[IndexedSentence]) -> str:
    first = sentences[0].text
    return f"질문 '{query}'와 관련해 '{first}' 문장이 핵심 근거입니다."


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
