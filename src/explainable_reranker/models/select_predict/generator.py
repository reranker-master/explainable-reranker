from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.distill.gates import hard_select_from_logits


@dataclass(frozen=True)
class GateOutput:
    sentence_id: str
    logit: float
    probability: float
    selected: int


class LexicalSentenceGenerator:
    """Generator stand-in that scores sentence evidence against the query."""

    def __init__(self, threshold: float = 0.08, max_selected: int = 3):
        self.threshold = threshold
        self.max_selected = max_selected

    def logits(self, query: str, sentences: tuple[IndexedSentence, ...] | list[IndexedSentence]) -> list[float]:
        return [(_char_bigram_jaccard(query, sentence.text) - self.threshold) * 10.0 for sentence in sentences]

    def select(self, query: str, sentences: tuple[IndexedSentence, ...] | list[IndexedSentence]) -> list[GateOutput]:
        logits = self.logits(query, sentences)
        selected = hard_select_from_logits(logits, threshold=0.0, min_selected=1, max_selected=self.max_selected)
        outputs = []
        for sentence, logit, is_selected in zip(sentences, logits, selected, strict=True):
            probability = 1.0 / (1.0 + pow(2.718281828459045, -logit))
            outputs.append(
                GateOutput(
                    sentence_id=sentence.sentence_id,
                    logit=logit,
                    probability=probability,
                    selected=is_selected,
                )
            )
        return outputs


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
