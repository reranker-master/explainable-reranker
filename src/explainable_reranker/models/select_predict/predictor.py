from __future__ import annotations


class PackedEvidencePredictor:
    """Predictor stand-in that only receives physically packed selected evidence."""

    def score(self, query: str, packed_evidence: str) -> float:
        if not packed_evidence.strip():
            return 0.0
        return _char_bigram_jaccard(query, packed_evidence) * 3.0


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
