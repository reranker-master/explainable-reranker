from __future__ import annotations


def set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    intersection = len(predicted & gold)
    precision = intersection / len(predicted)
    recall = intersection / len(gold)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_iou(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    union = predicted | gold
    return len(predicted & gold) / len(union) if union else 0.0


def token_f1(predicted_offsets: set[tuple[int, int]], gold_offsets: set[tuple[int, int]]) -> float:
    return set_f1({f"{start}:{end}" for start, end in predicted_offsets}, {f"{start}:{end}" for start, end in gold_offsets})


def comprehensiveness(original_score: float, score_without_rationale: float) -> float:
    return max(0.0, original_score - score_without_rationale)


def sufficiency(original_score: float, score_with_only_rationale: float) -> float:
    return abs(original_score - score_with_only_rationale)
