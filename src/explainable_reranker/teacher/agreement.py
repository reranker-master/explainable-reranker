from __future__ import annotations

from dataclasses import dataclass
from math import log2

from explainable_reranker.teacher.schemas import TeacherLabel


@dataclass(frozen=True)
class AgreementReport:
    weighted_kappa: float
    ndcg_at_10: float
    rationale_f1: float
    rationale_iou: float
    passed: bool


def self_consistency_report(labels: list[TeacherLabel]) -> AgreementReport:
    if len(labels) < 2:
        raise ValueError("self-consistency requires at least two teacher labels")

    kappas: list[float] = []
    ndcgs: list[float] = []
    f1s: list[float] = []
    ious: list[float] = []
    for left_idx in range(len(labels)):
        for right_idx in range(left_idx + 1, len(labels)):
            left = labels[left_idx]
            right = labels[right_idx]
            common_books = sorted(set(left.score_by_book()) & set(right.score_by_book()))
            left_grades = [score_to_grade(left.score_by_book()[book_id]) for book_id in common_books]
            right_grades = [score_to_grade(right.score_by_book()[book_id]) for book_id in common_books]
            kappas.append(weighted_kappa(left_grades, right_grades, max_grade=3))
            ndcgs.append(ndcg_agreement(left.score_by_book(), right.ranked_book_ids(), k=10))
            f1, iou = rationale_overlap(left, right)
            f1s.append(f1)
            ious.append(iou)

    report = AgreementReport(
        weighted_kappa=_mean(kappas),
        ndcg_at_10=_mean(ndcgs),
        rationale_f1=_mean(f1s),
        rationale_iou=_mean(ious),
        passed=False,
    )
    return AgreementReport(
        weighted_kappa=report.weighted_kappa,
        ndcg_at_10=report.ndcg_at_10,
        rationale_f1=report.rationale_f1,
        rationale_iou=report.rationale_iou,
        passed=(
            report.weighted_kappa >= 0.60
            and report.ndcg_at_10 >= 0.85
            and report.rationale_iou >= 0.45
        ),
    )


def score_to_grade(score: float) -> int:
    if score >= 2.5:
        return 3
    if score >= 1.5:
        return 2
    if score >= 0.5:
        return 1
    return 0


def weighted_kappa(left: list[int], right: list[int], *, max_grade: int = 3) -> float:
    if len(left) != len(right):
        raise ValueError("grade vectors must have the same length")
    if not left:
        return 0.0
    num_categories = max_grade + 1
    observed = [[0.0 for _ in range(num_categories)] for _ in range(num_categories)]
    for left_grade, right_grade in zip(left, right, strict=True):
        observed[left_grade][right_grade] += 1.0
    total = float(len(left))
    left_hist = [sum(row[idx] for idx in range(num_categories)) for row in observed]
    right_hist = [sum(observed[idx][col] for idx in range(num_categories)) for col in range(num_categories)]

    observed_weighted = 0.0
    expected_weighted = 0.0
    for i in range(num_categories):
        for j in range(num_categories):
            weight = ((i - j) ** 2) / (max_grade**2)
            observed_weighted += weight * observed[i][j] / total
            expected_weighted += weight * (left_hist[i] * right_hist[j]) / (total * total)
    if expected_weighted == 0.0:
        return 1.0 if observed_weighted == 0.0 else 0.0
    return 1.0 - observed_weighted / expected_weighted


def ndcg_agreement(reference_scores: dict[str, float], ranked_book_ids: list[str], *, k: int) -> float:
    gains = [reference_scores.get(book_id, 0.0) for book_id in ranked_book_ids[:k]]
    ideal = sorted(reference_scores.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 1.0
    return _dcg(gains) / ideal_dcg


def rationale_overlap(left: TeacherLabel, right: TeacherLabel) -> tuple[float, float]:
    f1s: list[float] = []
    ious: list[float] = []
    common_books = sorted(set(left.rationales) & set(right.rationales))
    for book_id in common_books:
        left_set = set(left.rationales[book_id].sentence_ids)
        right_set = set(right.rationales[book_id].sentence_ids)
        if not left_set and not right_set:
            f1s.append(1.0)
            ious.append(1.0)
            continue
        intersection = len(left_set & right_set)
        precision = intersection / len(left_set) if left_set else 0.0
        recall = intersection / len(right_set) if right_set else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        union = len(left_set | right_set)
        ious.append(intersection / union if union else 0.0)
    return _mean(f1s), _mean(ious)


def _dcg(gains: list[float]) -> float:
    return sum((2**gain - 1) / log2(rank + 2) for rank, gain in enumerate(gains))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
