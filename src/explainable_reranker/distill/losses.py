from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DistillationLossBreakdown:
    rank: float
    hard: float
    select: float
    sparsity: float
    continuity: float

    @property
    def total(self) -> float:
        return self.rank + self.hard + self.select + self.sparsity + self.continuity


def softmax(values: list[float], *, temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not values:
        return []
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def kl_divergence(target_distribution: list[float], student_distribution: list[float]) -> float:
    if len(target_distribution) != len(student_distribution):
        raise ValueError("distributions must have the same length")
    loss = 0.0
    for target, student in zip(target_distribution, student_distribution, strict=True):
        if target == 0.0:
            continue
        loss += target * math.log(target / max(student, 1e-12))
    return loss


def listwise_kd_loss(
    teacher_scores: list[float],
    student_scores: list[float],
    *,
    temperature: float = 1.0,
) -> float:
    return kl_divergence(
        softmax(teacher_scores, temperature=temperature),
        softmax(student_scores, temperature=temperature),
    )


def binary_cross_entropy(predictions: list[float], targets: list[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    total = 0.0
    for prediction, target in zip(predictions, targets, strict=True):
        probability = min(max(prediction, 1e-12), 1.0 - 1e-12)
        total += -(target * math.log(probability) + (1 - target) * math.log(1.0 - probability))
    return total / len(predictions)


def sparsity_loss(gates: list[float], *, target_fraction: float = 0.2, weight: float = 1.0) -> float:
    if not gates:
        return 0.0
    selected_fraction = sum(gates) / len(gates)
    return weight * max(0.0, selected_fraction - target_fraction)


def continuity_loss(gates: list[float], *, weight: float = 1.0) -> float:
    if len(gates) < 2:
        return 0.0
    return weight * sum(abs(right - left) for left, right in zip(gates, gates[1:])) / (len(gates) - 1)


def hard_anchor_loss(student_scores: list[float], hard_labels: list[int | None], *, weight: float) -> float:
    pairs = [(score, label) for score, label in zip(student_scores, hard_labels, strict=True) if label is not None]
    if not pairs or weight == 0.0:
        return 0.0
    normalized_scores = [1.0 / (1.0 + math.exp(-score)) for score, _label in pairs]
    labels = [int(label) for _score, label in pairs]
    return weight * binary_cross_entropy(normalized_scores, labels)


def total_distillation_loss(
    *,
    teacher_scores: list[float],
    student_scores: list[float],
    gate_probabilities: list[list[float]],
    gate_targets: list[list[int]],
    hard_labels: list[int | None] | None = None,
    tau: float = 1.0,
    alpha: float = 0.1,
    beta: float = 1.0,
    lambda_sparsity: float = 0.05,
    lambda_continuity: float = 0.05,
) -> DistillationLossBreakdown:
    flat_gate_probabilities = [value for gates in gate_probabilities for value in gates]
    flat_gate_targets = [value for targets in gate_targets for value in targets]
    rank = listwise_kd_loss(teacher_scores, student_scores, temperature=tau)
    hard = hard_anchor_loss(student_scores, hard_labels or [None for _ in student_scores], weight=alpha)
    select = beta * binary_cross_entropy(flat_gate_probabilities, flat_gate_targets)
    sparsity = sum(
        sparsity_loss(gates, target_fraction=0.2, weight=lambda_sparsity) for gates in gate_probabilities
    )
    continuity = sum(continuity_loss(gates, weight=lambda_continuity) for gates in gate_probabilities)
    return DistillationLossBreakdown(
        rank=rank,
        hard=hard,
        select=select,
        sparsity=sparsity,
        continuity=continuity,
    )
