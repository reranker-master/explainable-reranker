from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.distill.losses import listwise_kd_loss


@dataclass(frozen=True)
class FullInputKDResult:
    student_scores: list[float]
    kd_loss: float


class FullInputKDStudent:
    """Explanation-free teacher-score distillation baseline contract."""

    def __init__(self, initial_bias: float = 0.0):
        self.initial_bias = initial_bias

    def score(self, feature_scores: list[float]) -> list[float]:
        return [feature_score + self.initial_bias for feature_score in feature_scores]

    def evaluate_kd_loss(self, teacher_scores: list[float], feature_scores: list[float], *, tau: float = 1.0) -> FullInputKDResult:
        student_scores = self.score(feature_scores)
        return FullInputKDResult(
            student_scores=student_scores,
            kd_loss=listwise_kd_loss(teacher_scores, student_scores, temperature=tau),
        )
