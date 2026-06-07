from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.distill.dataset import QueryTrainingBatch
from explainable_reranker.distill.losses import DistillationLossBreakdown, total_distillation_loss
from explainable_reranker.models.select_predict.model import SelectThenPredictModel


@dataclass(frozen=True)
class TrainingSchedule:
    teacher_mask_ratio: float
    generator_mask_ratio: float
    hard_concrete_temperature: float

    @staticmethod
    def for_step(step: int, *, warmup_steps: int, total_steps: int) -> "TrainingSchedule":
        if step < warmup_steps:
            return TrainingSchedule(
                teacher_mask_ratio=1.0,
                generator_mask_ratio=0.0,
                hard_concrete_temperature=1.5,
            )
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return TrainingSchedule(
            teacher_mask_ratio=1.0 - progress,
            generator_mask_ratio=progress,
            hard_concrete_temperature=1.5 - progress,
        )


def run_loss_only_step(
    model: SelectThenPredictModel,
    batch: QueryTrainingBatch,
    *,
    tau: float = 1.0,
) -> DistillationLossBreakdown:
    outputs = model.rerank_batch(batch)
    student_scores = [output.score for output in outputs]
    gate_probabilities = [[gate.probability for gate in output.gates] for output in outputs]
    gate_targets = [[label.selected for label in candidate.sentences] for candidate in batch.candidates]
    return total_distillation_loss(
        teacher_scores=batch.teacher_scores(),
        student_scores=student_scores,
        gate_probabilities=gate_probabilities,
        gate_targets=gate_targets,
        hard_labels=[candidate.hard_label for candidate in batch.candidates],
        tau=tau,
    )
