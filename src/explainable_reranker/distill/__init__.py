"""Select-then-predict distillation data, losses, gates, and training helpers."""

from .dataset import CandidateTrainingExample, QueryTrainingBatch, build_training_batch
from .trainer import TrainingSchedule, run_loss_only_step
from .training import (
    GeneratorCheckpoint,
    SelectionSample,
    TrainableSelectionGenerator,
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
    selection_accuracy,
    selection_samples,
    sentence_features,
    train_selection,
)

__all__ = [
    "CandidateTrainingExample",
    "QueryTrainingBatch",
    "build_training_batch",
    "TrainingSchedule",
    "run_loss_only_step",
    "TrainableSelectionGenerator",
    "SelectionSample",
    "TrainingHistory",
    "GeneratorCheckpoint",
    "sentence_features",
    "selection_samples",
    "train_selection",
    "selection_accuracy",
    "save_checkpoint",
    "load_checkpoint",
]
