"""Select-then-predict distillation data, losses, gates, and training helpers."""

from .dataset import CandidateTrainingExample, QueryTrainingBatch, build_training_batch

__all__ = ["CandidateTrainingExample", "QueryTrainingBatch", "build_training_batch"]
