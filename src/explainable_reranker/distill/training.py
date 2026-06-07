from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.distill.dataset import QueryTrainingBatch
from explainable_reranker.distill.gates import HardConcreteGate, hard_select_from_logits
from explainable_reranker.models.select_predict.generator import GateOutput


# ---------------------------------------------------------------------------
# Features and a trainable logistic selection gate
# ---------------------------------------------------------------------------


def sentence_features(query: str, sentence: IndexedSentence) -> list[float]:
    """Small, dependency-free feature vector for a (query, sentence) pair.

    This stands in for the neural Generator's per-sentence representation while
    keeping the training loop real (analytic gradients, genuine optimization).
    The neural backend (`HFSentenceGenerator`) replaces these features with a
    single-forward pooled hidden state, but reuses the same gate/selection math.
    """

    overlap = _char_bigram_jaccard(query, sentence.text)
    length_norm = min(len(sentence.text) / 80.0, 1.0)
    return [overlap, length_norm, 1.0]  # last entry is the bias feature


@dataclass
class TrainableSelectionGenerator:
    """Logistic gate over :func:`sentence_features`, trained on teacher citations.

    Implements the ``SentenceGeneratorBackend`` protocol (``select``) so a trained
    instance can be dropped straight into ``SelectThenPredictModel``. Selection is
    supervised by teacher rationale labels (plan §2.3b), which is what stabilizes
    select-then-predict versus the unsupervised Lei+2016 original.
    """

    weights: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gate: HardConcreteGate = field(default_factory=HardConcreteGate)
    min_selected: int = 1
    max_selected: int = 3

    def logit(self, features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features, strict=True))

    def probability(self, features: Sequence[float]) -> float:
        return 1.0 / (1.0 + math.exp(-self.logit(features)))

    def logits(self, query: str, sentences: Sequence[IndexedSentence]) -> list[float]:
        return [self.logit(sentence_features(query, sentence)) for sentence in sentences]

    def select(self, query: str, sentences: Sequence[IndexedSentence]) -> list[GateOutput]:
        logits = self.logits(query, sentences)
        selected = hard_select_from_logits(
            logits, threshold=0.0, min_selected=self.min_selected, max_selected=self.max_selected
        )
        return [
            GateOutput(
                sentence_id=sentence.sentence_id,
                logit=logit,
                probability=self.gate.probability(logit),
                selected=is_selected,
            )
            for sentence, logit, is_selected in zip(sentences, logits, selected, strict=True)
        ]


# ---------------------------------------------------------------------------
# Training data and the gradient-descent loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionSample:
    features: tuple[float, ...]
    target: int


def selection_samples(batches: Sequence[QueryTrainingBatch]) -> list[SelectionSample]:
    """Flatten query batches into per-sentence (features, teacher-selected) pairs."""

    samples: list[SelectionSample] = []
    for batch in batches:
        for candidate in batch.candidates:
            for label in candidate.sentences:
                features = sentence_features(batch.query, label.sentence)
                samples.append(SelectionSample(features=tuple(features), target=int(label.selected)))
    return samples


@dataclass
class TrainingHistory:
    losses: list[float] = field(default_factory=list)

    @property
    def initial(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def final(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def train_selection(
    generator: TrainableSelectionGenerator,
    samples: Sequence[SelectionSample],
    *,
    epochs: int = 200,
    learning_rate: float = 0.5,
    lambda_sparsity: float = 0.02,
    l2: float = 0.0,
) -> TrainingHistory:
    """Full-batch gradient descent on supervised selection (BCE) + sparsity.

    Analytic gradients (sigmoid + BCE → ``p - y``) keep this a genuine training
    loop with no autograd dependency. Sparsity nudges the gate toward selecting
    few sentences, mirroring ``L_sparsity`` in plan §2.3c.
    """

    if not samples:
        return TrainingHistory()
    history = TrainingHistory()
    dim = len(generator.weights)
    n = len(samples)
    for _ in range(epochs):
        grad = [0.0] * dim
        total_loss = 0.0
        for sample in samples:
            features = sample.features
            logit = generator.logit(features)
            prob = 1.0 / (1.0 + math.exp(-logit))
            prob_clamped = min(max(prob, 1e-12), 1.0 - 1e-12)
            total_loss += -(
                sample.target * math.log(prob_clamped)
                + (1 - sample.target) * math.log(1.0 - prob_clamped)
            )
            # dBCE/dlogit = (p - y); add sparsity pressure dp/dlogit = p(1-p).
            error = (prob - sample.target) + lambda_sparsity * prob * (1.0 - prob)
            for i in range(dim):
                grad[i] += error * features[i]
        for i in range(dim):
            gradient = grad[i] / n + l2 * generator.weights[i]
            generator.weights[i] -= learning_rate * gradient
        history.losses.append(total_loss / n)
    return history


def selection_accuracy(
    generator: TrainableSelectionGenerator, samples: Sequence[SelectionSample]
) -> float:
    """Fraction of sentences whose thresholded gate matches the teacher label."""

    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        predicted = 1 if generator.logit(sample.features) > 0.0 else 0
        correct += int(predicted == sample.target)
    return correct / len(samples)


# ---------------------------------------------------------------------------
# Checkpoint IO (plan §2.1: base id + adapter params + metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorCheckpoint:
    base_model: str
    weights: tuple[float, ...]
    min_selected: int
    max_selected: int
    feature_names: tuple[str, ...]
    metadata: dict


def save_checkpoint(
    path: str | Path,
    generator: TrainableSelectionGenerator,
    *,
    base_model: str,
    metadata: dict | None = None,
) -> Path:
    checkpoint = {
        "format": "explainable-reranker.generator.v1",
        "base_model": base_model,
        "weights": list(generator.weights),
        "min_selected": generator.min_selected,
        "max_selected": generator.max_selected,
        "feature_names": ["query_overlap", "length_norm", "bias"],
        "metadata": metadata or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_checkpoint(path: str | Path) -> tuple[TrainableSelectionGenerator, GeneratorCheckpoint]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    generator = TrainableSelectionGenerator(
        weights=list(payload["weights"]),
        min_selected=int(payload.get("min_selected", 1)),
        max_selected=int(payload.get("max_selected", 3)),
    )
    checkpoint = GeneratorCheckpoint(
        base_model=str(payload.get("base_model", "")),
        weights=tuple(payload["weights"]),
        min_selected=int(payload.get("min_selected", 1)),
        max_selected=int(payload.get("max_selected", 3)),
        feature_names=tuple(payload.get("feature_names", [])),
        metadata=payload.get("metadata", {}),
    )
    return generator, checkpoint


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
