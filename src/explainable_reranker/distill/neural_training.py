"""Torch joint-distillation loop for the neural select-then-predict backends.

This is the GPU counterpart of :mod:`explainable_reranker.distill.training` (which
trains the dependency-free linear stand-in). It optimizes the bge-reranker-v2-m3
LoRA adapters + heads with autograd, reusing the exact loss *semantics* defined in
:mod:`explainable_reranker.distill.losses` but on torch tensors so gradients flow.

Design (plan §2.3):
  * The Generator is supervised directly by teacher citations (select BCE) — this
    is what stabilizes select-then-predict versus the unsupervised Lei+2016 form.
    select/sparsity/continuity gradients flow into G.
  * The Predictor sees only physically packed evidence and is trained with the
    listwise KD + hard-anchor ranking loss. Packing is discrete (the faithfulness
    invariant), so no gradient crosses it back into G — G gets its signal from the
    select loss. Which sentences get packed follows the warmup→anneal schedule
    (:class:`TrainingSchedule`): teacher citations early, generator selections late.

torch is imported lazily so importing this module off-GPU never fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from explainable_reranker.distill.dataset import (
    CandidateTrainingExample,
    QueryTrainingBatch,
    pack_selected_evidence,
)
from explainable_reranker.distill.gates import hard_select_from_logits
from explainable_reranker.distill.trainer import TrainingSchedule
from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only off-GPU
        raise RuntimeError(
            "Neural distillation requires torch. Install training extras: "
            "`pip install -e '.[gpu]'` on the DGX Spark."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Tensor losses (mirror distill.losses semantics; keep gradients)
# ---------------------------------------------------------------------------


def _listwise_kd(torch, teacher_scores, student_scores, *, tau: float):
    """KL(softmax(teacher/τ) || softmax(student/τ)) — teacher detached."""

    teacher = torch.softmax(teacher_scores.detach() / tau, dim=0)
    student_log = torch.log_softmax(student_scores / tau, dim=0)
    return torch.sum(teacher * (torch.log(teacher.clamp_min(1e-12)) - student_log))


def _select_bce(torch, gate_probabilities, gate_targets):
    """Mean BCE of per-sentence gate probabilities against teacher selections."""

    if gate_probabilities.numel() == 0:
        return gate_probabilities.new_zeros(())
    probs = gate_probabilities.clamp(1e-12, 1.0 - 1e-12)
    losses = -(gate_targets * torch.log(probs) + (1 - gate_targets) * torch.log(1 - probs))
    return losses.mean()


def _sparsity(torch, gate_probabilities, *, target_fraction: float, weight: float):
    if gate_probabilities.numel() == 0:
        return gate_probabilities.new_zeros(())
    selected_fraction = gate_probabilities.mean()
    return weight * torch.clamp(selected_fraction - target_fraction, min=0.0)


def _continuity(torch, gate_probabilities, *, weight: float):
    if gate_probabilities.numel() < 2:
        return gate_probabilities.new_zeros(())
    diffs = (gate_probabilities[1:] - gate_probabilities[:-1]).abs()
    return weight * diffs.mean()


def _hard_anchor(torch, student_scores, hard_labels, *, weight: float):
    if weight == 0.0:
        return student_scores.new_zeros(())
    rows = [(score, label) for score, label in zip(student_scores, hard_labels) if label is not None]
    if not rows:
        return student_scores.new_zeros(())
    scores = torch.stack([score for score, _ in rows])
    targets = student_scores.new_tensor([float(label) for _, label in rows])
    probs = torch.sigmoid(scores).clamp(1e-12, 1.0 - 1e-12)
    losses = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    return weight * losses.mean()


# ---------------------------------------------------------------------------
# Config + history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralTrainConfig:
    epochs: int = 3
    warmup_steps: int = 200
    total_steps: int = 2000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    tau: float = 1.0
    alpha: float = 0.1  # hard-anchor weight
    beta: float = 1.0  # select-BCE weight
    lambda_sparsity: float = 0.05
    lambda_continuity: float = 0.05
    sparsity_target_fraction: float = 0.2
    log_every: int = 50


@dataclass
class NeuralTrainHistory:
    losses: list[float] = field(default_factory=list)
    rank: list[float] = field(default_factory=list)
    select: list[float] = field(default_factory=list)

    @property
    def final(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


# ---------------------------------------------------------------------------
# Selection-for-packing policy (teacher early, generator late)
# ---------------------------------------------------------------------------


def _annealed_gate_probabilities(torch, generator_logits, temperature: float):
    """Sigmoid gate probabilities with the schedule's HardConcrete temperature.

    plan §2.4-2: anneal the gate temperature high→low so selection sharpens over
    training (soft, exploratory early; near-discrete late). ``TrainingSchedule``
    supplies ``hard_concrete_temperature`` (1.5 → 0.5); dividing the logits by it
    flattens probs toward 0.5 when high and pushes them toward 0/1 when low. The
    hard selection in ``_selected_ids_for_packing`` thresholds the raw logit at 0,
    so temperature only shapes the *soft* probs used by the select/sparsity/
    continuity losses — exactly the term the schedule was meant to drive.
    """

    safe_temperature = max(float(temperature), 1e-6)
    return torch.sigmoid(generator_logits / safe_temperature)


def _selected_ids_for_packing(
    torch,
    candidate: CandidateTrainingExample,
    generator_logits,
    *,
    schedule: TrainingSchedule,
    generator: HFSentenceGenerator,
    rng,
) -> set[str]:
    """Pick which sentence IDs feed the Predictor for this candidate.

    With probability ``teacher_mask_ratio`` use the teacher's citations
    (teacher-forcing the predictor early); otherwise use the generator's own
    hard selection (detached — packing is non-differentiable by design).
    """

    use_teacher = float(torch.rand((), generator=rng)) < schedule.teacher_mask_ratio
    if use_teacher:
        return set(candidate.teacher_rationale_ids())
    logits = generator_logits.detach().float().cpu().tolist()
    selected = hard_select_from_logits(
        logits,
        threshold=0.0,
        min_selected=generator.min_selected,
        max_selected=generator.max_selected,
    )
    return {
        label.sentence.sentence_id
        for label, is_selected in zip(candidate.sentences, selected, strict=True)
        if is_selected
    }


# ---------------------------------------------------------------------------
# Joint training loop
# ---------------------------------------------------------------------------


def train_joint(
    generator: HFSentenceGenerator,
    predictor: HFPackedEvidencePredictor,
    batches: Sequence[QueryTrainingBatch],
    config: NeuralTrainConfig = NeuralTrainConfig(),
    *,
    seed: int = 0,
) -> NeuralTrainHistory:  # pragma: no cover - requires torch + GPU
    torch = _import_torch()
    generator._ensure_loaded()
    predictor._ensure_loaded()
    generator.train_mode()
    predictor.train_mode()

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    params = generator.trainable_parameters() + predictor.trainable_parameters()
    optimizer = torch.optim.AdamW(
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = NeuralTrainHistory()
    step = 0
    for _epoch in range(config.epochs):
        for batch in batches:
            if not batch.candidates:
                continue
            schedule = TrainingSchedule.for_step(
                step, warmup_steps=config.warmup_steps, total_steps=config.total_steps
            )
            optimizer.zero_grad()

            student_scores = []
            gate_prob_chunks = []
            gate_target_chunks = []
            hard_labels = []
            for candidate in batch.candidates:
                sentences = tuple(label.sentence for label in candidate.sentences)
                gen_logits = generator._forward_logits(batch.query, sentences)
                gate_probs = _annealed_gate_probabilities(
                    torch, gen_logits, schedule.hard_concrete_temperature
                )
                gate_prob_chunks.append(gate_probs)
                gate_target_chunks.append(
                    gate_probs.new_tensor([float(label.selected) for label in candidate.sentences])
                )

                selected_ids = _selected_ids_for_packing(
                    torch, candidate, gen_logits,
                    schedule=schedule, generator=generator, rng=rng,
                )
                packed = pack_selected_evidence(candidate, selected_ids)
                student_scores.append(predictor._forward_score(batch.query, packed))
                hard_labels.append(candidate.hard_label)

            student_scores_t = torch.stack(student_scores)
            teacher_scores_t = student_scores_t.new_tensor(batch.teacher_scores())
            all_probs = torch.cat(gate_prob_chunks) if gate_prob_chunks else student_scores_t.new_zeros(0)
            all_targets = torch.cat(gate_target_chunks) if gate_target_chunks else student_scores_t.new_zeros(0)

            rank = _listwise_kd(torch, teacher_scores_t, student_scores_t, tau=config.tau)
            hard = _hard_anchor(torch, student_scores_t, hard_labels, weight=config.alpha)
            select = config.beta * _select_bce(torch, all_probs, all_targets)
            sparsity = sum(
                (_sparsity(torch, probs, target_fraction=config.sparsity_target_fraction,
                           weight=config.lambda_sparsity) for probs in gate_prob_chunks),
                student_scores_t.new_zeros(()),
            )
            continuity = sum(
                (_continuity(torch, probs, weight=config.lambda_continuity) for probs in gate_prob_chunks),
                student_scores_t.new_zeros(()),
            )
            loss = rank + hard + select + sparsity + continuity

            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()

            history.losses.append(float(loss.detach().cpu()))
            history.rank.append(float(rank.detach().cpu()))
            history.select.append(float(select.detach().cpu()))
            if config.log_every and step % config.log_every == 0:
                print(
                    f"step={step} loss={history.losses[-1]:.4f} rank={history.rank[-1]:.4f} "
                    f"select={history.select[-1]:.4f} teacher_mask={schedule.teacher_mask_ratio:.2f}"
                )
            step += 1

    generator.eval_mode()
    predictor.eval_mode()
    return history


def save_neural_checkpoint(
    directory: str | Path,
    generator: HFSentenceGenerator,
    predictor: HFPackedEvidencePredictor,
) -> Path:  # pragma: no cover - requires torch
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    generator.save_pretrained(out)
    predictor.save_pretrained(out)
    return out
