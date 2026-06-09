from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import IndexedSentence, build_sentence_index
from explainable_reranker.distill.dataset import build_training_batch
from explainable_reranker.distill.neural_training import _annealed_gate_probabilities
from explainable_reranker.distill.trainer import TrainingSchedule
from explainable_reranker.distill.training import (
    SelectionSample,
    TrainableSelectionGenerator,
    load_checkpoint,
    save_checkpoint,
    selection_accuracy,
    selection_samples,
    sentence_features,
    train_selection,
)
from explainable_reranker.models.select_predict.backends import SentenceGeneratorBackend
from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import (
    HeuristicRationaleTeacher,
    merge_ranking_and_rationales,
)
from explainable_reranker.teacher.schemas import parse_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


def _make_sentence(book_id: str, idx: int, text: str) -> IndexedSentence:
    return IndexedSentence(
        sentence_id=f"{book_id}:s{idx}",
        response_id="resp",
        book_id=book_id,
        source_type="review",
        source_id=f"src{idx}",
        sent_idx=idx,
        text=text,
        text_hash="hash",
        char_start=0,
        char_end=len(text),
        token_offsets=(),
    )


class TrainingLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)
        self.batch = self._teacher_batch()

    def _teacher_batch(self):
        ranking = HeuristicRankingTeacher().label(self.response, self.index)
        ranked = [item["book"] for item in ranking["ranking"]]
        rationale = HeuristicRationaleTeacher().label(self.response, self.index, ranked)
        merged = merge_ranking_and_rationales(ranking, rationale)
        label = parse_teacher_label(
            merged, query_id=self.response.query_id, response_id=self.response.response_id
        )
        return build_training_batch(self.response, self.index, label)

    def test_training_optimizes_and_overfits_separable_set(self) -> None:
        # Linearly separable supervised set proves the gradient loop actually learns.
        samples = (
            [SelectionSample(features=(0.6, 0.5, 1.0), target=1) for _ in range(6)]
            + [SelectionSample(features=(0.03, 0.5, 1.0), target=0) for _ in range(6)]
        )
        generator = TrainableSelectionGenerator()  # zero-init -> p=0.5, BCE=ln2
        history = train_selection(generator, samples, epochs=400, learning_rate=0.8)
        self.assertLess(history.final, history.initial)
        self.assertLess(history.final, 0.05)
        self.assertEqual(selection_accuracy(generator, samples), 1.0)

    def test_training_improves_on_real_teacher_batch(self) -> None:
        samples = selection_samples([self.batch])
        self.assertEqual({s.target for s in samples}, {0, 1})  # non-trivial task
        generator = TrainableSelectionGenerator()
        history = train_selection(generator, samples, epochs=300, learning_rate=0.8)
        self.assertLess(history.final, history.initial)  # loss decreases on real data

    def test_no_collapse_on_many_sentence_book(self) -> None:
        query = "위로되는 잔잔한 가족 이야기"
        positives = [
            _make_sentence("book_x", i, "위로되는 잔잔한 가족 이야기가 마음을 어루만진다")
            for i in range(3)
        ]
        negatives = [
            _make_sentence("book_x", 3 + i, "잔혹한 공포 스릴러 연쇄 살인 추적 ")
            for i in range(5)
        ]
        sentences = positives + negatives
        samples = [
            SelectionSample(features=tuple(sentence_features(query, s)), target=1 if s in positives else 0)
            for s in sentences
        ]
        generator = TrainableSelectionGenerator(max_selected=4)
        train_selection(generator, samples, epochs=400, learning_rate=0.8)

        gates = generator.select(query, sentences)
        selected = sum(g.selected for g in gates)
        self.assertGreaterEqual(selected, 1)          # not z->0
        self.assertLess(selected, len(sentences))     # not z->1

    def test_trained_generator_plugs_into_model(self) -> None:
        samples = selection_samples([self.batch])
        generator = TrainableSelectionGenerator()
        train_selection(generator, samples, epochs=200, learning_rate=0.8)
        self.assertIsInstance(generator, SentenceGeneratorBackend)
        model = SelectThenPredictModel(generator=generator)
        outputs = model.rerank_batch(self.batch)
        self.assertTrue(outputs)
        self.assertTrue(all(out.rationale_sentence_ids for out in outputs))

    def test_checkpoint_round_trip(self) -> None:
        samples = selection_samples([self.batch])
        generator = TrainableSelectionGenerator(max_selected=2)
        train_selection(generator, samples, epochs=50)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(
                Path(tmp) / "gen.json",
                generator,
                base_model="BAAI/bge-reranker-v2-m3",
                metadata={"step": 50},
            )
            restored, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint.base_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(checkpoint.metadata["step"], 50)
        self.assertEqual(restored.max_selected, 2)
        for original, loaded in zip(generator.weights, restored.weights, strict=True):
            self.assertAlmostEqual(original, loaded)

    def test_schedule_transitions_from_teacher_to_generator_masks(self) -> None:
        warm = TrainingSchedule.for_step(0, warmup_steps=10, total_steps=100)
        self.assertEqual(warm.teacher_mask_ratio, 1.0)
        self.assertEqual(warm.generator_mask_ratio, 0.0)
        late = TrainingSchedule.for_step(100, warmup_steps=10, total_steps=100)
        self.assertAlmostEqual(late.generator_mask_ratio, 1.0)
        self.assertLess(late.hard_concrete_temperature, warm.hard_concrete_temperature)


class _FakeTensor:
    """Minimal tensor stub so the gate math is testable without torch (GPU-only)."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __truediv__(self, scalar: float) -> "_FakeTensor":
        return _FakeTensor([value / scalar for value in self.values])


class _FakeTorch:
    @staticmethod
    def sigmoid(tensor: _FakeTensor) -> _FakeTensor:
        return _FakeTensor([1.0 / (1.0 + math.exp(-value)) for value in tensor.values])


class AnnealedGateTemperatureTest(unittest.TestCase):
    def test_temperature_is_applied_and_sharpens_when_low(self) -> None:
        logits = _FakeTensor([2.0])
        warm = _annealed_gate_probabilities(_FakeTorch, logits, 1.5).values[0]
        late = _annealed_gate_probabilities(_FakeTorch, logits, 0.5).values[0]

        # Temperature must actually divide the logit (not be ignored): T=1.5 differs
        # from a plain sigmoid, and a lower T pushes a positive logit's prob higher.
        self.assertAlmostEqual(warm, 1.0 / (1.0 + math.exp(-2.0 / 1.5)))
        self.assertAlmostEqual(late, 1.0 / (1.0 + math.exp(-2.0 / 0.5)))
        self.assertNotAlmostEqual(warm, 1.0 / (1.0 + math.exp(-2.0)))
        self.assertGreater(late, warm)  # low temperature => sharper (closer to 1.0)

    def test_zero_temperature_does_not_divide_by_zero(self) -> None:
        prob = _annealed_gate_probabilities(_FakeTorch, _FakeTensor([0.0]), 0.0).values[0]
        self.assertAlmostEqual(prob, 0.5)


if __name__ == "__main__":
    unittest.main()
