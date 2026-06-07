from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.distill.dataset import build_training_batch
from explainable_reranker.distill.gates import HardConcreteGate, hard_select_from_logits
from explainable_reranker.distill.losses import listwise_kd_loss, total_distillation_loss
from explainable_reranker.distill.trainer import TrainingSchedule, run_loss_only_step
from explainable_reranker.models.baseline import LexicalBaselineReranker
from explainable_reranker.models.full_input_kd import FullInputKDStudent
from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import HeuristicRationaleTeacher, merge_ranking_and_rationales
from explainable_reranker.teacher.schemas import parse_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class DistillPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)
        ranking_payload = HeuristicRankingTeacher().label(self.response, self.index)
        ranked_book_ids = [item["book"] for item in ranking_payload["ranking"]]
        rationale_payload = HeuristicRationaleTeacher().label(self.response, self.index, ranked_book_ids)
        self.teacher_label = parse_teacher_label(
            merge_ranking_and_rationales(ranking_payload, rationale_payload),
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )

    def test_training_batch_maps_teacher_rationales(self) -> None:
        batch = build_training_batch(self.response, self.index, self.teacher_label)
        self.assertEqual(batch.query_id, "q_demo_001")
        self.assertEqual(len(batch.candidates), 3)
        self.assertGreaterEqual(len(batch.candidates[0].teacher_rationale_ids()), 1)
        selected_counts = [sum(label.selected for label in candidate.sentences) for candidate in batch.candidates]
        self.assertTrue(any(count > 0 for count in selected_counts))

    def test_distillation_losses_are_finite(self) -> None:
        self.assertAlmostEqual(listwise_kd_loss([3.0, 1.0], [3.0, 1.0]), 0.0)
        breakdown = total_distillation_loss(
            teacher_scores=[3.0, 1.0],
            student_scores=[2.7, 1.2],
            gate_probabilities=[[0.9, 0.1], [0.6]],
            gate_targets=[[1, 0], [1]],
            hard_labels=[1, None],
        )
        self.assertGreaterEqual(breakdown.total, 0.0)
        self.assertGreaterEqual(breakdown.select, 0.0)

    def test_hard_concrete_and_hard_selection(self) -> None:
        gate = HardConcreteGate()
        self.assertGreater(gate.deterministic(2.0), gate.deterministic(-2.0))
        self.assertGreater(gate.expected_l0(1.0), gate.expected_l0(-1.0))
        self.assertEqual(hard_select_from_logits([-5.0, -2.0, -3.0], min_selected=1), [0, 1, 0])

    def test_select_then_predict_physically_packs_selected_evidence(self) -> None:
        batch = build_training_batch(self.response, self.index, self.teacher_label)
        model = SelectThenPredictModel()
        outputs = model.rerank_batch(batch)
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertGreaterEqual(len(output.rationale_sentence_ids), 1)
            selected_texts = [span.text for span in output.spans]
            for selected_text in selected_texts:
                self.assertIn(selected_text, output.packed_evidence)
            unselected = [gate.sentence_id for gate in output.gates if not gate.selected]
            for sentence_id in unselected:
                sentence_text = next(
                    sentence.text for sentence in self.index if sentence.sentence_id == sentence_id
                )
                self.assertNotIn(sentence_text, output.packed_evidence)

        loss = run_loss_only_step(model, batch)
        self.assertGreaterEqual(loss.total, 0.0)

    def test_baseline_and_full_input_kd_contracts(self) -> None:
        baseline_scores = LexicalBaselineReranker().score(self.response, self.index)
        self.assertEqual(len(baseline_scores), 3)
        kd = FullInputKDStudent().evaluate_kd_loss(
            teacher_scores=[3.0, 1.0, 0.0],
            feature_scores=[score.score for score in baseline_scores],
        )
        self.assertEqual(len(kd.student_scores), 3)
        self.assertGreaterEqual(kd.kd_loss, 0.0)

    def test_training_schedule_transitions_from_teacher_to_generator_masks(self) -> None:
        warmup = TrainingSchedule.for_step(0, warmup_steps=10, total_steps=100)
        later = TrainingSchedule.for_step(100, warmup_steps=10, total_steps=100)
        self.assertEqual(warmup.teacher_mask_ratio, 1.0)
        self.assertEqual(warmup.generator_mask_ratio, 0.0)
        self.assertEqual(later.teacher_mask_ratio, 0.0)
        self.assertEqual(later.generator_mask_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
