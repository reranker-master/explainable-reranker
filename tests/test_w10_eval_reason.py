from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.distill.dataset import build_training_batch
from explainable_reranker.eval.faithfulness import set_f1, set_iou
from explainable_reranker.eval.ir_metrics import ndcg_at_k, recall_at_k
from explainable_reranker.eval.run_eval import (
    PredictionItem,
    QueryQrels,
    evaluate_predictions,
    load_predictions,
    load_qrels,
    report_to_dict,
)
from explainable_reranker.explain.reason_builder import build_reason
from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import HeuristicRationaleTeacher, merge_ranking_and_rationales
from explainable_reranker.teacher.schemas import parse_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response


ROOT = Path(__file__).parent
TOPA_FIXTURE = ROOT / "fixtures" / "topa_page_response.json"
EVAL_FIXTURE = ROOT / "fixtures" / "human_eval_set.json"


class EvalReasonTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(TOPA_FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)
        ranking_payload = HeuristicRankingTeacher().label(self.response, self.index)
        ranked_book_ids = [item["book"] for item in ranking_payload["ranking"]]
        rationale_payload = HeuristicRationaleTeacher().label(self.response, self.index, ranked_book_ids)
        teacher_label = parse_teacher_label(
            merge_ranking_and_rationales(ranking_payload, rationale_payload),
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )
        self.batch = build_training_batch(self.response, self.index, teacher_label)
        self.outputs = SelectThenPredictModel().rerank_batch(self.batch)

    def test_ir_and_rationale_metrics(self) -> None:
        relevance = {"book_001": 3, "book_003": 2, "book_002": 0}
        ranking = ["book_001", "book_003", "book_002"]
        self.assertEqual(ndcg_at_k(relevance, ranking, k=3), 1.0)
        self.assertEqual(recall_at_k(relevance, ranking, k=1), 0.5)
        self.assertEqual(set_f1({"a", "b"}, {"b", "c"}), 0.5)
        self.assertAlmostEqual(set_iou({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_reason_builder_uses_selected_spans_only(self) -> None:
        top_output = self.outputs[0]
        reason = build_reason(self.response.query, top_output.spans)
        for span in top_output.spans[:2]:
            self.assertIn(span.text, reason)
        unselected_texts = [
            sentence.text
            for sentence in self.index
            if sentence.book_id == top_output.book_id and sentence.sentence_id not in top_output.rationale_sentence_ids
        ]
        for text in unselected_texts:
            self.assertNotIn(text, reason)

    def test_evaluate_predictions_against_independent_qrels(self) -> None:
        fixture = json.loads(EVAL_FIXTURE.read_text(encoding="utf-8"))
        qrels = QueryQrels(
            query_id="q_demo_001",
            relevance_by_book=fixture["q_demo_001"]["relevance_by_book"],
            rationale_ids_by_book={
                output.book_id: set(output.rationale_sentence_ids) for output in self.outputs
            },
        )
        predictions = [
            PredictionItem(
                book_id=output.book_id,
                score=output.score,
                rationale_sentence_ids=output.rationale_sentence_ids,
            )
            for output in self.outputs
        ]
        report = evaluate_predictions({"q_demo_001": qrels}, {"q_demo_001": predictions})
        self.assertGreaterEqual(report.ndcg_at_10, 0.0)
        self.assertEqual(report.rationale_f1, 1.0)
        self.assertEqual(report.rationale_iou, 1.0)


class EvalDriverIOTest(unittest.TestCase):
    """Covers the file loaders that the scripts/evaluate.py CLI is built on."""

    def test_loaders_round_trip_into_evaluate_predictions(self) -> None:
        import tempfile

        qrels_spec = {
            "q1": {
                "relevance_by_book": {"A": 3.0, "B": 0.0},
                "rationale_ids_by_book": {"A": ["s1", "s2"]},
            }
        }
        predictions_spec = {
            "q1": [
                {"book_id": "A", "score": 2.5, "rationale_sentence_ids": ["s1", "s2"]},
                {"book_id": "B", "score": 0.1, "rationale_sentence_ids": []},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            qrels_path = Path(tmp) / "qrels.json"
            predictions_path = Path(tmp) / "predictions.json"
            qrels_path.write_text(json.dumps(qrels_spec), encoding="utf-8")
            predictions_path.write_text(json.dumps(predictions_spec), encoding="utf-8")

            qrels = load_qrels(qrels_path)
            predictions = load_predictions(predictions_path)

        self.assertEqual(qrels["q1"].rationale_ids_by_book["A"], {"s1", "s2"})
        self.assertEqual(predictions["q1"][0].book_id, "A")

        report = evaluate_predictions(qrels, predictions)
        as_dict = report_to_dict(report)
        self.assertIn("ndcg_at_10", as_dict)
        # A (relevant, ranked first by score) → perfect ranking and exact rationale match.
        self.assertEqual(as_dict["ndcg_at_10"], 1.0)
        self.assertEqual(as_dict["rationale_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
