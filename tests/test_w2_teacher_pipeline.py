from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.teacher.agreement import self_consistency_report, weighted_kappa
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import HeuristicRationaleTeacher, merge_ranking_and_rationales
from explainable_reranker.teacher.prompts import build_listwise_prompt, build_rationale_prompt
from explainable_reranker.teacher.schemas import parse_teacher_label, validate_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class TeacherPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)

    def test_prompts_are_grounded_by_sentence_ids(self) -> None:
        prompt = build_listwise_prompt(self.response, self.index)
        self.assertIn("[QUERY] 잔잔하고 위로되는 가족 이야기", prompt)
        self.assertIn(self.index[0].sentence_id, prompt)
        self.assertIn("strict JSON", prompt)

        rationale_prompt = build_rationale_prompt(
            self.response,
            self.index,
            ranked_book_ids=["book_001", "book_002"],
            top_k=2,
        )
        self.assertIn("Select 1 to 3 sentence IDs", rationale_prompt)
        self.assertIn("[BOOK book_001]", rationale_prompt)

    def test_heuristic_teacher_output_validates(self) -> None:
        ranking_payload = HeuristicRankingTeacher().label(self.response, self.index)
        ranked_book_ids = [item["book"] for item in ranking_payload["ranking"]]
        rationale_payload = HeuristicRationaleTeacher().label(self.response, self.index, ranked_book_ids)
        label_payload = merge_ranking_and_rationales(ranking_payload, rationale_payload)
        label = parse_teacher_label(
            label_payload,
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )

        sentence_ids_by_book: dict[str, set[str]] = {}
        for sentence in self.index:
            sentence_ids_by_book.setdefault(sentence.book_id, set()).add(sentence.sentence_id)
        errors = validate_teacher_label(
            label,
            candidate_book_ids={candidate.book_id for candidate in self.response.candidates},
            sentence_ids_by_book=sentence_ids_by_book,
            require_rationales_for_top_k=3,
        )
        self.assertEqual(errors, [])

    def test_validation_rejects_unknown_sentence_id(self) -> None:
        label = parse_teacher_label(
            {
                "ranking": [{"book": "book_001", "score": 2.5}],
                "rationales": {"book_001": {"sentence_ids": ["missing"], "reason": "bad"}},
            },
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )
        errors = validate_teacher_label(
            label,
            candidate_book_ids={"book_001"},
            sentence_ids_by_book={"book_001": {self.index[0].sentence_id}},
            require_rationales_for_top_k=1,
        )
        self.assertTrue(any("unknown sentence_id=missing" in error for error in errors))

    def test_agreement_metrics_gate_consistent_labels(self) -> None:
        ranking_payload = {
            "ranking": [
                {"book": "book_001", "score": 2.8},
                {"book": "book_003", "score": 1.2},
                {"book": "book_002", "score": 0.3},
            ],
            "rationales": {
                "book_001": {"sentence_ids": [self.index[0].sentence_id], "reason": "a"},
                "book_003": {"sentence_ids": [self.index[-1].sentence_id], "reason": "b"},
            },
        }
        first = parse_teacher_label(
            ranking_payload,
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )
        second = parse_teacher_label(
            ranking_payload,
            query_id=self.response.query_id,
            response_id=self.response.response_id,
        )
        report = self_consistency_report([first, second])
        self.assertTrue(report.passed)
        self.assertEqual(weighted_kappa([3, 2, 0], [3, 2, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
