from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.teacher.agreement import self_consistency_report
from explainable_reranker.teacher.grounded_teacher import (
    GroundedTeacherConfig,
    LLMGroundedTeacher,
    TeacherLabelingError,
)
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import HeuristicRationaleTeacher
from explainable_reranker.teacher.llm_client import (
    BedrockClaudeChatModel,
    ScriptedChatModel,
    extract_json_object,
)
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class _FakeBedrockClient:
    """Stands in for a boto3 bedrock-runtime client to exercise the prod path."""

    def __init__(self, text: str):
        self.text = text
        self.last_kwargs: dict | None = None

    def invoke_model(self, **kwargs):
        self.last_kwargs = kwargs
        body = json.dumps({"content": [{"type": "text", "text": self.text}]}).encode("utf-8")
        return {"body": io.BytesIO(body)}


class LLMTeacherTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)

    def _grounded_chat_model(self) -> ScriptedChatModel:
        """A dummy LLM that returns grounded JSON, simulating Bedrock Claude.

        It branches on the 2-pass prompt and reuses the heuristic labelers so the
        synthesized output always cites real sentence IDs for the current order.
        """

        def respond(system: str, user: str) -> str:
            if "Task A:" in user:
                ranking = HeuristicRankingTeacher().label(self.response, self.index)
                return "```json\n" + json.dumps(ranking, ensure_ascii=False) + "\n```"
            ranked = [item["book"] for item in HeuristicRankingTeacher().label(self.response, self.index)["ranking"]]
            rationale = HeuristicRationaleTeacher().label(self.response, self.index, ranked)
            return "Here is the grounded output: " + json.dumps(rationale, ensure_ascii=False)

        return ScriptedChatModel(respond)

    def test_grounded_teacher_produces_valid_label(self) -> None:
        teacher = LLMGroundedTeacher(
            self._grounded_chat_model(),
            GroundedTeacherConfig(top_k_rationale=3),
        )
        label = teacher.label(self.response, self.index)

        candidate_ids = {candidate.book_id for candidate in self.response.candidates}
        self.assertTrue(set(label.score_by_book()).issubset(candidate_ids))
        for book_id in label.ranked_book_ids()[:3]:
            self.assertIn(book_id, label.rationales)
            self.assertTrue(label.rationales[book_id].sentence_ids)

    def test_extract_json_object_handles_prose_and_fences(self) -> None:
        self.assertEqual(extract_json_object('prefix {"a": 1} suffix'), {"a": 1})
        self.assertEqual(extract_json_object('```json\n{"b": 2}\n```'), {"b": 2})
        with self.assertRaises(ValueError):
            extract_json_object("no json here")

    def test_retries_until_parseable_then_raises(self) -> None:
        # First two responses are garbage, third is valid JSON ranking.
        ranking = HeuristicRankingTeacher().label(self.response, self.index)
        chat = ScriptedChatModel(["nonsense", "still nonsense", json.dumps(ranking)])
        teacher = LLMGroundedTeacher(chat, GroundedTeacherConfig(max_retries=2, top_k_rationale=3))
        payload = teacher._complete_json("Task A: rank")
        self.assertIn("ranking", payload)

        broken = ScriptedChatModel(["nope", "nope", "nope"])
        broken_teacher = LLMGroundedTeacher(broken, GroundedTeacherConfig(max_retries=2))
        with self.assertRaises(TeacherLabelingError):
            broken_teacher._complete_json("Task A: rank")

    def test_self_consistency_labels_feed_agreement(self) -> None:
        teacher = LLMGroundedTeacher(
            self._grounded_chat_model(),
            GroundedTeacherConfig(top_k_rationale=3),
        )
        labels = teacher.label_with_self_consistency(self.response, self.index, runs=3, seed=7)
        self.assertEqual(len(labels), 3)
        report = self_consistency_report(labels)
        # Deterministic dummy teacher → identical scores across shuffles → passes gate.
        self.assertTrue(report.passed)

    def test_bedrock_adapter_request_and_extraction(self) -> None:
        ranking = HeuristicRankingTeacher().label(self.response, self.index)
        fake = _FakeBedrockClient(json.dumps(ranking))
        model = BedrockClaudeChatModel(client=fake, max_tokens=512)

        body = model._request_body(system="sys", user="hello")
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["messages"][0]["content"][0]["text"], "hello")

        text = model.generate(system="sys", user="hello")
        self.assertEqual(extract_json_object(text), ranking)
        self.assertEqual(fake.last_kwargs["modelId"], BedrockClaudeChatModel.DEFAULT_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
