from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.distill.dataset import build_training_batch
from explainable_reranker.teacher.hard_negatives import (
    REASON_SAME_GENRE_DIFFERENT_MOOD,
    REASON_TITLE_VARIANT,
    HardNegative,
    MemgraphHardNegativeSource,
    StaticHardNegativeSource,
    hard_label_map,
    inject_hard_negatives,
)
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import (
    HeuristicRationaleTeacher,
    merge_ranking_and_rationales,
)
from explainable_reranker.teacher.prompts import SYSTEM_INSTRUCTIONS, build_listwise_prompt
from explainable_reranker.teacher.schemas import parse_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response

FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _negatives() -> list[HardNegative]:
    return [
        HardNegative(
            book_id="book_900",
            title="검은 항구의 비명",  # 같은 미스터리 장르, 정반대 무드(공포)
            reason=REASON_SAME_GENRE_DIFFERENT_MOOD,
            evidence=("항구를 배경으로 한 잔혹한 연쇄 살인이 독자를 압박한다.",),
        ),
        HardNegative(
            book_id="book_001_set",
            title="아버지의 정원 (개정판)",  # 제목정규화로 잡히는 세트/개정판 중복
            reason=REASON_TITLE_VARIANT,
            evidence=("아버지의 정원 합본 개정판.",),
        ),
    ]


class InjectHardNegativesTest(unittest.TestCase):
    def test_injection_marks_candidates_and_preserves_originals(self) -> None:
        augmented = inject_hard_negatives(_payload(), _negatives())
        response = parse_topa_page_response(augmented)

        by_id = {c.book_id: c for c in response.candidates}
        self.assertEqual(len(response.candidates), 5)  # 3 real + 2 injected
        self.assertFalse(by_id["book_001"].is_hard_negative)
        self.assertTrue(by_id["book_900"].is_hard_negative)
        self.assertEqual(by_id["book_900"].hard_negative_reason, REASON_SAME_GENRE_DIFFERENT_MOOD)
        self.assertEqual(by_id["book_001_set"].hard_negative_reason, REASON_TITLE_VARIANT)
        # injected evidence is indexed like any other candidate's sentences
        index = build_sentence_index(response)
        self.assertTrue(any(s.book_id == "book_900" for s in index))

    def test_injection_does_not_mutate_input_or_relabel_real_candidates(self) -> None:
        payload = _payload()
        # a "negative" colliding with a real in-pool book must be skipped, not relabel it
        collision = [HardNegative(book_id="book_001", title="dup", reason=REASON_TITLE_VARIANT)]
        augmented = inject_hard_negatives(payload, collision)
        self.assertEqual(len(payload["candidates"]), 3)  # input untouched
        self.assertEqual(len(augmented["candidates"]), 3)  # collision skipped
        response = parse_topa_page_response(augmented)
        self.assertFalse(any(c.is_hard_negative for c in response.candidates))

    def test_max_negatives_caps_injection(self) -> None:
        augmented = inject_hard_negatives(_payload(), _negatives(), max_negatives=1)
        response = parse_topa_page_response(augmented)
        self.assertEqual(sum(c.is_hard_negative for c in response.candidates), 1)


class HardLabelFlowTest(unittest.TestCase):
    def test_hard_labels_reach_training_batch(self) -> None:
        augmented = inject_hard_negatives(_payload(), _negatives())
        response = parse_topa_page_response(augmented)
        index = build_sentence_index(response)

        ranking = HeuristicRankingTeacher().label(response, index)
        ranked_ids = [item["book"] for item in ranking["ranking"]]
        rationale = HeuristicRationaleTeacher().label(response, index, ranked_ids)
        label = parse_teacher_label(
            merge_ranking_and_rationales(ranking, rationale),
            query_id=response.query_id,
            response_id=response.response_id,
        )

        hard_labels = hard_label_map(response)
        self.assertEqual(hard_labels, {"book_900": 0, "book_001_set": 0})

        batch = build_training_batch(response, index, label, hard_labels=hard_labels)
        by_id = {c.book_id: c for c in batch.candidates}
        self.assertEqual(by_id["book_900"].hard_label, 0)
        self.assertEqual(by_id["book_001_set"].hard_label, 0)
        self.assertIsNone(by_id["book_001"].hard_label)


class HardNegativeSourceTest(unittest.TestCase):
    def test_static_source_default_applies_to_any_query(self) -> None:
        source = StaticHardNegativeSource(default=_negatives())
        negatives = source.fetch("any query", _payload())
        self.assertEqual([n.book_id for n in negatives], ["book_900", "book_001_set"])

    def test_static_source_from_file(self) -> None:
        import tempfile

        spec = {
            "잔잔하고 위로되는 가족 이야기": [
                {
                    "book_id": "book_900",
                    "title": "검은 항구의 비명",
                    "reason": REASON_SAME_GENRE_DIFFERENT_MOOD,
                    "evidence": "항구 배경의 잔혹한 연쇄 살인.",
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False)
            file_path = fh.name
        source = StaticHardNegativeSource.from_file(file_path)
        negatives = source.fetch("잔잔하고 위로되는 가족 이야기", _payload())
        self.assertEqual(negatives[0].book_id, "book_900")
        self.assertEqual(negatives[0].evidence, ("항구 배경의 잔혹한 연쇄 살인.",))
        self.assertEqual(source.fetch("unmapped query", _payload()), [])

    def test_memgraph_source_requires_executor(self) -> None:
        with self.assertRaises(RuntimeError):
            MemgraphHardNegativeSource().fetch("q", _payload())

    def test_memgraph_source_uses_injected_executor(self) -> None:
        def executor(cypher: str, params: dict) -> list[dict]:
            if "MOOD" in cypher:
                return [{"book_id": "mg_1", "title": "동일장르 다른무드", "synopsis": "..."}]
            return [{"book_id": "mg_2", "title": "개정판", "synopsis": "..."}]

        source = MemgraphHardNegativeSource(query_executor=executor)
        negatives = source.fetch("q", _payload())
        reasons = {n.book_id: n.reason for n in negatives}
        self.assertEqual(reasons["mg_1"], REASON_SAME_GENRE_DIFFERENT_MOOD)
        self.assertEqual(reasons["mg_2"], REASON_TITLE_VARIANT)


class PromptDistractorAwarenessTest(unittest.TestCase):
    def test_prompt_warns_about_hard_negatives(self) -> None:
        self.assertIn("hard negative", SYSTEM_INSTRUCTIONS.lower())
        response = parse_topa_page_response(_payload())
        prompt = build_listwise_prompt(response, build_sentence_index(response))
        self.assertIn("hard negative", prompt.lower())


if __name__ == "__main__":
    unittest.main()
