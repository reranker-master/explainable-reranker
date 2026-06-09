from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.evidence_fallback import (
    FALLBACK_SOURCE_TYPE,
    QdrantEvidenceFallback,
    StaticEvidenceFallback,
    augment_with_fallback,
)
from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.topa.adapter import parse_topa_page_response

FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class AugmentWithFallbackTest(unittest.TestCase):
    def _counts(self, payload: dict) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sentence in build_sentence_index(parse_topa_page_response(payload)):
            counts[sentence.book_id] = counts.get(sentence.book_id, 0) + 1
        return counts

    def test_thin_candidate_is_backfilled_only_for_the_shortfall(self) -> None:
        payload = _payload()
        before = self._counts(payload)
        # book_002 has 2 sentences in the fixture; ask for at least 4.
        self.assertEqual(before["book_002"], 2)
        source = StaticEvidenceFallback(
            by_book={"book_002": ["보강 문장 1.", "보강 문장 2.", "보강 문장 3."]}
        )

        augmented = augment_with_fallback(payload, source, min_sentences=4)
        after = self._counts(augmented)
        # exactly the shortfall (4-2=2) is added, not all 3 available.
        self.assertEqual(after["book_002"], 4)
        # candidates already at/above the floor are untouched.
        self.assertEqual(after["book_001"], before["book_001"])
        # backfilled sentences carry the fallback source_type.
        fallback_sentences = [
            s
            for s in build_sentence_index(parse_topa_page_response(augmented))
            if s.source_type == FALLBACK_SOURCE_TYPE
        ]
        self.assertEqual(len(fallback_sentences), 2)

    def test_disabled_when_min_sentences_zero_and_input_untouched(self) -> None:
        payload = _payload()
        source = StaticEvidenceFallback(by_book={"book_002": ["x."]})
        self.assertIs(augment_with_fallback(payload, source, min_sentences=0), payload)
        # non-mutating even when active
        augment_with_fallback(payload, source, min_sentences=5)
        self.assertEqual(self._counts(payload)["book_002"], 2)

    def test_static_from_file(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump({"book_002": ["보강."]}, fh, ensure_ascii=False)
            path = fh.name
        source = StaticEvidenceFallback.from_file(path)
        self.assertEqual(source.fetch("book_002", "q", need=5), ["보강."])
        self.assertEqual(source.fetch("missing", "q", need=5), [])


class QdrantEvidenceFallbackTest(unittest.TestCase):
    def test_requires_executor(self) -> None:
        with self.assertRaises(RuntimeError):
            QdrantEvidenceFallback().fetch("book_1", "q", need=2)

    def test_uses_injected_executor_and_caps_need(self) -> None:
        def executor(collection: str, params: dict) -> list[dict]:
            self.assertEqual(collection, "book_chunks")
            return [{"text": "a"}, {"text": ""}, {"text": "b"}, {"text": "c"}]

        source = QdrantEvidenceFallback(query_executor=executor)
        self.assertEqual(source.fetch("book_1", "q", need=2), ["a", "b"])
        self.assertEqual(source.fetch("book_1", "q", need=0), [])


if __name__ == "__main__":
    unittest.main()
