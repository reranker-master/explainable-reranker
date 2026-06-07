from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.serve.api import rerank_payload


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class ServeSmokeTest(unittest.TestCase):
    def test_rerank_payload_returns_spans_and_reasons(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        response = rerank_payload(payload)

        self.assertEqual(response["schema_version"], "explainable-reranker.rerank.v1")
        self.assertEqual(response["query_id"], "q_demo_001")
        self.assertEqual(len(response["results"]), 3)
        for result in response["results"]:
            self.assertIn("score", result)
            self.assertGreaterEqual(len(result["rationale_sentence_ids"]), 1)
            self.assertGreaterEqual(len(result["spans"]), 1)
            self.assertIn("reason", result)
            for span in result["spans"]:
                self.assertIn(span["sentence_id"], result["rationale_sentence_ids"])
                self.assertGreater(span["char_end"], span["char_start"])
                self.assertIn(span["text"], result["reason"])


if __name__ == "__main__":
    unittest.main()
