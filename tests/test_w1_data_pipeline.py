from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from explainable_reranker.data.query_synth import generate_synthetic_queries
from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.data.snapshot_store import SnapshotStore
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class DataPipelineTest(unittest.TestCase):
    def test_parse_and_index_topa_response(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        response = parse_topa_page_response(payload)

        self.assertEqual(response.response_id, "resp_demo_001")
        self.assertEqual(len(response.candidates), 3)
        self.assertEqual(response.candidates[0].evidence[0].source_type, "synopsis")

        index = build_sentence_index(response)
        self.assertEqual(len(index), 7)
        first = index[0]
        self.assertTrue(first.sentence_id.startswith("resp_demo_001:book_001:synopsis:syn_001:1:"))
        self.assertEqual(first.char_start, 0)
        self.assertGreater(first.char_end, first.char_start)
        self.assertGreaterEqual(len(first.token_offsets), 5)

    def test_snapshot_store_is_immutable_by_response_id(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore(tmpdir)
            record = store.save(payload, request_timestamp="2026-06-07T00:00:00+00:00")
            same_record = store.save(payload, request_timestamp="2026-06-08T00:00:00+00:00")
            self.assertEqual(record.payload_sha256, same_record.payload_sha256)

            changed = dict(payload)
            changed["query"] = "다른 질문"
            with self.assertRaises(ValueError):
                store.save(changed)

            loaded = store.load("topa.page.v1", "resp_demo_001")
            self.assertEqual(loaded.query_id, "q_demo_001")

    def test_generate_synthetic_queries_balances_families(self) -> None:
        queries = generate_synthetic_queries(25, seed=11)
        self.assertEqual(len(queries), 25)
        self.assertEqual(len({query.text for query in queries}), 25)
        families = {query.family for query in queries}
        self.assertEqual(
            families,
            {"mood", "relationship", "trope", "negative_preference", "composite"},
        )


if __name__ == "__main__":
    unittest.main()
