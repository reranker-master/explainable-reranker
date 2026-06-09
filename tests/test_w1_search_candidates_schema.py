from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.topa.adapter import parse_topa_page_response
from explainable_reranker.topa.client import HttpTopaPageClient

# Trimmed capture of the live /api/search/search-candidates response (3 candidates).
FIXTURE = Path(__file__).parent / "fixtures" / "topa_search_candidates_response.json"


class _FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SearchCandidatesSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_parses_nested_book_score_and_dict_chunks(self) -> None:
        response = parse_topa_page_response(self.payload)
        self.assertEqual(len(response.candidates), 3)
        first = response.candidates[0]
        # book_id comes from the nested book.isbn, title from book.title.
        self.assertEqual(first.book_id, "9791162200582")
        self.assertEqual(first.title, "문유 3")
        # score comes from retrieval_debug.rrf_score; rank from pre_rerank_rank.
        self.assertIsNotNone(first.score)
        self.assertEqual(first.rank, 1)
        # chunks dict {synopsis, review} becomes evidence items by source_type.
        self.assertEqual({e.source_type for e in first.evidence}, {"synopsis", "review"})

    def test_sentence_index_splits_chunk_paragraphs(self) -> None:
        response = parse_topa_page_response(self.payload)
        index = build_sentence_index(response)
        self.assertTrue(index)
        # The long synopsis/review paragraphs get split into multiple sentences.
        first_book = response.candidates[0].book_id
        self.assertGreater(len([s for s in index if s.book_id == first_book]), 1)
        for sentence in index:
            self.assertIn(sentence.source_type, {"synopsis", "review"})

    def test_http_client_omits_top_k_by_default(self) -> None:
        captured: dict = {}

        def fake_opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeHttpResponse(json.dumps(self.payload).encode("utf-8"))

        client = HttpTopaPageClient(opener=fake_opener)
        client.fetch_page("우주를 다룬 SF 소설 추천")

        self.assertEqual(captured["url"], "https://www.topa.page/api/search/search-candidates")
        self.assertEqual(captured["body"], {"query": "우주를 다룬 SF 소설 추천"})  # no top_k


if __name__ == "__main__":
    unittest.main()
