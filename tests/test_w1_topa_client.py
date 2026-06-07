from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from explainable_reranker.data.snapshot_store import SnapshotStore
from explainable_reranker.topa.client import (
    DummyTopaPageClient,
    HttpTopaPageClient,
    collect_snapshot,
)


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class _FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TopaClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_dummy_client_replays_and_records(self) -> None:
        client = DummyTopaPageClient(default=self.payload)
        result = client.fetch_page("잔잔한 가족 이야기", top_k=20, params={"lang": "ko"})
        self.assertEqual(result, self.payload)
        self.assertEqual(client.calls[0], ("잔잔한 가족 이야기", 20, {"lang": "ko"}))

    def test_dummy_client_raises_for_unmapped_query(self) -> None:
        with self.assertRaises(KeyError):
            DummyTopaPageClient().fetch_page("없는 쿼리")

    def test_http_client_builds_request_and_parses(self) -> None:
        captured: dict = {}

        def fake_opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["auth"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _FakeHttpResponse(json.dumps(self.payload).encode("utf-8"))

        client = HttpTopaPageClient(
            "https://topa.example.com/",
            path="/search/page",
            auth_token="secret",
            timeout=12.0,
            opener=fake_opener,
        )
        result = client.fetch_page("위로되는 책", top_k=50)

        self.assertEqual(result, self.payload)
        self.assertEqual(captured["url"], "https://topa.example.com/search/page")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["body"], {"query": "위로되는 책", "top_k": 50})
        self.assertEqual(captured["auth"], "Bearer secret")
        self.assertEqual(captured["timeout"], 12.0)

    def test_collect_snapshot_persists_immutable_raw(self) -> None:
        client = DummyTopaPageClient(default=self.payload)
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(tmp)
            record, response = collect_snapshot(client, store, "잔잔한 가족 이야기", top_k=30)

            self.assertEqual(response.response_id, record.response_id)
            self.assertTrue(response.candidates)
            # Round-trips through the immutable store under the schema version dir.
            reloaded = store.load(record.schema_version, record.response_id)
            self.assertEqual(reloaded.response_id, response.response_id)
            self.assertTrue(record.payload_sha256)


if __name__ == "__main__":
    unittest.main()
