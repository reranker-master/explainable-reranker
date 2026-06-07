from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from explainable_reranker.serve.http_app import RerankApp, make_handler


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class ServeHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.app = RerankApp()

    def test_healthcheck(self) -> None:
        result = self.app.dispatch("GET", "/healthz", None)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.body["status"], "ok")

    def test_rerank_route_returns_spans_and_reason(self) -> None:
        body = json.dumps(self.payload).encode("utf-8")
        result = self.app.dispatch("POST", "/rerank", body)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.body["schema_version"], "explainable-reranker.rerank.v1")
        self.assertTrue(result.body["results"])
        for item in result.body["results"]:
            self.assertIn("spans", item)
            self.assertIn("reason", item)

    def test_invalid_json_is_400(self) -> None:
        result = self.app.dispatch("POST", "/rerank", b"{not json")
        self.assertEqual(result.status, 400)

    def test_method_and_route_guards(self) -> None:
        self.assertEqual(self.app.dispatch("GET", "/rerank", None).status, 405)
        self.assertEqual(self.app.dispatch("POST", "/healthz", b"").status, 405)
        self.assertEqual(self.app.dispatch("GET", "/unknown", None).status, 404)

    def test_trailing_slash_and_query_string_normalized(self) -> None:
        self.assertEqual(self.app.dispatch("GET", "/healthz/", None).status, 200)
        self.assertEqual(self.app.dispatch("GET", "/healthz?probe=1", None).status, 200)

    def test_live_server_round_trip(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            request = urllib.request.Request(
                f"http://{host}:{port}/rerank",
                data=json.dumps(self.payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                body = json.loads(response.read().decode("utf-8"))
            self.assertEqual(body["schema_version"], "explainable-reranker.rerank.v1")
            self.assertTrue(body["results"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
