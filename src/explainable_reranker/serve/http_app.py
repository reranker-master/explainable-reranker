from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.serve.api import rerank_payload


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict[str, Any]


class RerankApp:
    """Drop-in `/rerank` application wrapping :func:`rerank_payload`.

    Routing/serialization live in :meth:`dispatch`, which is pure (no sockets) so
    it is fully unit-testable. The stdlib HTTP server is a thin shell on top. The
    `model` is injectable so a trained generator/predictor can be served; it
    defaults to the lexical stand-in for smoke runs.
    """

    RERANK_PATH = "/rerank"
    HEALTH_PATH = "/healthz"

    def __init__(self, model: SelectThenPredictModel | None = None):
        self.model = model or SelectThenPredictModel()

    def dispatch(self, method: str, path: str, body: bytes | None) -> HttpResult:
        route = path.split("?", 1)[0].rstrip("/") or "/"
        if route == self.HEALTH_PATH:
            if method != "GET":
                return HttpResult(405, {"error": "method not allowed"})
            return HttpResult(200, {"status": "ok", "model": type(self.model).__name__})
        if route == self.RERANK_PATH:
            if method != "POST":
                return HttpResult(405, {"error": "method not allowed"})
            try:
                payload = json.loads((body or b"").decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return HttpResult(400, {"error": "request body must be valid JSON"})
            if not isinstance(payload, dict):
                return HttpResult(400, {"error": "request body must be a JSON object"})
            try:
                response = rerank_payload(payload, model=self.model)
            except (KeyError, ValueError) as exc:
                return HttpResult(422, {"error": f"could not rerank payload: {exc}"})
            return HttpResult(200, response)
        return HttpResult(404, {"error": "not found"})


def make_handler(app: RerankApp) -> type[BaseHTTPRequestHandler]:
    """Bind a :class:`RerankApp` into a BaseHTTPRequestHandler subclass."""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "ExplainableReranker/1.0"

        def _respond(self, result: HttpResult) -> None:
            payload = json.dumps(result.body, ensure_ascii=False).encode("utf-8")
            self.send_response(result.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0) or 0)
            return self.rfile.read(length) if length else b""

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._respond(app.dispatch("GET", self.path, None))

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            self._respond(app.dispatch("POST", self.path, self._read_body()))

        def log_message(self, *args: Any) -> None:  # silence default stderr logging
            return

    return _Handler


def serve(app: RerankApp, *, host: str = "0.0.0.0", port: int = 8080) -> None:  # pragma: no cover
    """Run the blocking HTTP server (production entry point)."""

    server = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    finally:
        server.server_close()
