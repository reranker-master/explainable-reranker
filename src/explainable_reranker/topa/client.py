from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from explainable_reranker.topa.adapter import TopaPageResponse, parse_topa_page_response

if TYPE_CHECKING:  # avoid a topa <-> data.snapshot_store import cycle at load time
    from explainable_reranker.data.snapshot_store import SnapshotRecord, SnapshotStore


@runtime_checkable
class TopaPageClient(Protocol):
    """External boundary for topa.page candidate retrieval.

    The whole data path depends on this protocol only, so offline pipelines use
    :class:`DummyTopaPageClient` while production uses :class:`HttpTopaPageClient`.
    """

    def fetch_page(self, query: str, *, top_k: int = 60, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the raw topa.page JSON payload for a query."""


@dataclass
class DummyTopaPageClient:
    """Offline topa.page client that replays canned payloads.

    `payloads` maps a query string to a raw response dict. `default` is returned
    for any unmapped query (useful for smoke runs with a single fixture). Every
    request is recorded in :attr:`calls` for assertions.
    """

    payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    default: dict[str, Any] | None = None
    calls: list[tuple[str, int, dict[str, Any]]] = field(default_factory=list)

    def fetch_page(self, query: str, *, top_k: int = 60, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((query, top_k, dict(params or {})))
        if query in self.payloads:
            return self.payloads[query]
        if self.default is not None:
            return self.default
        raise KeyError(f"no canned topa.page payload for query={query!r}")


class HttpTopaPageClient:
    """Real adapter skeleton for the topa.page HTTP endpoint.

    Uses only the stdlib ``urllib`` so the package stays dependency-free. The
    network ``opener`` is injectable, which keeps the request-construction and
    response-parsing logic unit-testable without real sockets. Production simply
    relies on the default ``urllib.request.urlopen``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        path: str = "/page",
        auth_token: str | None = None,
        timeout: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.auth_token = auth_token
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def _build_request(self, query: str, top_k: int, params: dict[str, Any] | None) -> urllib.request.Request:
        body = {"query": query, "top_k": top_k, **(params or {})}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return urllib.request.Request(
            url=self.base_url + self.path,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def fetch_page(self, query: str, *, top_k: int = 60, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request = self._build_request(query, top_k, params)
        with self._opener(request, timeout=self.timeout) as response:
            raw = response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)


def collect_snapshot(
    client: TopaPageClient,
    store: SnapshotStore,
    query: str,
    *,
    top_k: int = 60,
    params: dict[str, Any] | None = None,
    request_timestamp: str | None = None,
) -> tuple[SnapshotRecord, TopaPageResponse]:
    """Fetch a topa.page payload, persist it as an immutable raw snapshot, parse it.

    This is the §9 data-entry seam: request → store raw snapshot (with hash and
    version) → normalized response. The raw payload is never mutated before it is
    saved, preserving reproducibility.
    """

    payload = client.fetch_page(query, top_k=top_k, params=params)
    record = store.save(payload, request_timestamp=request_timestamp)
    response = parse_topa_page_response(payload)
    return record, response
