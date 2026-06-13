"""Record and reuse every cost-incurring external call (topa + the LLM teacher).

This is the durability/replay seam around the two paid boundaries the pipeline
touches:

* :class:`CachingTopaPageClient` wraps any :class:`~explainable_reranker.topa.client.TopaPageClient`
  (the topa.page retrieval endpoint).
* :class:`CachingChatModel` wraps any :class:`~explainable_reranker.teacher.llm_client.ChatModel`
  (Bedrock/Anthropic Opus).

Both decorators do the same three things:

1. **Content-address the request.** The cache key is a SHA-256 of the full input,
   so an identical request never hits the network twice — reruns reuse the saved
   response at zero cost and snapshots stay byte-stable.
2. **Persist the full input *and* output** under ``<cache_dir>/<sha>.json`` so the
   raw exchange can be inspected or re-parsed later without re-calling.
3. **Append one audit line per call** to a shared ``ledger.jsonl`` with timestamp,
   provider, latency, token usage (when the model exposes it), and whether the
   call was served from cache — a flat cost/reuse record.

The wrappers preserve the exact Protocol of what they wrap, so they drop in
without touching the rest of the pipeline and stay fully testable offline.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_ledger(ledger_path: Path | None, entry: dict[str, Any]) -> None:
    """Append one JSON object as a line to the cost/reuse ledger (no-op if unset)."""

    if ledger_path is None:
        return
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@dataclass
class CachingTopaPageClient:
    """Wrap a TopaPageClient so each query's request+response is cached and logged.

    The cache key is the (query, top_k, params) triple, so repeated retrieval of
    the same query is served from disk instead of re-hitting topa.page.
    """

    inner: Any
    cache_dir: Path
    ledger_path: Path | None = None
    now: Callable[[], str] = _utc_now_iso

    def _key(self, query: str, top_k: int | None, params: dict[str, Any] | None) -> str:
        canonical = json.dumps(
            {"query": query, "top_k": top_k, "params": params or {}},
            ensure_ascii=False,
            sort_keys=True,
        )
        return _sha256(canonical)

    def fetch_page(
        self, query: str, *, top_k: int | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        key = self._key(query, top_k, params)
        path = self.cache_dir / f"{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            append_ledger(
                self.ledger_path,
                {
                    "timestamp": self.now(),
                    "provider": "topa",
                    "cache_key": key,
                    "query": query,
                    "cached": True,
                },
            )
            return cached["response"]

        start = time.perf_counter()
        response = self.inner.fetch_page(query, top_k=top_k, params=params)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        _write_cache(
            path,
            {
                "provider": "topa",
                "query": query,
                "top_k": top_k,
                "params": params or {},
                "response": response,
                "fetched_at": self.now(),
                "latency_ms": latency_ms,
            },
        )
        append_ledger(
            self.ledger_path,
            {
                "timestamp": self.now(),
                "provider": "topa",
                "cache_key": key,
                "query": query,
                "cached": False,
                "latency_ms": latency_ms,
            },
        )
        return response


@dataclass
class CachingChatModel:
    """Wrap a ChatModel so each (system, user) -> output exchange is cached and logged.

    Caching makes relabeling idempotent: an identical prompt (same model + system +
    user) replays the saved completion instead of paying for another Opus call. The
    cache file stores the full prompt and completion so labels can be re-derived
    offline, and token usage (when the inner model exposes ``last_usage``) lands in
    the ledger for cost accounting.
    """

    inner: Any
    cache_dir: Path
    model_id: str = "unknown"
    provider: str = "llm"
    ledger_path: Path | None = None
    now: Callable[[], str] = _utc_now_iso
    calls: int = field(default=0)
    cache_hits: int = field(default=0)

    def _key(self, system: str, user: str) -> str:
        return _sha256(f"{self.model_id}\n\x00SYSTEM\x00\n{system}\n\x00USER\x00\n{user}")

    def generate(self, *, system: str, user: str) -> str:
        self.calls += 1
        key = self._key(system, user)
        path = self.cache_dir / f"{key}.json"
        cached = _read_cache(path)
        if cached is not None:
            self.cache_hits += 1
            append_ledger(
                self.ledger_path,
                {
                    "timestamp": self.now(),
                    "provider": self.provider,
                    "model_id": self.model_id,
                    "cache_key": key,
                    "cached": True,
                },
            )
            return cached["output"]

        start = time.perf_counter()
        output = self.inner.generate(system=system, user=user)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        usage = getattr(self.inner, "last_usage", None)
        _write_cache(
            path,
            {
                "provider": self.provider,
                "model_id": self.model_id,
                "system": system,
                "user": user,
                "output": output,
                "usage": usage,
                "created_at": self.now(),
                "latency_ms": latency_ms,
            },
        )
        append_ledger(
            self.ledger_path,
            {
                "timestamp": self.now(),
                "provider": self.provider,
                "model_id": self.model_id,
                "cache_key": key,
                "cached": False,
                "latency_ms": latency_ms,
                "usage": usage,
            },
        )
        return output
