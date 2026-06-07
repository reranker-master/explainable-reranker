"""Adapters for topa.page-style candidate responses."""

from .adapter import EvidenceItem, TopaBookCandidate, TopaPageResponse, parse_topa_page_response
from .client import (
    DummyTopaPageClient,
    HttpTopaPageClient,
    TopaPageClient,
    collect_snapshot,
)

__all__ = [
    "EvidenceItem",
    "TopaBookCandidate",
    "TopaPageResponse",
    "parse_topa_page_response",
    "TopaPageClient",
    "DummyTopaPageClient",
    "HttpTopaPageClient",
    "collect_snapshot",
]
