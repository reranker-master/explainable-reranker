"""Adapters for topa.page-style candidate responses."""

from .adapter import EvidenceItem, TopaBookCandidate, TopaPageResponse, parse_topa_page_response

__all__ = [
    "EvidenceItem",
    "TopaBookCandidate",
    "TopaPageResponse",
    "parse_topa_page_response",
]
