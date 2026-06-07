from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from explainable_reranker.topa.adapter import TopaBookCandidate, TopaPageResponse


@dataclass(frozen=True)
class TokenOffset:
    token_index: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class IndexedSentence:
    sentence_id: str
    response_id: str
    book_id: str
    source_type: str
    source_id: str
    sent_idx: int
    text: str
    text_hash: str
    char_start: int
    char_end: int
    token_offsets: tuple[TokenOffset, ...]


def build_sentence_index(response: TopaPageResponse) -> list[IndexedSentence]:
    """Assign stable sentence IDs and offsets to evidence bundled in topa.page JSON."""

    indexed: list[IndexedSentence] = []
    for candidate in response.candidates:
        indexed.extend(_index_candidate(response.response_id, candidate))
    return indexed


def sentence_id(
    response_id: str,
    book_id: str,
    source_type: str,
    source_id: str,
    sent_idx: int,
    text: str,
) -> str:
    digest = text_hash(text)[:8]
    return f"{response_id}:{book_id}:{source_type}:{source_id}:{sent_idx}:{digest}"


def text_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def simple_token_offsets(text: str) -> tuple[TokenOffset, ...]:
    """Return deterministic token offsets without external tokenizer dependencies."""

    offsets: list[TokenOffset] = []
    for idx, match in enumerate(re.finditer(r"\S+", text)):
        offsets.append(
            TokenOffset(
                token_index=idx,
                char_start=match.start(),
                char_end=match.end(),
                text=match.group(0),
            )
        )
    return tuple(offsets)


def split_if_needed(text: str) -> list[str]:
    """Split paragraph text only on the exception path where topa did not send sentences."""

    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= 120:
        return [stripped]
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？다요죠함음])\s+", stripped) if part.strip()]
    return parts or [stripped]


def _index_candidate(response_id: str, candidate: TopaBookCandidate) -> list[IndexedSentence]:
    indexed: list[IndexedSentence] = []
    for evidence_item in candidate.evidence:
        char_cursor = 0
        for sent_idx, sentence_text in enumerate(split_if_needed(evidence_item.text), start=1):
            char_start = evidence_item.text.find(sentence_text, char_cursor)
            if char_start < 0:
                char_start = char_cursor
            char_end = char_start + len(sentence_text)
            char_cursor = char_end
            indexed.append(
                IndexedSentence(
                    sentence_id=sentence_id(
                        response_id,
                        candidate.book_id,
                        evidence_item.source_type,
                        evidence_item.source_id,
                        sent_idx,
                        sentence_text,
                    ),
                    response_id=response_id,
                    book_id=candidate.book_id,
                    source_type=evidence_item.source_type,
                    source_id=evidence_item.source_id,
                    sent_idx=sent_idx,
                    text=sentence_text,
                    text_hash=text_hash(sentence_text),
                    char_start=char_start,
                    char_end=char_end,
                    token_offsets=simple_token_offsets(sentence_text),
                )
            )
    return indexed
