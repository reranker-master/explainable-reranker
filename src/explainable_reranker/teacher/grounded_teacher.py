from __future__ import annotations

import random
from dataclasses import dataclass, replace

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.teacher.llm_client import ChatModel, extract_json_object
from explainable_reranker.teacher.prompts import (
    SYSTEM_INSTRUCTIONS,
    build_listwise_prompt,
    build_rationale_prompt,
)
from explainable_reranker.teacher.schemas import (
    TeacherLabel,
    parse_teacher_label,
    validate_teacher_label,
)
from explainable_reranker.topa.adapter import TopaPageResponse


class TeacherLabelingError(RuntimeError):
    """Raised when the LLM teacher cannot produce a valid label after retries."""


@dataclass(frozen=True)
class GroundedTeacherConfig:
    top_k_rationale: int = 10
    max_sentences_per_book: int = 16
    max_retries: int = 2


class LLMGroundedTeacher:
    """2-pass grounded teacher backed by a :class:`ChatModel`.

    Pass A produces a listwise ranking; Pass B produces grounded rationale
    sentence IDs for the top-ranked books. Each pass is parsed, merged, and
    validated against the candidate/sentence universe so malformed or
    hallucinated IDs raise instead of silently entering the label set.

    The ChatModel seam keeps this fully testable offline: inject
    ``ScriptedChatModel`` for tests and a real client in production —
    ``AnthropicClaudeChatModel`` by default (see ``scripts/collect_and_label.py``),
    or ``BedrockClaudeChatModel`` where Bedrock is used instead.
    """

    def __init__(self, chat_model: ChatModel, config: GroundedTeacherConfig | None = None):
        self.chat_model = chat_model
        self.config = config or GroundedTeacherConfig()

    def label(
        self,
        response: TopaPageResponse,
        sentence_index: list[IndexedSentence],
    ) -> TeacherLabel:
        ranking_payload = self._complete_json(
            build_listwise_prompt(
                response,
                sentence_index,
                max_sentences_per_book=self.config.max_sentences_per_book,
            )
        )
        ranked_book_ids = _ranked_book_ids(ranking_payload)

        rationale_payload = self._complete_json(
            build_rationale_prompt(
                response,
                sentence_index,
                ranked_book_ids=ranked_book_ids,
                top_k=self.config.top_k_rationale,
                max_sentences_per_book=self.config.max_sentences_per_book,
            )
        )

        # The model is asked to return ranking sorted by descending score, but it
        # does not always order the array perfectly even when every score itself is
        # valid. Normalize the order here so an otherwise-grounded label is not
        # rejected over presentation; scores are untouched, and training re-sorts by
        # score anyway, so this changes no signal.
        merged = {
            "ranking": _sorted_ranking(ranking_payload.get("ranking", [])),
            # In-pool hard negatives come from Pass A (the teacher sees the whole pool
            # there); carry them onto the final label for the §2 anchor loss.
            "hard_negatives": ranking_payload.get("hard_negatives", {}),
            # Some models (e.g. DeepSeek) cite a sentence by its trailing hash segment
            # instead of the full colon-delimited id; resolve those back to canonical
            # ids so a correct grounding is not rejected on formatting.
            "rationales": _resolve_rationale_ids(
                rationale_payload.get("rationales", {}), sentence_index
            ),
        }
        label = parse_teacher_label(
            merged,
            query_id=response.query_id,
            response_id=response.response_id,
        )
        errors = validate_teacher_label(
            label,
            candidate_book_ids={candidate.book_id for candidate in response.candidates},
            sentence_ids_by_book=_sentence_ids_by_book(sentence_index),
            require_rationales_for_top_k=self.config.top_k_rationale,
        )
        if errors:
            raise TeacherLabelingError("; ".join(errors))
        return label

    def label_with_self_consistency(
        self,
        response: TopaPageResponse,
        sentence_index: list[IndexedSentence],
        *,
        runs: int = 3,
        seed: int = 0,
    ) -> list[TeacherLabel]:
        """Relabel the same query with shuffled candidate order for §1.4 kappa.

        Candidate order is permuted per run so that agreement across runs
        measures teacher stability rather than position bias. Results feed
        ``teacher.agreement.self_consistency_report``.
        """

        rng = random.Random(seed)
        labels: list[TeacherLabel] = []
        for _ in range(max(1, runs)):
            order = list(response.candidates)
            rng.shuffle(order)
            shuffled = replace(response, candidates=tuple(order))
            labels.append(self.label(shuffled, sentence_index))
        return labels

    def _complete_json(self, prompt: str) -> dict:
        last_error: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            text = self.chat_model.generate(system=SYSTEM_INSTRUCTIONS, user=prompt)
            try:
                return extract_json_object(text)
            except ValueError as exc:
                last_error = exc
        raise TeacherLabelingError(f"chat model returned no parseable JSON: {last_error}")


def _ranking_score(item: dict) -> float:
    try:
        return float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sorted_ranking(items: list) -> list:
    """Stable-sort ranking entries by descending score, leaving non-dicts in place.

    Only well-formed ``{"book": ..., "score": ...}`` dicts are reordered; anything
    else is passed through untouched so ``parse_teacher_label`` still raises on it.
    """

    dict_items = [item for item in items if isinstance(item, dict)]
    if len(dict_items) != len(items):
        return items
    return sorted(dict_items, key=_ranking_score, reverse=True)


def _ranked_book_ids(ranking_payload: dict) -> list[str]:
    items = ranking_payload.get("ranking", [])
    scored = []
    for item in items:
        if not isinstance(item, dict):
            continue
        book_id = str(item.get("book") or item.get("book_id") or "")
        if not book_id:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scored.append((score, book_id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [book_id for _score, book_id in scored]


def _resolve_rationale_ids(rationales: dict, sentence_index: list[IndexedSentence]) -> dict:
    """Map abbreviated rationale sentence ids back to their canonical full ids.

    Models sometimes cite only the trailing hash segment (``76537e3d``) or a tail of
    the full id (``...:6:76537e3d``) instead of the whole colon-delimited string. For
    each cited id not already known, resolve it against the book's own sentence ids by
    exact, last-segment, or suffix match; only a *unique* match is rewritten, so an
    ambiguous or hallucinated id still fails validation downstream.
    """

    ids_by_book: dict[str, list[str]] = {}
    for sentence in sentence_index:
        ids_by_book.setdefault(sentence.book_id, []).append(sentence.sentence_id)

    resolved: dict = {}
    for book_id, payload in rationales.items():
        if not isinstance(payload, dict):
            resolved[book_id] = payload
            continue
        known = ids_by_book.get(str(book_id), [])
        known_set = set(known)
        new_ids: list[str] = []
        for raw in payload.get("sentence_ids", []):
            cid = str(raw)
            if cid in known_set:
                new_ids.append(cid)
                continue
            matches = [
                k for k in known
                if k.rsplit(":", 1)[-1] == cid or k.endswith(":" + cid)
            ]
            unique = list(dict.fromkeys(matches))
            new_ids.append(unique[0] if len(unique) == 1 else cid)
        resolved[book_id] = {**payload, "sentence_ids": new_ids}
    return resolved


def _sentence_ids_by_book(sentence_index: list[IndexedSentence]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for sentence in sentence_index:
        grouped.setdefault(sentence.book_id, set()).add(sentence.sentence_id)
    return grouped
