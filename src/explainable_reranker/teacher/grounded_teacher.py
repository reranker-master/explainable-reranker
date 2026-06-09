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

        merged = {
            "ranking": ranking_payload.get("ranking", []),
            "rationales": rationale_payload.get("rationales", {}),
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


def _sentence_ids_by_book(sentence_index: list[IndexedSentence]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for sentence in sentence_index:
        grouped.setdefault(sentence.book_id, set()).add(sentence.sentence_id)
    return grouped
