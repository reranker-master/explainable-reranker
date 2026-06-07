from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.teacher.schemas import TeacherLabel
from explainable_reranker.topa.adapter import TopaPageResponse


@dataclass(frozen=True)
class SentenceTrainingLabel:
    sentence: IndexedSentence
    selected: int


@dataclass(frozen=True)
class CandidateTrainingExample:
    book_id: str
    title: str
    teacher_score: float
    sentences: tuple[SentenceTrainingLabel, ...]
    hard_label: int | None = None

    def teacher_rationale_ids(self) -> tuple[str, ...]:
        return tuple(label.sentence.sentence_id for label in self.sentences if label.selected)


@dataclass(frozen=True)
class QueryTrainingBatch:
    query_id: str
    response_id: str
    query: str
    candidates: tuple[CandidateTrainingExample, ...]

    def teacher_scores(self) -> list[float]:
        return [candidate.teacher_score for candidate in self.candidates]


def build_training_batch(
    response: TopaPageResponse,
    sentence_index: list[IndexedSentence],
    teacher_label: TeacherLabel,
    *,
    hard_labels: dict[str, int] | None = None,
) -> QueryTrainingBatch:
    scores = teacher_label.score_by_book()
    rationales = teacher_label.rationales
    sentences_by_book: dict[str, list[IndexedSentence]] = {}
    for sentence in sentence_index:
        sentences_by_book.setdefault(sentence.book_id, []).append(sentence)

    examples: list[CandidateTrainingExample] = []
    for candidate in response.candidates:
        selected_ids = set(rationales.get(candidate.book_id).sentence_ids) if candidate.book_id in rationales else set()
        sentence_labels = tuple(
            SentenceTrainingLabel(sentence=sentence, selected=1 if sentence.sentence_id in selected_ids else 0)
            for sentence in sentences_by_book.get(candidate.book_id, [])
        )
        examples.append(
            CandidateTrainingExample(
                book_id=candidate.book_id,
                title=candidate.title,
                teacher_score=scores.get(candidate.book_id, 0.0),
                sentences=sentence_labels,
                hard_label=hard_labels.get(candidate.book_id) if hard_labels else None,
            )
        )

    examples.sort(key=lambda example: example.teacher_score, reverse=True)
    return QueryTrainingBatch(
        query_id=response.query_id,
        response_id=response.response_id,
        query=response.query,
        candidates=tuple(examples),
    )


def pack_selected_evidence(candidate: CandidateTrainingExample, selected_sentence_ids: set[str]) -> str:
    """Physically pack only selected sentences for Predictor input."""

    selected_texts = [
        label.sentence.text for label in candidate.sentences if label.sentence.sentence_id in selected_sentence_ids
    ]
    return "\n".join(selected_texts)
