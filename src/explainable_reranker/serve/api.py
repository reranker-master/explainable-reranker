from __future__ import annotations

from dataclasses import asdict
from typing import Any

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.distill.dataset import CandidateTrainingExample, QueryTrainingBatch, SentenceTrainingLabel
from explainable_reranker.explain.reason_builder import build_reason
from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.topa.adapter import TopaPageResponse, parse_topa_page_response


def rerank_payload(payload: dict[str, Any], *, model: SelectThenPredictModel | None = None) -> dict[str, Any]:
    """Return a drop-in rerank response with spans and grounded reason fields."""

    response = parse_topa_page_response(payload)
    sentence_index = build_sentence_index(response)
    batch = _batch_from_response(response, sentence_index)
    active_model = model or SelectThenPredictModel()
    outputs = active_model.rerank_batch(batch)
    candidate_by_id = {candidate.book_id: candidate for candidate in response.candidates}

    return {
        "response_id": response.response_id,
        "query_id": response.query_id,
        "query": response.query,
        "schema_version": "explainable-reranker.rerank.v1",
        "results": [
            {
                "book_id": output.book_id,
                "title": candidate_by_id[output.book_id].title,
                "score": round(output.score, 6),
                "rationale_sentence_ids": list(output.rationale_sentence_ids),
                "spans": [
                    {
                        "sentence_id": span.sentence_id,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                        "text": span.text,
                        "token_offsets": [asdict(token) for token in span.token_offsets],
                    }
                    for span in output.spans
                ],
                "reason": build_reason(response.query, output.spans),
            }
            for output in outputs
        ],
    }


def _batch_from_response(response: TopaPageResponse, sentence_index) -> QueryTrainingBatch:
    sentences_by_book = {}
    for sentence in sentence_index:
        sentences_by_book.setdefault(sentence.book_id, []).append(sentence)
    examples = []
    for candidate in response.candidates:
        examples.append(
            CandidateTrainingExample(
                book_id=candidate.book_id,
                title=candidate.title,
                teacher_score=0.0,
                sentences=tuple(
                    SentenceTrainingLabel(sentence=sentence, selected=0)
                    for sentence in sentences_by_book.get(candidate.book_id, [])
                ),
            )
        )
    return QueryTrainingBatch(
        query_id=response.query_id,
        response_id=response.response_id,
        query=response.query,
        candidates=tuple(examples),
    )
