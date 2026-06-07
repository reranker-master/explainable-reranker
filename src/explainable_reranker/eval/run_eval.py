from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.eval.faithfulness import set_f1, set_iou
from explainable_reranker.eval.ir_metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class PredictionItem:
    book_id: str
    score: float
    rationale_sentence_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryQrels:
    query_id: str
    relevance_by_book: dict[str, float]
    rationale_ids_by_book: dict[str, set[str]]


@dataclass(frozen=True)
class EvaluationReport:
    ndcg_at_1: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    recall_at_10: float
    rationale_f1: float
    rationale_iou: float


def evaluate_predictions(
    qrels_by_query: dict[str, QueryQrels],
    predictions_by_query: dict[str, list[PredictionItem]],
) -> EvaluationReport:
    ndcg1: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    rationale_f1s: list[float] = []
    rationale_ious: list[float] = []

    for query_id, qrels in qrels_by_query.items():
        predictions = sorted(
            predictions_by_query.get(query_id, []),
            key=lambda item: item.score,
            reverse=True,
        )
        ranked_book_ids = [prediction.book_id for prediction in predictions]
        ndcg1.append(ndcg_at_k(qrels.relevance_by_book, ranked_book_ids, k=1))
        ndcg5.append(ndcg_at_k(qrels.relevance_by_book, ranked_book_ids, k=5))
        ndcg10.append(ndcg_at_k(qrels.relevance_by_book, ranked_book_ids, k=10))
        mrrs.append(mean_reciprocal_rank(qrels.relevance_by_book, ranked_book_ids, threshold=1.0))
        recalls.append(recall_at_k(qrels.relevance_by_book, ranked_book_ids, k=10, threshold=1.0))

        for prediction in predictions:
            gold = qrels.rationale_ids_by_book.get(prediction.book_id)
            if gold is None:
                continue
            predicted = set(prediction.rationale_sentence_ids)
            rationale_f1s.append(set_f1(predicted, gold))
            rationale_ious.append(set_iou(predicted, gold))

    return EvaluationReport(
        ndcg_at_1=_mean(ndcg1),
        ndcg_at_5=_mean(ndcg5),
        ndcg_at_10=_mean(ndcg10),
        mrr=_mean(mrrs),
        recall_at_10=_mean(recalls),
        rationale_f1=_mean(rationale_f1s),
        rationale_iou=_mean(rationale_ious),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
