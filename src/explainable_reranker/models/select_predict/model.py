from __future__ import annotations

from dataclasses import dataclass

from explainable_reranker.data.sentence_index import IndexedSentence, TokenOffset
from explainable_reranker.distill.dataset import CandidateTrainingExample, QueryTrainingBatch
from explainable_reranker.models.select_predict.generator import GateOutput, LexicalSentenceGenerator
from explainable_reranker.models.select_predict.predictor import PackedEvidencePredictor


@dataclass(frozen=True)
class Span:
    sentence_id: str
    char_start: int
    char_end: int
    token_offsets: tuple[TokenOffset, ...]
    text: str


@dataclass(frozen=True)
class RerankOutput:
    book_id: str
    score: float
    rationale_sentence_ids: tuple[str, ...]
    spans: tuple[Span, ...]
    packed_evidence: str
    gates: tuple[GateOutput, ...]


class SelectThenPredictModel:
    """Local select-then-predict model contract.

    The Predictor receives only `packed_evidence`, which is built from selected
    sentence IDs. That physical removal is the core faithfulness invariant.
    """

    def __init__(
        self,
        generator: LexicalSentenceGenerator | None = None,
        predictor: PackedEvidencePredictor | None = None,
    ):
        self.generator = generator or LexicalSentenceGenerator()
        self.predictor = predictor or PackedEvidencePredictor()

    def rerank_batch(self, batch: QueryTrainingBatch) -> list[RerankOutput]:
        outputs = [self.score_candidate(batch.query, candidate) for candidate in batch.candidates]
        return sorted(outputs, key=lambda output: output.score, reverse=True)

    def score_candidate(self, query: str, candidate: CandidateTrainingExample) -> RerankOutput:
        sentences = tuple(label.sentence for label in candidate.sentences)
        gates = tuple(self.generator.select(query, sentences))
        selected_ids = tuple(gate.sentence_id for gate in gates if gate.selected)
        selected_sentences = [sentence for sentence in sentences if sentence.sentence_id in set(selected_ids)]
        packed_evidence = "\n".join(sentence.text for sentence in selected_sentences)
        score = self.predictor.score(query, packed_evidence)
        spans = tuple(_span_from_sentence(sentence) for sentence in selected_sentences)
        return RerankOutput(
            book_id=candidate.book_id,
            score=score,
            rationale_sentence_ids=selected_ids,
            spans=spans,
            packed_evidence=packed_evidence,
            gates=gates,
        )


def _span_from_sentence(sentence: IndexedSentence) -> Span:
    return Span(
        sentence_id=sentence.sentence_id,
        char_start=sentence.char_start,
        char_end=sentence.char_end,
        token_offsets=sentence.token_offsets,
        text=sentence.text,
    )
