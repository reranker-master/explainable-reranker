from __future__ import annotations

import json
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.models.select_predict.backends import (
    EvidencePredictorBackend,
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
    LoraConfig,
    SentenceGeneratorBackend,
    gate_outputs_from_logits,
    load_lora_config,
)
from explainable_reranker.models.select_predict.generator import LexicalSentenceGenerator
from explainable_reranker.models.select_predict.model import SelectThenPredictModel
from explainable_reranker.models.select_predict.predictor import PackedEvidencePredictor
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"
LORA_CONFIG = Path(__file__).parents[1] / "configs" / "lora_target_modules.yaml"


class NeuralBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.response = parse_topa_page_response(payload)
        self.index = build_sentence_index(self.response)

    def test_lexical_backends_satisfy_protocol(self) -> None:
        self.assertIsInstance(LexicalSentenceGenerator(), SentenceGeneratorBackend)
        self.assertIsInstance(PackedEvidencePredictor(), EvidencePredictorBackend)

    def test_hf_skeletons_satisfy_protocol(self) -> None:
        config = load_lora_config(LORA_CONFIG)
        self.assertIsInstance(HFSentenceGenerator(config), SentenceGeneratorBackend)
        self.assertIsInstance(HFPackedEvidencePredictor(config), EvidencePredictorBackend)

    def test_shared_gate_policy_respects_min_and_max(self) -> None:
        sentences = [s for s in self.index if s.book_id == self.index[0].book_id]
        logits = [-5.0 for _ in sentences]  # nothing crosses threshold
        gates = gate_outputs_from_logits(sentences, logits, min_selected=1, max_selected=2)
        self.assertEqual(len(gates), len(sentences))
        self.assertEqual(sum(g.selected for g in gates), 1)  # min_selected forces one

        high = [9.0 for _ in sentences]
        capped = gate_outputs_from_logits(sentences, high, min_selected=1, max_selected=2)
        self.assertLessEqual(sum(g.selected for g in capped), 2)
        for gate in capped:
            self.assertTrue(0.0 <= gate.probability <= 1.0)

    def test_load_lora_config_parses_default_allowlist(self) -> None:
        config = load_lora_config(LORA_CONFIG)
        self.assertIsInstance(config, LoraConfig)
        self.assertEqual(config.base_model, "BAAI/bge-reranker-v2-m3")
        self.assertIn("q_proj", config.target_modules)
        self.assertEqual(config.generator_adapter.r, 16)
        self.assertEqual(config.predictor_adapter.alpha, 32)

    def test_load_lora_config_parses_inspected_target_modules(self, ) -> None:
        inspected = (
            "base_model: BAAI/bge-reranker-v2-m3\n"
            "strategy: inspected_attention_projection_modules\n"
            "generator_adapter:\n  r: 32\n  alpha: 64\n  dropout: 0.1\n"
            "predictor_adapter:\n  r: 8\n  alpha: 16\n  dropout: 0.0\n"
            "target_modules:\n"
            "  - encoder.layer.0.attention.self.query\n"
            "  - encoder.layer.0.attention.self.value\n"
        )
        tmp = Path(self._tmpfile())
        tmp.write_text(inspected, encoding="utf-8")
        config = load_lora_config(tmp)
        self.assertEqual(config.target_modules[0], "encoder.layer.0.attention.self.query")
        self.assertEqual(config.generator_adapter.r, 32)
        self.assertEqual(config.predictor_adapter.r, 8)

    def test_hf_backend_raises_without_torch(self) -> None:
        config = load_lora_config(LORA_CONFIG)
        # Offline envs without torch/transformers/peft should fail clearly; GPU
        # environments with the local model cache should produce one logit per
        # sentence instead.
        try:
            logits = HFSentenceGenerator(config).logits("q", list(self.index))
        except (RuntimeError, NotImplementedError):
            return
        self.assertEqual(len(logits), len(self.index))

    def test_model_accepts_swappable_backends(self) -> None:
        model = SelectThenPredictModel(
            generator=LexicalSentenceGenerator(),
            predictor=PackedEvidencePredictor(),
        )
        batch = _batch(self.response, self.index)
        outputs = model.rerank_batch(batch)
        self.assertTrue(outputs)
        self.assertTrue(all(out.rationale_sentence_ids for out in outputs))

    def _tmpfile(self) -> str:
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name


def _batch(response, index):
    from explainable_reranker.distill.dataset import (
        CandidateTrainingExample,
        QueryTrainingBatch,
        SentenceTrainingLabel,
    )

    by_book: dict[str, list] = {}
    for sentence in index:
        by_book.setdefault(sentence.book_id, []).append(sentence)
    examples = tuple(
        CandidateTrainingExample(
            book_id=candidate.book_id,
            title=candidate.title,
            teacher_score=0.0,
            sentences=tuple(SentenceTrainingLabel(sentence=s, selected=0) for s in by_book.get(candidate.book_id, [])),
        )
        for candidate in response.candidates
    )
    return QueryTrainingBatch(
        query_id=response.query_id,
        response_id=response.response_id,
        query=response.query,
        candidates=examples,
    )


if __name__ == "__main__":
    unittest.main()
