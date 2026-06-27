"""Load a trained neural select-then-predict model for inference/serving.

The serving layer (:func:`explainable_reranker.serve.api.rerank_payload`) accepts
any :class:`SelectThenPredictModel`, so a trained neural model drops straight in:

    from explainable_reranker.models.select_predict.neural_model import load_neural_model
    model = load_neural_model("checkpoints/neural-v1", "configs/lora_target_modules.yaml")
    rerank_payload(topa_json, model=model)

torch/transformers/peft are imported lazily by the backends, so importing this
module on a CPU-only box never fails.
"""

from __future__ import annotations

from pathlib import Path

from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
    load_lora_config,
)
from explainable_reranker.models.select_predict.model import SelectThenPredictModel


def load_neural_model(
    checkpoint_dir: str | Path,
    lora_config_path: str | Path,
    *,
    device: str | None = None,
    compute_dtype: str = "bfloat16",
    max_length: int = 8192,
    max_selected: int = 3,
    select_fp32: bool = False,
) -> SelectThenPredictModel:  # pragma: no cover - requires torch + checkpoint
    """Reconstruct the trained generator/predictor adapters into a serving model.

    ``select_fp32`` runs the generator's selection encoder in fp32 at inference for fully
    deterministic rationale (no bf16/padding wobble) at ~50% extra latency; off by default.
    """

    lora_config = load_lora_config(lora_config_path)
    common = {"device": device, "compute_dtype": compute_dtype, "max_length": max_length}
    generator = HFSentenceGenerator.from_pretrained(
        checkpoint_dir, lora_config, max_selected=max_selected, select_fp32=select_fp32, **common
    )
    predictor = HFPackedEvidencePredictor.from_pretrained(checkpoint_dir, lora_config, **common)
    return SelectThenPredictModel(generator=generator, predictor=predictor)
