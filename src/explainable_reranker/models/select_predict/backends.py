from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.distill.gates import HardConcreteGate, hard_select_from_logits
from explainable_reranker.models.select_predict.generator import GateOutput


@runtime_checkable
class SentenceGeneratorBackend(Protocol):
    """Backend contract for the select stage: query+sentences -> per-sentence gates.

    Both the lexical stand-in and the HF/LoRA neural backend implement this, so
    :class:`SelectThenPredictModel` is agnostic to which one is plugged in.
    """

    def select(
        self, query: str, sentences: Sequence[IndexedSentence]
    ) -> list[GateOutput]:
        ...


@runtime_checkable
class EvidencePredictorBackend(Protocol):
    """Backend contract for the predict stage: query+packed evidence -> score."""

    def score(self, query: str, packed_evidence: str) -> float:
        ...


def gate_outputs_from_logits(
    sentences: Sequence[IndexedSentence],
    logits: Sequence[float],
    *,
    gate: HardConcreteGate | None = None,
    min_selected: int = 1,
    max_selected: int | None = 3,
) -> list[GateOutput]:
    """Turn per-sentence logits into gates with the shared selection policy.

    Used by every generator backend (lexical or neural) so that the
    hard-selection rule (threshold, min/max sentences) and the probability
    mapping stay identical regardless of how the logits were produced.
    """

    gate = gate or HardConcreteGate()
    selected = hard_select_from_logits(
        list(logits), threshold=0.0, min_selected=min_selected, max_selected=max_selected
    )
    outputs: list[GateOutput] = []
    for sentence, logit, is_selected in zip(sentences, logits, selected, strict=True):
        outputs.append(
            GateOutput(
                sentence_id=sentence.sentence_id,
                logit=float(logit),
                probability=gate.probability(float(logit)),
                selected=int(is_selected),
            )
        )
    return outputs


# ---------------------------------------------------------------------------
# LoRA configuration (read by the neural backends; written by inspect script)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


@dataclass(frozen=True)
class LoraConfig:
    base_model: str
    strategy: str
    target_modules: tuple[str, ...]
    generator_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    predictor_adapter: AdapterConfig = field(default_factory=AdapterConfig)


def load_lora_config(path: str | Path) -> LoraConfig:
    """Load the LoRA target config produced in W1.

    Supports both the inspected layout (``target_modules:`` concrete names) and
    the default allowlist layout (``target_module_patterns:``). Parsing is a tiny
    purpose-built reader so the package stays dependency-free (no PyYAML).
    """

    data = _parse_simple_yaml(Path(path).read_text(encoding="utf-8"))
    targets = data.get("target_modules") or data.get("target_module_patterns") or []
    if not isinstance(targets, list) or not targets:
        raise ValueError("LoRA config must list target_modules or target_module_patterns")
    return LoraConfig(
        base_model=str(data.get("base_model", "")),
        strategy=str(data.get("strategy", "")),
        target_modules=tuple(str(item) for item in targets),
        generator_adapter=_adapter_from(data.get("generator_adapter")),
        predictor_adapter=_adapter_from(data.get("predictor_adapter")),
    )


def _adapter_from(payload: object) -> AdapterConfig:
    if not isinstance(payload, dict):
        return AdapterConfig()
    return AdapterConfig(
        r=int(payload.get("r", 16)),
        alpha=int(payload.get("alpha", 32)),
        dropout=float(payload.get("dropout", 0.05)),
    )


def _parse_simple_yaml(text: str) -> dict:
    """Parse the constrained YAML subset used by lora_target_modules.yaml.

    Handles: comments, ``key: value`` scalars, one level of nested mappings, and
    ``- item`` lists under a key. Not a general YAML parser.
    """

    root: dict = {}
    current_key: str | None = None
    current_kind: str | None = None  # "map" | "list"
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            if stripped.endswith(":"):
                current_key = stripped[:-1].strip()
                current_kind = None
                root[current_key] = None
            else:
                key, _, value = stripped.partition(":")
                root[key.strip()] = _coerce_scalar(value.strip())
                current_key = None
                current_kind = None
            continue
        # indented child of current_key
        if current_key is None:
            continue
        if stripped.startswith("- "):
            if current_kind != "list":
                root[current_key] = []
                current_kind = "list"
            root[current_key].append(_coerce_scalar(stripped[2:].strip()))
        else:
            if current_kind != "map":
                root[current_key] = {}
                current_kind = "map"
            key, _, value = stripped.partition(":")
            root[current_key][key.strip()] = _coerce_scalar(value.strip())
    return root


def _coerce_scalar(value: str):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Neural (HF + LoRA) backend skeletons — production wiring point
# ---------------------------------------------------------------------------


class HFSentenceGenerator:
    """bge-reranker-v2-m3 + LoRA_g Generator skeleton (§2.1).

    Encodes query+all sentences in a single forward, mean-pools per sentence
    offset range, projects to a logit, and reuses the shared gate policy. torch /
    transformers / peft are imported lazily so importing this module never
    requires them; the offline test-suite exercises only the pure selection
    contract via :func:`gate_outputs_from_logits`.
    """

    def __init__(
        self,
        config: LoraConfig,
        *,
        gate: HardConcreteGate | None = None,
        min_selected: int = 1,
        max_selected: int = 3,
    ):
        self.config = config
        self.gate = gate or HardConcreteGate()
        self.min_selected = min_selected
        self.max_selected = max_selected
        self._model = None
        self._tokenizer = None
        self._head = None

    def _ensure_loaded(self):  # pragma: no cover - requires torch/transformers/peft
        if self._model is not None:
            return
        try:
            import torch.nn as nn
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFSentenceGenerator requires torch, transformers, and peft in the "
                "training environment."
            ) from exc
        base = AutoModel.from_pretrained(self.config.base_model)
        peft_config = PeftLoraConfig(
            r=self.config.generator_adapter.r,
            lora_alpha=self.config.generator_adapter.alpha,
            lora_dropout=self.config.generator_adapter.dropout,
            target_modules=list(self.config.target_modules),
        )
        self._model = get_peft_model(base, peft_config)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self._head = nn.Linear(base.config.hidden_size, 1)

    def logits(self, query: str, sentences: Sequence[IndexedSentence]) -> list[float]:  # pragma: no cover
        self._ensure_loaded()
        raise NotImplementedError(
            "Wire the single-forward encode + per-sentence mean-pool here once GPU "
            "training is enabled; see plan §2.1."
        )

    def select(self, query: str, sentences: Sequence[IndexedSentence]) -> list[GateOutput]:  # pragma: no cover
        return gate_outputs_from_logits(
            sentences,
            self.logits(query, sentences),
            gate=self.gate,
            min_selected=self.min_selected,
            max_selected=self.max_selected,
        )


class HFPackedEvidencePredictor:
    """bge-reranker-v2-m3 + LoRA_p Predictor skeleton (§2.1).

    Receives only physically packed selected evidence and emits a relevance
    score from the [CLS] representation. Same lazy-import discipline as the
    generator skeleton.
    """

    def __init__(self, config: LoraConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):  # pragma: no cover - requires torch/transformers/peft
        if self._model is not None:
            return
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFPackedEvidencePredictor requires torch, transformers, and peft in "
                "the training environment."
            ) from exc
        base = AutoModelForSequenceClassification.from_pretrained(self.config.base_model)
        peft_config = PeftLoraConfig(
            r=self.config.predictor_adapter.r,
            lora_alpha=self.config.predictor_adapter.alpha,
            lora_dropout=self.config.predictor_adapter.dropout,
            target_modules=list(self.config.target_modules),
        )
        self._model = get_peft_model(base, peft_config)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

    def score(self, query: str, packed_evidence: str) -> float:  # pragma: no cover
        self._ensure_loaded()
        raise NotImplementedError(
            "Wire the query+packed-evidence cross-encoder forward here once GPU "
            "training is enabled; see plan §2.1."
        )
