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
# Torch runtime helpers (GB10 / DGX Spark: CUDA + bf16 autocast by default)
# ---------------------------------------------------------------------------


def _import_torch():
    """Lazily import torch so this module imports cleanly without a GPU stack."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only off-GPU
        raise RuntimeError(
            "The neural backend requires torch, transformers, and peft. Install the "
            "training extras: `pip install -e '.[gpu]'` on the DGX Spark."
        ) from exc
    return torch


def _resolve_device(torch, device: str | None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype_from_name(torch, name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported compute_dtype={name!r}")
    return mapping[name]


def _autocast_factory(torch, device, compute_dtype):
    """Return a callable yielding an autocast context (no-op on CPU).

    Master weights stay fp32 (stable LoRA/head optimization); only the backbone
    forward runs in bf16, which is the recommended recipe on Blackwell/GB10.
    """

    import contextlib

    if device.type == "cuda":
        return lambda: torch.autocast("cuda", dtype=compute_dtype)
    return lambda: contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Neural (HF + LoRA) backends — bge-reranker-v2-m3 backbone + 2 LoRA adapters
# ---------------------------------------------------------------------------


class HFSentenceGenerator:
    """bge-reranker-v2-m3 + LoRA_g Generator (plan §2.1).

    Encodes query+all candidate sentences in a *single* forward (sentences may
    cross-attend), mean-pools the backbone hidden states over each sentence's
    character span, projects each pooled vector to a logit π_i, then reuses the
    shared hard-concrete gate/selection policy. The Predictor — not the Generator
    — guarantees faithfulness (physical evidence removal), so cross-attention in
    G is safe and lets it compare which sentence is the better rationale.

    torch / transformers / peft are imported lazily, so importing this module on
    a CPU-only box never fails; the offline test-suite only exercises the pure
    selection contract via :func:`gate_outputs_from_logits`.
    """

    def __init__(
        self,
        config: LoraConfig,
        *,
        gate: HardConcreteGate | None = None,
        min_selected: int = 1,
        max_selected: int = 3,
        device: str | None = None,
        compute_dtype: str = "bfloat16",
        max_length: int = 8192,
        sentence_separator: str = "\n",
    ):
        self.config = config
        self.gate = gate or HardConcreteGate()
        self.min_selected = min_selected
        self.max_selected = max_selected
        self.device = device
        self.compute_dtype = compute_dtype
        self.max_length = max_length
        self.sentence_separator = sentence_separator
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._head = None
        self._device = None
        self._autocast = None
        self.truncated_sentences = 0  # count of sentences pooled from CLS fallback

    def _ensure_loaded(self):  # pragma: no cover - requires torch/transformers/peft
        if self._model is not None:
            return
        torch = _import_torch()
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
        self._torch = torch
        self._device = _resolve_device(torch, self.device)
        self._autocast = _autocast_factory(
            torch, self._device, _dtype_from_name(torch, self.compute_dtype)
        )
        base = AutoModel.from_pretrained(self.config.base_model)
        peft_config = PeftLoraConfig(
            r=self.config.generator_adapter.r,
            lora_alpha=self.config.generator_adapter.alpha,
            lora_dropout=self.config.generator_adapter.dropout,
            target_modules=list(self.config.target_modules),
        )
        self._model = get_peft_model(base, peft_config).to(self._device)
        # Gradient checkpointing: this loop retains one full encoder graph per candidate
        # (~50/step), so the stored O(L^2) attention activations blow up unified memory.
        # Recomputing them in backward keeps the full pool in one step at a fraction of the
        # memory (only active in train mode; eval inference is unaffected). use_reentrant=
        # False + input grads is the recipe that works with a frozen LoRA base.
        self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._model.enable_input_require_grads()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        # fp32 master head for stable optimization regardless of autocast dtype.
        self._head = nn.Linear(base.config.hidden_size, 1).to(self._device)

    def _build_packed_input(
        self, query: str, sentences: Sequence[IndexedSentence]
    ) -> tuple[str, list[tuple[int, int]]]:
        """Concatenate query + sentences into one string, tracking each sentence's
        character span so tokens can be mapped back to sentences after encoding."""

        text = query + self.sentence_separator
        cursor = len(text)
        spans: list[tuple[int, int]] = []
        for idx, sentence in enumerate(sentences):
            start = cursor
            text += sentence.text
            cursor += len(sentence.text)
            spans.append((start, cursor))
            if idx < len(sentences) - 1:
                text += self.sentence_separator
                cursor += len(self.sentence_separator)
        return text, spans

    def _forward_logits(self, query: str, sentences: Sequence[IndexedSentence]):  # pragma: no cover
        """Single-forward encode + per-sentence mean-pool + linear head.

        Returns an fp32 tensor of shape ``[len(sentences)]`` carrying gradients
        when the model is in train mode (used by the distillation loop).
        """

        self._ensure_loaded()
        torch = self._torch
        sentences = list(sentences)
        if not sentences:
            return torch.empty(0, dtype=torch.float32, device=self._device)
        text, spans = self._build_packed_input(query, sentences)
        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._autocast():
            hidden = self._model(**encoded).last_hidden_state[0]  # [seq, H]
        pooled = []
        for span_start, span_end in spans:
            token_indices = [
                token
                for token, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > tok_start and tok_start < span_end and tok_end > span_start
            ]
            if token_indices:
                index = torch.tensor(token_indices, device=self._device)
                pooled.append(hidden.index_select(0, index).mean(dim=0))
            else:
                # Sentence fell outside max_length: fall back to the CLS state and
                # record it so the caller can tighten evidence preselect (plan §1.5).
                self.truncated_sentences += 1
                pooled.append(hidden[0])
        stacked = torch.stack(pooled).float()  # promote to fp32 for the head
        return self._head(stacked).squeeze(-1)

    def logits(self, query: str, sentences: Sequence[IndexedSentence]) -> list[float]:  # pragma: no cover
        self._ensure_loaded()
        torch = self._torch
        was_training = self._model.training
        self._model.eval()
        with torch.no_grad():
            values = self._forward_logits(query, sentences)
        if was_training:
            self._model.train()
        return values.detach().float().cpu().tolist()

    def select(self, query: str, sentences: Sequence[IndexedSentence]) -> list[GateOutput]:  # pragma: no cover
        return gate_outputs_from_logits(
            sentences,
            self.logits(query, sentences),
            gate=self.gate,
            min_selected=self.min_selected,
            max_selected=self.max_selected,
        )

    def trainable_parameters(self):  # pragma: no cover
        """LoRA adapter params + the linear head — the only tensors we optimize."""

        self._ensure_loaded()
        params = [p for p in self._model.parameters() if p.requires_grad]
        params += list(self._head.parameters())
        return params

    def train_mode(self):  # pragma: no cover
        self._ensure_loaded()
        self._model.train()
        self._head.train()

    def eval_mode(self):  # pragma: no cover
        self._ensure_loaded()
        self._model.eval()
        self._head.eval()

    def save_pretrained(self, directory: str | Path) -> Path:  # pragma: no cover
        self._ensure_loaded()
        torch = self._torch
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out / "generator_adapter"))
        torch.save(self._head.state_dict(), out / "generator_head.pt")
        return out

    @classmethod
    def from_pretrained(cls, directory: str | Path, config: LoraConfig, **kwargs):  # pragma: no cover
        instance = cls(config, **kwargs)
        instance._ensure_loaded()
        torch = instance._torch
        from peft import PeftModel
        from transformers import AutoModel

        out = Path(directory)
        base = AutoModel.from_pretrained(config.base_model)
        instance._model = PeftModel.from_pretrained(base, str(out / "generator_adapter")).to(
            instance._device
        )
        instance._head.load_state_dict(
            torch.load(out / "generator_head.pt", map_location=instance._device)
        )
        instance.eval_mode()
        return instance


class HFPackedEvidencePredictor:
    """bge-reranker-v2-m3 + LoRA_p Predictor (plan §2.1).

    Receives only physically packed selected evidence as a (query, evidence)
    cross-encoder pair and emits a single relevance score from the sequence
    classification head. The classification head is trained alongside the LoRA
    adapter (``modules_to_save``) since it produces the final score. Same lazy
    torch-import discipline as the generator.
    """

    def __init__(
        self,
        config: LoraConfig,
        *,
        device: str | None = None,
        compute_dtype: str = "bfloat16",
        max_length: int = 8192,
        head_module_names: tuple[str, ...] = ("classifier",),
    ):
        self.config = config
        self.device = device
        self.compute_dtype = compute_dtype
        self.max_length = max_length
        self.head_module_names = head_module_names
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._autocast = None

    def _ensure_loaded(self):  # pragma: no cover - requires torch/transformers/peft
        if self._model is not None:
            return
        torch = _import_torch()
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFPackedEvidencePredictor requires torch, transformers, and peft in "
                "the training environment."
            ) from exc
        self._torch = torch
        self._device = _resolve_device(torch, self.device)
        self._autocast = _autocast_factory(
            torch, self._device, _dtype_from_name(torch, self.compute_dtype)
        )
        base = AutoModelForSequenceClassification.from_pretrained(self.config.base_model)
        peft_config = PeftLoraConfig(
            r=self.config.predictor_adapter.r,
            lora_alpha=self.config.predictor_adapter.alpha,
            lora_dropout=self.config.predictor_adapter.dropout,
            target_modules=list(self.config.target_modules),
            modules_to_save=list(self.head_module_names),
        )
        self._model = get_peft_model(base, peft_config).to(self._device)
        # See HFSentenceGenerator: recompute encoder activations in backward so a full
        # 50-candidate step fits in unified memory (train mode only).
        self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._model.enable_input_require_grads()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

    def _forward_score(self, query: str, packed_evidence: str):  # pragma: no cover
        """Cross-encoder forward over (query, packed evidence) -> scalar score tensor."""

        self._ensure_loaded()
        torch = self._torch
        if not packed_evidence.strip():
            return torch.zeros((), dtype=torch.float32, device=self._device)
        encoded = self._tokenizer(
            query,
            packed_evidence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._autocast():
            logits = self._model(**encoded).logits  # [1, num_labels] (num_labels=1 for rerankers)
        return logits.float().reshape(-1)[0]

    def score(self, query: str, packed_evidence: str) -> float:  # pragma: no cover
        self._ensure_loaded()
        torch = self._torch
        was_training = self._model.training
        self._model.eval()
        with torch.no_grad():
            value = self._forward_score(query, packed_evidence)
        if was_training:
            self._model.train()
        return float(value.detach().cpu())

    def trainable_parameters(self):  # pragma: no cover
        self._ensure_loaded()
        return [p for p in self._model.parameters() if p.requires_grad]

    def train_mode(self):  # pragma: no cover
        self._ensure_loaded()
        self._model.train()

    def eval_mode(self):  # pragma: no cover
        self._ensure_loaded()
        self._model.eval()

    def save_pretrained(self, directory: str | Path) -> Path:  # pragma: no cover
        self._ensure_loaded()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out / "predictor_adapter"))
        return out

    @classmethod
    def from_pretrained(cls, directory: str | Path, config: LoraConfig, **kwargs):  # pragma: no cover
        instance = cls(config, **kwargs)
        instance._ensure_loaded()
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        out = Path(directory)
        base = AutoModelForSequenceClassification.from_pretrained(config.base_model)
        instance._model = PeftModel.from_pretrained(
            base, str(out / "predictor_adapter")
        ).to(instance._device)
        instance.eval_mode()
        return instance
