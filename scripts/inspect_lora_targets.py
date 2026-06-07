#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


ATTENTION_PATTERNS = (
    r"(q_proj|k_proj|v_proj|o_proj)$",
    r"(query|key|value)$",
    r"attention\.output\.dense$",
)
EXCLUDED_PATTERNS = (r"classifier", r"score", r"pooler")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LoRA target module names for a HF reranker.")
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", default="configs/lora_target_modules.yaml")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Install training dependencies before LoRA inspection."
        ) from exc

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    module_names = [name for name, _module in model.named_modules()]
    targets = [
        name
        for name in module_names
        if any(re.search(pattern, name) for pattern in ATTENTION_PATTERNS)
        and not any(re.search(pattern, name) for pattern in EXCLUDED_PATTERNS)
    ]
    if not targets:
        raise SystemExit("no attention projection modules matched; inspect named_modules() manually")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"base_model: {args.model_id}",
                "strategy: inspected_attention_projection_modules",
                "generator_adapter:",
                "  r: 16",
                "  alpha: 32",
                "  dropout: 0.05",
                "predictor_adapter:",
                "  r: 16",
                "  alpha: 32",
                "  dropout: 0.05",
                "target_modules:",
                *[f"  - {name}" for name in targets],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(targets)} target modules to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
