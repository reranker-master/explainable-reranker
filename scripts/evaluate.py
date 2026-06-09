#!/usr/bin/env python3
"""Run offline evaluation (plan §4): IR metrics + rationale faithfulness.

Reads gold qrels and model predictions and prints the EvaluationReport
(NDCG@1/5/10, MRR, Recall@10, rationale F1/IoU).

IMPORTANT (plan §4 "라벨 분리 원칙"): the qrels here must be an INDEPENDENT eval
set — not teacher labels and not derived from the model's own outputs — otherwise
the rationale metrics are tautological. This script only consumes the files.

File formats:
  --qrels <json>:        {"<query_id>": {"relevance_by_book": {"<book>": 3.0, ...},
                                          "rationale_ids_by_book": {"<book>": ["<sid>", ...]}}}
  --predictions <json>:  {"<query_id>": [{"book_id": "...", "score": 1.2,
                                          "rationale_sentence_ids": ["<sid>", ...]}, ...]}

Example:
  PYTHONPATH=src python3 scripts/evaluate.py \
      --qrels data/eval/qrels.json --predictions data/eval/predictions.json --out report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explainable_reranker.eval.run_eval import (
    evaluate_predictions,
    load_predictions,
    load_qrels,
    report_to_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline reranker + rationale evaluation.")
    parser.add_argument("--qrels", required=True, type=Path, help="independent gold qrels JSON")
    parser.add_argument("--predictions", required=True, type=Path, help="model predictions JSON")
    parser.add_argument("--out", type=Path, default=None, help="optional path to write the report JSON")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    predictions = load_predictions(args.predictions)
    missing = sorted(set(qrels) - set(predictions))
    if missing:
        print(f"warning: {len(missing)} qrels query(ies) had no predictions: {missing[:5]}...")

    report = report_to_dict(evaluate_predictions(qrels, predictions))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
