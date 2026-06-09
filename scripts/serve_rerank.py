#!/usr/bin/env python3
"""Serve the drop-in `/rerank` HTTP endpoint (plan §3).

By default this serves the dependency-free lexical stand-in so the endpoint runs
anywhere. Pass `--checkpoint` (+ `--lora-config`) to load a trained neural
select-then-predict model instead — the serving contract is identical, only the
generator/predictor backends change.

Examples:
  # lexical stand-in (no GPU, smoke/integration)
  PYTHONPATH=src python3 scripts/serve_rerank.py --port 8080

  # trained student (needs torch + a checkpoint from scripts/train_neural.py)
  PYTHONPATH=src python3 scripts/serve_rerank.py \
      --checkpoint checkpoints/neural-v1 --lora-config configs/lora_target_modules.yaml

Then POST a topa.page response JSON to /rerank, or GET /healthz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from explainable_reranker.serve.http_app import RerankApp, serve


def build_model(args: argparse.Namespace):
    """Lexical stand-in by default; the trained neural model when --checkpoint is set."""

    if args.checkpoint is None:
        return None  # RerankApp falls back to SelectThenPredictModel() (lexical)
    # Imported lazily: load_neural_model pulls in the torch-backed backends, which we
    # only want on a box that actually has a checkpoint to serve.
    from explainable_reranker.models.select_predict.neural_model import load_neural_model

    return load_neural_model(
        args.checkpoint,
        args.lora_config,
        device=args.device,
        compute_dtype=args.compute_dtype,
        max_length=args.max_length,
        max_selected=args.max_selected,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the drop-in /rerank endpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="trained neural checkpoint dir (omit for lexical)"
    )
    parser.add_argument("--lora-config", type=Path, default=Path("configs/lora_target_modules.yaml"))
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (auto if unset)")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-selected", type=int, default=3)
    args = parser.parse_args()

    app = RerankApp(model=build_model(args))
    backend = type(app.model).__name__
    print(f"serving /rerank ({backend}) on http://{args.host}:{args.port}  (GET /healthz)")
    serve(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
