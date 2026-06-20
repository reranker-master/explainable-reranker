#!/usr/bin/env python3
"""Measure /rerank inference latency on the UNLABELED snapshots (the queries whose
teacher labeling failed, so they never entered training — a leakage-free latency set).

Loads the trained select-then-predict model once, then times rerank_payload() per
query in-process (no HTTP server, no network — this is the pure model-compute latency
a /rerank request spends). The first few calls warm up CUDA/lazy init and are excluded
from the statistics.

  PYTHONPATH=src python3 scripts/bench_rerank.py \
      --checkpoint checkpoints/neural-v2/epoch-4 --lora-config configs/lora_target_modules.yaml
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=Path("data/snapshots/topa.page.compat.v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural-v2/epoch-4"))
    ap.add_argument("--lora-config", type=Path, default=Path("configs/lora_target_modules.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-dtype", default="bfloat16")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-selected", type=int, default=3)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.75)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap number of queries (0=all unlabeled)")
    args = ap.parse_args()

    import torch
    if args.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)

    # import serve.api first: it initializes the model->backends chain in the order that
    # avoids the distill<->backends circular import (same order serve_rerank.py uses).
    from explainable_reranker.serve.api import rerank_payload
    from explainable_reranker.models.select_predict.neural_model import load_neural_model

    # unlabeled snapshots = failed/unused queries (no teacher label)
    label_ids = {p.stem for p in args.labels.glob("*.json")}
    snaps = [p for p in sorted(args.snapshots.glob("*.json")) if p.stem not in label_ids]
    if args.limit:
        snaps = snaps[: args.limit]
    print(f"unlabeled (failed/unused) snapshots: {len(snaps)}", flush=True)

    print(f"loading model {args.checkpoint} ...", flush=True)
    t0 = time.perf_counter()
    model = load_neural_model(
        args.checkpoint, args.lora_config, device=args.device,
        compute_dtype=args.compute_dtype, max_length=args.max_length, max_selected=args.max_selected,
    )
    print(f"model loaded in {time.perf_counter()-t0:.1f}s ({type(model).__name__})", flush=True)

    rows: list[tuple[str, int, float]] = []  # (response_id, n_candidates, seconds)
    failures = 0
    for i, snap in enumerate(snaps):
        env = json.loads(snap.read_text(encoding="utf-8"))
        payload = env.get("payload", env)
        try:
            t = time.perf_counter()
            out = rerank_payload(payload, model=model)
            dt = time.perf_counter() - t
        except Exception as exc:  # noqa: BLE001 - a malformed pool shouldn't abort the bench
            failures += 1
            print(f"  [{i+1}/{len(snaps)}] {snap.stem}: rerank failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        n = len(out["results"])
        rows.append((snap.stem, n, dt))
        tag = " (warmup)" if i < args.warmup else ""
        print(f"  [{i+1}/{len(snaps)}] {snap.stem}: {n} cand -> {dt*1000:.0f} ms{tag}", flush=True)

    timed = rows[args.warmup:]  # exclude warmup
    if not timed:
        print("no timed samples", flush=True)
        return 1
    secs = [s for _, _, s in timed]
    cands = [n for _, n, _ in timed]
    per_cand_ms = [1000 * s / n for _, n, s in timed if n]
    secs_sorted = sorted(secs)
    p = lambda q: secs_sorted[min(len(secs_sorted) - 1, int(q * len(secs_sorted)))]

    print("\n===== /rerank latency (unlabeled set, warmup excluded) =====", flush=True)
    print(f"queries timed: {len(timed)} (+{args.warmup} warmup, {failures} failed)", flush=True)
    print(f"candidates/query: min {min(cands)} / median {int(statistics.median(cands))} / max {max(cands)}", flush=True)
    print(f"latency/query (s): mean {statistics.mean(secs):.3f} | median {statistics.median(secs):.3f} | "
          f"p90 {p(0.90):.3f} | p95 {p(0.95):.3f} | min {min(secs):.3f} | max {max(secs):.3f}", flush=True)
    print(f"per-candidate: mean {statistics.mean(per_cand_ms):.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
