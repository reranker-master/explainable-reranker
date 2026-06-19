#!/usr/bin/env python3
"""Data-size ablation (learning curve) to answer: are we data-bound?

Reuses the EXACT seed-0 split from train_optimal.py so the held-out valid set is
identical across data-points and comparable to checkpoints/neural-v1. Then trains
fresh student adapters on nested subsets of the train split (200 ⊂ 400 ⊂ ...) and
scores the SAME valid split each time. If the metric is still climbing at the full
size, more labels will help; if it has flattened, the ceiling is label quality /
loss design, not quantity.

  PYTHONPATH=src python3 scripts/learning_curve.py --epochs 3 --out .tmp/learning_curve

Trains smallest-first and dumps results incrementally, so the early points give a
signal before the whole sweep finishes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for scripts.*

from explainable_reranker.distill.neural_training import NeuralTrainConfig, train_joint
from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
    load_lora_config,
)
from scripts.train_neural import load_batches
from scripts.train_optimal import _cap_candidates, _evaluate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels"))
    ap.add_argument("--lora-config", type=Path, default=Path("configs/lora_target_modules.yaml"))
    ap.add_argument("--out", type=Path, default=Path(".tmp/learning_curve"))
    ap.add_argument("--epochs", type=int, default=3, help="fixed epochs per data-point (peak was epoch 3)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.78)
    # Cap train candidates to keep the sweep tractable. Applied identically to EVERY
    # data-point, so the cross-size trend (the thing we care about) stays fair; only the
    # absolute level shifts vs the uncapped production run. valid always uses the full pool.
    ap.add_argument("--max-train-candidates", type=int, default=24)
    ap.add_argument(
        "--fractions", type=str, default="0.25,0.5,0.75,1.0",
        help="comma list of train-split fractions to sweep (nested subsets)",
    )
    args = ap.parse_args()

    import torch  # local import so --help works off-GPU

    if args.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- identical split to train_optimal.py (seed 0) -------------------------
    batches = load_batches(args.snapshots, args.labels)
    rng = random.Random(args.seed)
    rng.shuffle(batches)
    n = len(batches)
    n_test = max(1, n // 10)
    n_valid = max(1, n // 10)
    valid_b = batches[n_test : n_test + n_valid]
    train_full = batches[n_test + n_valid :]
    print(f"split: train_full={len(train_full)} valid={len(valid_b)} test={n_test} (of {n})", flush=True)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    sizes = sorted({max(1, int(round(f * len(train_full)))) for f in fractions})

    cfg = load_lora_config(args.lora_config)
    rows: list[dict] = []
    t0 = time.time()

    for k in sizes:
        # nested subset: first k of the (already shuffled) train pool
        subset = [_cap_candidates(b, args.max_train_candidates) for b in train_full[:k]]
        warmup = max(20, k // 2)
        total = max(warmup + 1, args.epochs * k)
        config = NeuralTrainConfig(
            epochs=args.epochs, learning_rate=args.lr,
            warmup_steps=warmup, total_steps=total, log_every=50,
        )
        # fresh adapters each data-point (no warm-starting between sizes)
        gen = HFSentenceGenerator(cfg, device="cuda", compute_dtype="bfloat16", max_length=args.max_length)
        pred = HFPackedEvidencePredictor(cfg, device="cuda", compute_dtype="bfloat16", max_length=args.max_length)

        ts = time.time()
        print(f"\n=== n_train={k}  ({args.epochs} epochs, warmup {warmup}) ===", flush=True)
        train_joint(gen, pred, subset, config, seed=args.seed)
        report = _evaluate(torch, gen, pred, valid_b)
        row = {"n_train": k, **report, "minutes": round((time.time() - ts) / 60, 1)}
        rows.append(row)
        (args.out / "curve.json").write_text(
            json.dumps({"valid_curve_by_size": rows, "n_valid": len(valid_b),
                        "epochs": args.epochs, "max_train_candidates": args.max_train_candidates,
                        "seed": args.seed}, indent=2),
            encoding="utf-8",
        )
        print(f"[n={k}] valid NDCG@10={report['ndcg_at_10']:.4f} NDCG@5={report['ndcg_at_5']:.4f} "
              f"rationale_f1={report['rationale_f1']:.4f} IoU={report['rationale_iou']:.4f} "
              f"recall@10={report['recall_at_10']:.4f}  ({row['minutes']}min)", flush=True)

        # free for next size
        del gen, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- conclusion ----------------------------------------------------------
    print(f"\n===== learning curve (valid, n_valid={len(valid_b)}, total {(time.time()-t0)/60:.1f}min) =====", flush=True)
    print(f"{'n_train':>8} | {'NDCG@10':>8} | {'NDCG@5':>8} | {'rat_f1':>7} | {'rat_IoU':>7} | {'rec@10':>7}", flush=True)
    for r in rows:
        print(f"{r['n_train']:>8} | {r['ndcg_at_10']:>8.4f} | {r['ndcg_at_5']:>8.4f} | "
              f"{r['rationale_f1']:>7.4f} | {r['rationale_iou']:>7.4f} | {r['recall_at_10']:>7.4f}", flush=True)

    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        d_n = b["n_train"] - a["n_train"]
        for key in ("ndcg_at_10", "rationale_f1"):
            slope_per_100 = (b[key] - a[key]) / max(1, d_n) * 100
            print(f"\nlast-segment slope d({key})/d(+100 queries) = {slope_per_100:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
