#!/usr/bin/env python3
"""Lean training smoke: measure model load vs per-step time, confirm loss drops."""
from __future__ import annotations
import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `scripts.*`
from explainable_reranker.distill.neural_training import NeuralTrainConfig, train_joint
from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor, HFSentenceGenerator, load_lora_config,
)
from scripts.train_neural import load_batches

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

print(f"loading {N} batches...", flush=True)
batches = load_batches(Path("data/snapshots"), Path("data/labels"))[:N]
cand = sum(len(b.candidates) for b in batches)
print(f"  {len(batches)} batches, {cand} candidates total ({cand/len(batches):.0f}/query)", flush=True)

cfg = load_lora_config(Path("configs/lora_target_modules.yaml"))
gen = HFSentenceGenerator(cfg, device="cuda", compute_dtype="bfloat16", max_length=8192)
pred = HFPackedEvidencePredictor(cfg, device="cuda", compute_dtype="bfloat16", max_length=8192)

t0 = time.time()
print("loading models (downloads bge-reranker-v2-m3 on first run)...", flush=True)
gen._ensure_loaded(); pred._ensure_loaded()
load_s = time.time() - t0
print(f"  model load+download: {load_s:.1f}s", flush=True)

t1 = time.time()
hist = train_joint(gen, pred, batches,
                   NeuralTrainConfig(epochs=1, warmup_steps=3, total_steps=20, log_every=1), seed=0)
dt = time.time() - t1
steps = len(hist.losses)
per = dt / max(steps, 1)
print(f"\n=== RESULTS ===")
print(f"steps: {steps} in {dt:.1f}s  =>  {per:.2f}s/step", flush=True)
print(f"loss: first={hist.losses[0]:.4f}  last={hist.losses[-1]:.4f}  (down={'YES' if hist.losses[-1]<hist.losses[0] else 'no'})")
print(f"full-run ETA (989 batches x 3 epochs = 2967 steps): {2967*per/3600:.1f} hours")
