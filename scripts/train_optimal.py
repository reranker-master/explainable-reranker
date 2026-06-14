#!/usr/bin/env python3
"""Train the neural reranker with per-epoch checkpoints + validation-driven model selection.

Design goals (this is the "find the optimal, keep every rollback point, document it" run):
  1. Split the labeled set into train / valid / test (seeded, leakage caveat below).
  2. Train, and after EVERY epoch: save a full checkpoint (rollback point) AND score the
     held-out valid split (NDCG@5 / NDCG@10 / rationale-IoU / Recall@10).
  3. Pick the epoch with the best valid NDCG@5 (this is early-stopping by selection).
  4. Reload that best checkpoint and report its UNBIASED test-split metrics.
  5. Write training_report.md documenting the whole search + metrics.json for the curve.

Every epoch's checkpoint stays on disk (epoch-1..N), so you can roll back to any of them.

  PYTHONPATH=src python3 scripts/train_optimal.py --out checkpoints/neural-v1 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for scripts.*

from dataclasses import replace

from explainable_reranker.distill.dataset import QueryTrainingBatch, pack_selected_evidence
from explainable_reranker.distill.gates import hard_select_from_logits
from explainable_reranker.distill.neural_training import (
    NeuralTrainConfig,
    save_neural_checkpoint,
    train_joint,
)
from explainable_reranker.eval.run_eval import (
    PredictionItem,
    QueryQrels,
    evaluate_predictions,
    report_to_dict,
)
from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
    load_lora_config,
)
from scripts.train_neural import load_batches


# --------------------------------------------------------------------------- #
# Student inference + validation
# --------------------------------------------------------------------------- #
def _student_predictions(torch, gen, pred, batch: QueryTrainingBatch) -> list[PredictionItem]:
    """Run generator (select evidence) + predictor (score) for every candidate, no grad."""
    items: list[PredictionItem] = []
    with torch.no_grad():
        for cand in batch.candidates:
            sentences = tuple(label.sentence for label in cand.sentences)
            if sentences:
                logits = gen._forward_logits(batch.query, sentences)
                flags = hard_select_from_logits(
                    logits.detach().float().cpu().tolist(),
                    threshold=0.0,
                    min_selected=gen.min_selected,
                    max_selected=gen.max_selected,
                )
                selected = {
                    label.sentence.sentence_id
                    for label, keep in zip(cand.sentences, flags, strict=True)
                    if keep
                }
            else:
                selected = set()
            packed = pack_selected_evidence(cand, selected)
            score = float(pred._forward_score(batch.query, packed).detach().cpu())
            items.append(PredictionItem(cand.book_id, score, tuple(sorted(selected))))
    return items


def _cap_candidates(batch: QueryTrainingBatch, cap: int) -> QueryTrainingBatch:
    """Optional memory safety: keep top-by-score + all hard negatives, drop the rest.

    Only used if cap>0 and the batch exceeds it. Preserves the decision-boundary signal
    (top relevant books + the in-pool traps) while cutting per-step activation memory
    linearly. Applied to TRAIN batches only — valid/test keep the full pool for accurate
    metrics.
    """
    if cap <= 0 or len(batch.candidates) <= cap:
        return batch
    hard = [c for c in batch.candidates if c.hard_label == 0]
    rest = [c for c in batch.candidates if c.hard_label != 0]  # score-sorted desc already
    keep = (rest[: max(0, cap - len(hard))] + hard)[:cap]
    keep = sorted(keep, key=lambda c: c.teacher_score, reverse=True)
    return replace(batch, candidates=tuple(keep))


def _batch_qrels(batch: QueryTrainingBatch) -> QueryQrels:
    """Teacher label as the gold target for selection (graded score + cited sentences)."""
    relevance = {c.book_id: float(c.teacher_score) for c in batch.candidates}
    rationale = {
        c.book_id: set(c.teacher_rationale_ids())
        for c in batch.candidates
        if c.teacher_rationale_ids()
    }
    return QueryQrels(batch.query_id, relevance, rationale)


def _evaluate(torch, gen, pred, batches) -> dict:
    gen.eval_mode()
    pred.eval_mode()
    qrels = {b.query_id: _batch_qrels(b) for b in batches}
    preds = {b.query_id: _student_predictions(torch, gen, pred, b) for b in batches}
    report = report_to_dict(evaluate_predictions(qrels, preds))
    gen.train_mode()
    pred.train_mode()
    return report


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #
def _write_report(out: Path, meta: dict, rows: list[dict], best: dict, test: dict, when: str) -> None:
    def fmt(r):
        return (f"| {r['epoch']} | {r['train_loss']:.4f} | {r['ndcg_at_5']:.4f} | "
                f"{r['ndcg_at_10']:.4f} | {r['rationale_iou']:.4f} | {r['recall_at_10']:.4f} | "
                f"`{r['checkpoint']}` |")

    lines = [
        f"# Neural reranker training — optimal-point search\n",
        f"_Generated {when}_\n",
        "## How the optimal was found\n",
        "Validation-driven model selection (early stopping by selection): the model is trained for "
        f"{meta['epochs']} epochs; after **every** epoch a full checkpoint is saved (a rollback point) "
        "and the **held-out valid split** is scored. The epoch with the best **valid NDCG@5** is chosen; "
        "its **test-split** metrics (never seen during training or selection) are the unbiased headline. "
        "Training loss always falls, so it cannot pick the epoch — the valid curve does.\n",
        "## Data\n",
        f"- Labeled queries: **{meta['total']}** (teacher: {meta['teacher']})\n"
        f"- Split (seed {meta['seed']}): train **{meta['n_train']}** / valid **{meta['n_valid']}** / "
        f"test **{meta['n_test']}** ({meta['ratios']})\n",
        "## Config\n",
        f"- base model `{meta['base_model']}` + LoRA, lr {meta['lr']}, "
        f"warmup {meta['warmup']} / total {meta['total']} steps\n"
        f"- avg candidates/query {meta['avg_cand']:.1f}, step time ~{meta['step_s']:.1f}s\n",
        "## Per-epoch results (valid split)\n",
        "| epoch | train_loss | NDCG@5 | NDCG@10 | rationale_IoU | Recall@10 | checkpoint |",
        "|---|---|---|---|---|---|---|",
        *[fmt(r) for r in rows],
        "",
        f"## Selected: **epoch {best['epoch']}** (best valid NDCG@5 = {best['ndcg_at_5']:.4f})\n",
        "Why this epoch: highest validation NDCG@5; later epochs did not improve valid "
        "(training-loss gains there are memorization, not generalization).\n",
        "## Test-split metrics (unbiased, best checkpoint)\n",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in test.items()],
        "",
        "## Rollback\n",
        f"Every epoch checkpoint is kept under `{out}/epoch-N/`. The selected one is recorded in "
        f"`{out}/BEST.txt`. To roll back, load any `epoch-N` directory via "
        "`HFSentenceGenerator.from_pretrained` / `HFPackedEvidencePredictor.from_pretrained`.\n",
        "## Caveats\n",
        "- **Split is random by query, not family/cluster-deduplicated** (no family/cluster metadata "
        "available); near-duplicate query intents could straddle train/valid, making valid mildly "
        "optimistic. Re-split with `make_splits.py` once family/cluster ids exist.\n",
        "- **Gold = teacher labels**, so metrics measure how well the student reproduces the teacher on "
        "unseen queries (distillation generalization), not human relevance. An independent human qrels "
        "set is needed for an absolute quality claim.\n",
    ]
    (out / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels"))
    ap.add_argument("--lora-config", type=Path, default=Path("configs/lora_target_modules.yaml"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/neural-v1"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    # Inputs are short (query + ~11 sentences for G; 1-3 sentences for P), so 2048 is
    # ample and caps activation memory far below the 8192 default — part of keeping
    # this run from starving co-located services on the shared box.
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=0.78,
        help="hard cap on the fraction of (unified) GPU memory torch may allocate, so a spike "
        "raises a clean OOM here instead of freezing the box / co-located qdrant+topa services "
        "(0.78 of 128GB ~= 100GB for training, ~28GB reserved for services)",
    )
    ap.add_argument(
        "--max-train-candidates",
        type=int,
        default=0,
        help="if >0, cap candidates per TRAIN query (top-by-score + hard negatives) to bound "
        "per-step memory; 0 = use the full pool. valid/test always use the full pool",
    )
    args = ap.parse_args()

    import torch  # noqa: local import so --help works off-GPU

    if args.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)
        print(f"GPU memory capped at fraction {args.gpu_mem_fraction} of device 0", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    batches = load_batches(args.snapshots, args.labels)
    rng = random.Random(args.seed)
    rng.shuffle(batches)
    n = len(batches)
    n_test = max(1, n // 10)
    n_valid = max(1, n // 10)
    test_b = batches[:n_test]
    valid_b = batches[n_test : n_test + n_valid]
    train_b = [_cap_candidates(b, args.max_train_candidates) for b in batches[n_test + n_valid :]]
    print(f"split: train={len(train_b)} valid={len(valid_b)} test={len(test_b)} (of {n})"
          f"{' | train candidates capped at '+str(args.max_train_candidates) if args.max_train_candidates else ''}",
          flush=True)

    cfg = load_lora_config(args.lora_config)
    gen = HFSentenceGenerator(cfg, device="cuda", compute_dtype="bfloat16", max_length=args.max_length)
    pred = HFPackedEvidencePredictor(cfg, device="cuda", compute_dtype="bfloat16", max_length=args.max_length)

    warmup = max(20, len(train_b) // 2)
    total = max(warmup + 1, args.epochs * len(train_b))
    config = NeuralTrainConfig(epochs=args.epochs, learning_rate=args.lr,
                               warmup_steps=warmup, total_steps=total, log_every=20)

    rows: list[dict] = []
    t_start = [time.time()]

    def on_epoch_end(epoch, g, p, history):
        ep = epoch + 1
        ckpt = args.out / f"epoch-{ep}"
        save_neural_checkpoint(ckpt, g, p)
        report = _evaluate(torch, g, p, valid_b)
        row = {"epoch": ep, "train_loss": history.losses[-1], "checkpoint": ckpt.name, **report}
        rows.append(row)
        # incremental dump so an interrupted run still keeps the curve
        (args.out / "metrics.json").write_text(json.dumps({"valid_curve": rows}, indent=2), encoding="utf-8")
        dt = time.time() - t_start[0]
        print(f"[epoch {ep}/{args.epochs}] train_loss={row['train_loss']:.4f} "
              f"valid NDCG@5={report['ndcg_at_5']:.4f} NDCG@10={report['ndcg_at_10']:.4f} "
              f"IoU={report['rationale_iou']:.4f}  ({dt/60:.1f}min elapsed)", flush=True)

    print(f"training {args.epochs} epochs...", flush=True)
    train_joint(gen, pred, train_b, config, seed=args.seed, on_epoch_end=on_epoch_end)

    best = max(rows, key=lambda r: (r["ndcg_at_5"], r["rationale_iou"]))
    (args.out / "BEST.txt").write_text(best["checkpoint"] + "\n", encoding="utf-8")
    print(f"\nbest epoch: {best['epoch']} (valid NDCG@5={best['ndcg_at_5']:.4f})", flush=True)

    # unbiased test eval on the reloaded best checkpoint
    best_dir = args.out / best["checkpoint"]
    gen_b = HFSentenceGenerator.from_pretrained(best_dir, cfg, device="cuda",
                                                compute_dtype="bfloat16", max_length=args.max_length)
    pred_b = HFPackedEvidencePredictor.from_pretrained(best_dir, cfg, device="cuda",
                                                       compute_dtype="bfloat16", max_length=args.max_length)
    test_report = _evaluate(torch, gen_b, pred_b, test_b)
    print(f"TEST (best): NDCG@5={test_report['ndcg_at_5']:.4f} NDCG@10={test_report['ndcg_at_10']:.4f} "
          f"IoU={test_report['rationale_iou']:.4f}", flush=True)

    avg_cand = sum(len(b.candidates) for b in batches) / max(1, len(batches))
    meta = {"total": n, "teacher": "Opus 23 + DeepSeek", "seed": args.seed,
            "n_train": len(train_b), "n_valid": len(valid_b), "n_test": len(test_b),
            "ratios": "0.8/0.1/0.1", "base_model": cfg.base_model, "lr": args.lr,
            "warmup": warmup, "total": total, "epochs": args.epochs, "avg_cand": avg_cand,
            "step_s": (time.time() - t_start[0]) / max(1, args.epochs * len(train_b))}
    (args.out / "metrics.json").write_text(
        json.dumps({"valid_curve": rows, "best_epoch": best["epoch"], "test": test_report, "meta": meta},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(args.out, meta, rows, best, test_report, datetime.now().isoformat(timespec="minutes"))
    print(f"\nwrote {args.out}/training_report.md + metrics.json + epoch-1..{args.epochs}/ + BEST.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
