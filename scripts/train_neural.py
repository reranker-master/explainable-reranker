#!/usr/bin/env python3
"""Train the neural select-then-predict model on the DGX Spark (GB10).

Loads topa.page snapshots + grounded teacher labels from disk, builds per-query
training batches, and runs the joint LoRA distillation loop, then writes the
generator/predictor adapter checkpoints.

Expected layout (both produced by the W1/W2-3 pipeline):
  <snapshots>/<response_id>.json   # SnapshotStore envelope OR raw topa payload
  <labels>/<response_id>.json      # teacher label payload: {"ranking": [...], "rationales": {...}}

Example:
  PYTHONPATH=src python3 scripts/train_neural.py \
      --snapshots data/snapshots --labels data/labels \
      --lora-config configs/lora_target_modules.yaml \
      --out checkpoints/neural-v1 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.distill.dataset import QueryTrainingBatch, build_training_batch
from explainable_reranker.distill.neural_training import (
    NeuralTrainConfig,
    save_neural_checkpoint,
    train_joint,
)
from explainable_reranker.models.select_predict.backends import (
    HFPackedEvidencePredictor,
    HFSentenceGenerator,
    load_lora_config,
)
from explainable_reranker.teacher.schemas import parse_teacher_label
from explainable_reranker.topa.adapter import parse_topa_page_response


def _load_payload(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    # SnapshotStore wraps the topa payload under "payload"; accept raw too.
    return envelope.get("payload", envelope) if isinstance(envelope, dict) else envelope


def load_batches(snapshots_dir: Path, labels_dir: Path) -> list[QueryTrainingBatch]:
    batches: list[QueryTrainingBatch] = []
    skipped = 0
    for snapshot_path in sorted(snapshots_dir.glob("**/*.json")):
        response = parse_topa_page_response(_load_payload(snapshot_path))
        label_path = labels_dir / f"{response.response_id}.json"
        if not label_path.exists():
            skipped += 1
            continue
        sentence_index = build_sentence_index(response)
        teacher_label = parse_teacher_label(
            json.loads(label_path.read_text(encoding="utf-8")),
            query_id=response.query_id,
            response_id=response.response_id,
        )
        batches.append(build_training_batch(response, sentence_index, teacher_label))
    if skipped:
        print(f"warning: {skipped} snapshot(s) had no matching teacher label and were skipped")
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint LoRA distillation for the neural reranker.")
    parser.add_argument("--snapshots", required=True, type=Path, help="dir of topa.page snapshots")
    parser.add_argument("--labels", required=True, type=Path, help="dir of teacher label JSON files")
    parser.add_argument("--lora-config", default="configs/lora_target_modules.yaml", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="checkpoint output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--max-selected", type=int, default=3)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (auto if unset)")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batches = load_batches(args.snapshots, args.labels)
    if not batches:
        raise SystemExit("no training batches found — check --snapshots and --labels paths")
    print(f"loaded {len(batches)} query batches")

    lora_config = load_lora_config(args.lora_config)
    generator = HFSentenceGenerator(
        lora_config,
        max_selected=args.max_selected,
        device=args.device,
        compute_dtype=args.compute_dtype,
        max_length=args.max_length,
    )
    predictor = HFPackedEvidencePredictor(
        lora_config,
        device=args.device,
        compute_dtype=args.compute_dtype,
        max_length=args.max_length,
    )

    config = NeuralTrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )
    history = train_joint(generator, predictor, batches, config, seed=args.seed)
    print(f"training done: final loss={history.final:.4f}")
    if generator.truncated_sentences:
        print(
            f"note: {generator.truncated_sentences} sentence(s) exceeded max_length and used the "
            "CLS fallback — consider tightening evidence preselect (plan §1.5)"
        )

    out = save_neural_checkpoint(args.out, generator, predictor)
    (out / "training_meta.json").write_text(
        json.dumps(
            {
                "base_model": lora_config.base_model,
                "epochs": args.epochs,
                "num_batches": len(batches),
                "final_loss": history.final,
                "max_selected": args.max_selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote checkpoint to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
