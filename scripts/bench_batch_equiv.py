#!/usr/bin/env python3
"""Empirically measure whether candidate-BATCHED scoring changes the rerank result
vs the current SEQUENTIAL (one-candidate-at-a-time) path.

The model scores each candidate independently (no cross-candidate interaction), so in
exact arithmetic batching is identical. This quantifies the *numerical* divergence that
bf16 + padding actually introduce:

  - Predictor: re-score each candidate's SAME selected evidence (a) sequentially and
    (b) in one padded batch; compare scores and the induced ranking.
  - Rationale stability proxy: the generator's selection is a discrete top-k over
    sentence logits; report the decision margin so we know how close selections sit to
    flipping under that bf16 noise.

  PYTHONPATH=src python3 scripts/bench_batch_equiv.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def kendall_tau_distance(order_a: list[str], order_b: list[str]) -> tuple[int, int]:
    """Return (#discordant pairs, #total pairs) between two rankings of the same items."""
    pos_b = {x: i for i, x in enumerate(order_b)}
    items = [x for x in order_a if x in pos_b]
    n = len(items)
    disc = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            # a has items[i] before items[j]; discordant if b disagrees
            if pos_b[items[i]] > pos_b[items[j]]:
                disc += 1
    return disc, total


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
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    import torch
    if args.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)

    from explainable_reranker.serve.api import _batch_from_response
    from explainable_reranker.data.sentence_index import build_sentence_index
    from explainable_reranker.models.select_predict.neural_model import load_neural_model
    from explainable_reranker.topa.adapter import parse_topa_page_response

    label_ids = {p.stem for p in args.labels.glob("*.json")}
    snaps = [p for p in sorted(args.snapshots.glob("*.json")) if p.stem not in label_ids][: args.limit]
    print(f"unlabeled snapshots used: {len(snaps)}", flush=True)

    model = load_neural_model(
        args.checkpoint, args.lora_config, device=args.device,
        compute_dtype=args.compute_dtype, max_length=args.max_length, max_selected=args.max_selected,
    )
    pred = model.predictor
    tok, hf, autocast, device = pred._tokenizer, pred._model, pred._autocast, pred._device
    hf.eval()

    def batched_scores(query: str, packed_list: list[str]) -> list[float]:
        """Score all (query, packed) pairs in ONE padded forward. Empty -> 0.0 (matches seq)."""
        idx_nonempty = [i for i, p in enumerate(packed_list) if p.strip()]
        scores = [0.0] * len(packed_list)
        if not idx_nonempty:
            return scores
        enc = tok([query] * len(idx_nonempty), [packed_list[i] for i in idx_nonempty],
                  padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad(), autocast():
            logits = hf(**enc).logits.float().reshape(len(idx_nonempty), -1)[:, 0].cpu().tolist()
        for i, s in zip(idx_nonempty, logits):
            scores[i] = s
        return scores

    score_deltas: list[float] = []
    top1_flips = 0
    any_rank_change = 0
    top5_changes = 0
    tau_fracs: list[float] = []
    margins: list[float] = []          # top-k selection margin per candidate (min_sel - max_unsel logit)
    borderline_sentences = 0           # sentences within bf16-ish noise of 0.0 threshold
    n_candidates_total = 0

    for s, snap in enumerate(snaps):
        env = json.loads(snap.read_text(encoding="utf-8"))
        payload = env.get("payload", env)
        try:
            response = parse_topa_page_response(payload)
            sentence_index = build_sentence_index(response)
            batch = _batch_from_response(response, sentence_index)
            seq = model.rerank_batch(batch)  # sorted desc by sequential score
        except Exception as exc:  # noqa: BLE001
            print(f"  [{s+1}] {snap.stem}: skipped ({type(exc).__name__})", flush=True)
            continue

        query = response.query
        # original (book_id, packed, seq_score) preserving each candidate's selected evidence
        books = [o.book_id for o in seq]
        packs = [o.packed_evidence for o in seq]
        seq_scores = [o.score for o in seq]
        bat_scores = batched_scores(query, packs)

        for a, b in zip(seq_scores, bat_scores):
            score_deltas.append(abs(a - b))

        seq_order = books  # already sorted by seq score
        bat_order = [bk for bk, _ in sorted(zip(books, bat_scores), key=lambda x: x[1], reverse=True)]
        if seq_order[:1] != bat_order[:1]:
            top1_flips += 1
        if seq_order != bat_order:
            any_rank_change += 1
        if set(seq_order[:5]) != set(bat_order[:5]):
            top5_changes += 1
        disc, total = kendall_tau_distance(seq_order, bat_order)
        if total:
            tau_fracs.append(disc / total)

        # rationale stability proxy: selection margin from generator gates
        for o in seq:
            n_candidates_total += 1
            logits = sorted((g.logit for g in o.gates), reverse=True)
            for g in o.gates:
                if abs(g.logit) < 0.05:
                    borderline_sentences += 1
            k = sum(1 for g in o.gates if g.selected)
            if 0 < k < len(logits):
                margins.append(logits[k - 1] - logits[k])  # gap at the selection boundary

        print(f"  [{s+1}/{len(snaps)}] {snap.stem}: maxΔscore={max(abs(a-b) for a,b in zip(seq_scores,bat_scores)):.4f} "
              f"top1={'SAME' if seq_order[:1]==bat_order[:1] else 'FLIP'} "
              f"order={'same' if seq_order==bat_order else 'changed'}", flush=True)

    n = len(tau_fracs)
    print("\n===== sequential vs candidate-batched predictor =====", flush=True)
    print(f"queries: {n} | candidates total: {n_candidates_total}", flush=True)
    print(f"|Δscore| per candidate: mean {statistics.mean(score_deltas):.5f} | "
          f"median {statistics.median(score_deltas):.5f} | max {max(score_deltas):.5f}", flush=True)
    print(f"top-1 flips: {top1_flips}/{n}  ({100*top1_flips/max(1,n):.1f}%)", flush=True)
    print(f"top-5 set changes: {top5_changes}/{n}  ({100*top5_changes/max(1,n):.1f}%)", flush=True)
    print(f"any ranking change: {any_rank_change}/{n}  ({100*any_rank_change/max(1,n):.1f}%)", flush=True)
    print(f"Kendall-tau distance (frac discordant pairs): mean {statistics.mean(tau_fracs):.4f} | "
          f"max {max(tau_fracs):.4f}", flush=True)
    print("\n----- rationale (generator selection) stability proxy -----", flush=True)
    if margins:
        msz = sorted(margins)
        print(f"selection boundary margin (logit gap): mean {statistics.mean(margins):.3f} | "
              f"median {statistics.median(margins):.3f} | min {min(margins):.3f} | "
              f"p5 {msz[max(0,int(0.05*len(msz)))]:.3f}", flush=True)
    print(f"sentences within |logit|<0.05 of threshold: {borderline_sentences} "
          f"(of pooled candidates; these are the rationale-flip-prone ones)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
