#!/usr/bin/env python3
"""Verify that running the generator's selection logits in fp32 removes the
batched-vs-sequential rationale wobble (the discrete top-k flipping under bf16 noise).

For each candidate we compute its sentence logits two ways — SOLO (its own forward) and
BATCHED (padded together with the query's other candidates) — under bf16 and under fp32,
then count how many candidates' SELECTED sentence set changes solo→batched. Expectation:
bf16 shows some flips, fp32 shows ~0.

  PYTHONPATH=src python3 scripts/verify_generator_fp32.py --limit 30
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=Path("data/snapshots/topa.page.compat.v1"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels"))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural-v2/epoch-4"))
    ap.add_argument("--lora-config", type=Path, default=Path("configs/lora_target_modules.yaml"))
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-selected", type=int, default=3)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    import torch
    if args.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, 0)

    from explainable_reranker.serve.api import _batch_from_response
    from explainable_reranker.data.sentence_index import build_sentence_index
    from explainable_reranker.distill.gates import hard_select_from_logits
    from explainable_reranker.models.select_predict.neural_model import load_neural_model
    from explainable_reranker.topa.adapter import parse_topa_page_response

    model = load_neural_model(
        args.checkpoint, args.lora_config, device="cuda",
        compute_dtype="bfloat16", max_length=args.max_length, max_selected=args.max_selected,
    )
    gen = model.generator
    tok, hf, head, device = gen._tokenizer, gen._model, gen._head, gen._device
    hf.eval()
    min_sel = getattr(gen, "min_selected", 1)
    max_sel = gen.max_selected

    def pool(hidden_row, offs, spans):
        pooled = []
        for (s, e) in spans:
            idx = [t for t, (ts, te) in enumerate(offs) if te > ts and ts < e and te > s]
            if idx:
                ii = torch.tensor(idx, device=device)
                pooled.append(hidden_row.index_select(0, ii).mean(dim=0))
            else:
                pooled.append(hidden_row[0])
        return head(torch.stack(pooled).float()).squeeze(-1).detach().float().cpu().tolist()

    def solo_logits(text, spans, fp32):
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=args.max_length)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx = contextlib.nullcontext() if fp32 else gen._autocast()
        with torch.no_grad(), ctx:
            hidden = hf(**enc).last_hidden_state[0]
        return pool(hidden, offs, spans)

    def batched_logits(texts, spans_list, fp32):
        enc = tok(texts, padding=True, truncation=True, max_length=args.max_length,
                  return_offsets_mapping=True, return_tensors="pt")
        offs_all = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx = contextlib.nullcontext() if fp32 else gen._autocast()
        with torch.no_grad(), ctx:
            hidden = hf(**enc).last_hidden_state  # [B, L, H]
        return [pool(hidden[b], offs_all[b].tolist(), spans_list[b]) for b in range(len(texts))]

    def sel(logits):
        return tuple(hard_select_from_logits(logits, threshold=0.0,
                                             min_selected=min_sel, max_selected=max_sel))

    label_ids = {p.stem for p in args.labels.glob("*.json")}
    snaps = [p for p in sorted(args.snapshots.glob("*.json")) if p.stem not in label_ids][: args.limit]
    print(f"queries: {len(snaps)}", flush=True)

    n_cand = 0
    flips_bf16 = 0
    flips_fp32 = 0
    delta_bf16: list[float] = []
    delta_fp32: list[float] = []

    for qi, snap in enumerate(snaps):
        env = json.loads(snap.read_text(encoding="utf-8"))
        payload = env.get("payload", env)
        try:
            response = parse_topa_page_response(payload)
            sentence_index = build_sentence_index(response)
            batch = _batch_from_response(response, sentence_index)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{qi+1}] {snap.stem}: skipped ({type(exc).__name__})", flush=True)
            continue
        query = batch.query
        texts, spans_list, cand_sents = [], [], []
        for cand in batch.candidates:
            sents = [lbl.sentence for lbl in cand.sentences]
            if not sents:
                continue
            text, spans = gen._build_packed_input(query, sents)
            texts.append(text)
            spans_list.append(spans)
            cand_sents.append(sents)
        if len(texts) < 2:
            continue

        bat_bf = batched_logits(texts, spans_list, False)
        bat_fp = batched_logits(texts, spans_list, True)
        for c in range(len(texts)):
            n_cand += 1
            solo_bf = solo_logits(texts[c], spans_list[c], False)
            solo_fp = solo_logits(texts[c], spans_list[c], True)
            delta_bf16.extend(abs(a - b) for a, b in zip(solo_bf, bat_bf[c]))
            delta_fp32.extend(abs(a - b) for a, b in zip(solo_fp, bat_fp[c]))
            if sel(solo_bf) != sel(bat_bf[c]):
                flips_bf16 += 1
            if sel(solo_fp) != sel(bat_fp[c]):
                flips_fp32 += 1
        print(f"  [{qi+1}/{len(snaps)}] {snap.stem}: cum flips bf16={flips_bf16} fp32={flips_fp32} "
              f"(cands={n_cand})", flush=True)

    print("\n===== generator selection: solo vs batched =====", flush=True)
    print(f"candidates compared: {n_cand}", flush=True)
    print(f"rationale flips  bf16: {flips_bf16}/{n_cand} ({100*flips_bf16/max(1,n_cand):.2f}%)", flush=True)
    print(f"rationale flips  fp32: {flips_fp32}/{n_cand} ({100*flips_fp32/max(1,n_cand):.2f}%)", flush=True)
    if delta_bf16:
        print(f"|Δlogit| solo-vs-batched  bf16: mean {statistics.mean(delta_bf16):.5f} "
              f"max {max(delta_bf16):.5f}", flush=True)
    if delta_fp32:
        print(f"|Δlogit| solo-vs-batched  fp32: mean {statistics.mean(delta_fp32):.6f} "
              f"max {max(delta_fp32):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
