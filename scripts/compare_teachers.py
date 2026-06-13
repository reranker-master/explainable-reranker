#!/usr/bin/env python3
"""Bake-off: which open teacher's labels land closest to the Opus reference.

For each already-Opus-labeled query, re-run Pass A (ranking + in-pool hard
negatives) with each candidate model over the SAME snapshot candidate pool, then
score closeness to the Opus label on the decision-relevant axes:

  - top5_overlap   : |model top-5 ∩ opus top-5| / 5   (the reranker only shows 5)
  - hardneg_jaccard: agreement on which books are traps
  - score_spearman : rank correlation of per-book scores over shared books

Calls go through the cache/ledger, so reruns are free and costs are recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

from explainable_reranker.config.env import load_project_dotenv
from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.io_cache import CachingChatModel
from explainable_reranker.teacher.llm_client import BedrockConverseChatModel, extract_json_object
from explainable_reranker.teacher.prompts import SYSTEM_INSTRUCTIONS, build_listwise_prompt
from explainable_reranker.topa.adapter import parse_topa_page_response

CANDIDATES = {
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "deepseek-v3.2": "deepseek.v3.2",
}


def _ranked(payload: dict) -> list[tuple[str, float]]:
    out = []
    for it in payload.get("ranking", []):
        if isinstance(it, dict) and (it.get("book") or it.get("book_id")):
            try:
                out.append((str(it.get("book") or it.get("book_id")), float(it.get("score", 0.0))))
            except (TypeError, ValueError):
                pass
    out.sort(key=lambda p: p[1], reverse=True)
    return out


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    n = len(common)
    if n < 2:
        return float("nan")

    def ranks(scores):
        order = sorted(common, key=lambda k: scores[k])
        return {k: i for i, k in enumerate(order)}

    ra, rb = ranks(a), ranks(b)
    d2 = sum((ra[k] - rb[k]) ** 2 for k in common)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> int:
    load_project_dotenv()
    import os

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    out_dir = Path("data/_teacher_bakeoff")
    snap_q = {}
    snap_path = {}
    for f in Path("data/snapshots").glob("*/*.json"):
        m = json.loads(f.read_text())["metadata"]
        snap_q[m["response_id"]] = m["query"]
        snap_path[m["response_id"]] = f
    ref_ids = [p.stem for p in sorted(Path("data/labels").glob("*.json"))][:5]

    agg: dict[str, list] = {name: [] for name in CANDIDATES}
    for rid in ref_ids:
        opus = json.loads((Path("data/labels") / f"{rid}.json").read_text())
        opus_scores = {str(r["book"]): float(r["score"]) for r in opus["ranking"]}
        opus_top5 = [b for b, _ in sorted(opus_scores.items(), key=lambda p: p[1], reverse=True)[:5]]
        opus_hn = set(opus.get("hard_negatives", {}))

        response = parse_topa_page_response(json.loads(snap_path[rid].read_text())["payload"])
        sidx = build_sentence_index(response)
        prompt = build_listwise_prompt(response, sidx, max_sentences_per_book=16)

        print(f"\n=== {snap_q[rid]}  (opus: {len(opus_scores)} ranked, {len(opus_hn)} hard-neg) ===")
        for name, model_id in CANDIDATES.items():
            chat = CachingChatModel(
                BedrockConverseChatModel(model_id=model_id, region=region, max_tokens=8192),
                cache_dir=out_dir / "cache" / "teacher",
                model_id=model_id,
                provider="bakeoff",
                ledger_path=out_dir / "ledger.jsonl",
            )
            try:
                payload = extract_json_object(chat.generate(system=SYSTEM_INSTRUCTIONS, user=prompt))
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:14s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
                agg[name].append(None)
                continue
            ranked = _ranked(payload)
            m_scores = dict(ranked)
            m_top5 = [b for b, _ in ranked[:5]]
            m_hn = set(payload.get("hard_negatives", {}))
            top5 = len(set(m_top5) & set(opus_top5)) / 5.0
            jac = _jaccard(m_hn, opus_hn)
            sp = _spearman(opus_scores, m_scores)
            agg[name].append((top5, jac, sp))
            print(f"  {name:14s} top5={top5:.2f}  hardneg_jac={jac:.2f}  spearman={sp:.2f}  "
                  f"(ranked {len(ranked)}, hard-neg {len(m_hn)})")

    print("\n========== AGGREGATE (mean over queries) ==========")
    for name in CANDIDATES:
        rows = [r for r in agg[name] if r is not None]
        if not rows:
            print(f"  {name:14s}  no successful runs")
            continue
        t = sum(r[0] for r in rows) / len(rows)
        j = sum(r[1] for r in rows) / len(rows)
        s = sum(r[2] for r in rows if r[2] == r[2]) / max(1, sum(1 for r in rows if r[2] == r[2]))
        composite = (t + j + (s + 1) / 2) / 3  # spearman -1..1 -> 0..1
        print(f"  {name:14s}  top5={t:.2f}  hardneg_jac={j:.2f}  spearman={s:.2f}  "
              f"=> closeness={composite:.3f}  ({len(rows)}/5 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
