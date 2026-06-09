#!/usr/bin/env python3
"""Collect topa.page candidates and generate Opus teacher labels.

End-to-end data stage (plan §1–2): for each query
  1. request https://www.topa.page/api/search/search-candidates (no top_k → full set),
  2. save the raw response as an immutable snapshot,
  3. ask Claude Opus 4.8 for the grounded "best case" label (listwise ranking + rationales),
  4. write the label JSON next to the snapshot.

The output layout is exactly what scripts/train_neural.py consumes:
  <out>/snapshots/<schema_version>/<response_id>.json   (SnapshotStore envelope)
  <out>/labels/<response_id>.json                        (teacher label)

Example:
  PYTHONPATH=src ANTHROPIC_API_KEY=... python3 scripts/collect_and_label.py \
      --queries data/queries.txt --out data --max-sentences 16

Use --dummy to exercise the whole pipeline offline (scripted teacher, no API cost),
and --query "..." for a single ad-hoc query instead of a queries file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explainable_reranker.data.evidence_fallback import (
    StaticEvidenceFallback,
    augment_with_fallback,
)
from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.data.snapshot_store import SnapshotStore
from explainable_reranker.teacher.grounded_teacher import (
    GroundedTeacherConfig,
    LLMGroundedTeacher,
    TeacherLabelingError,
)
from explainable_reranker.teacher.agreement import self_consistency_report
from explainable_reranker.teacher.hard_negatives import (
    StaticHardNegativeSource,
    inject_hard_negatives,
)
from explainable_reranker.teacher.llm_client import AnthropicClaudeChatModel, ScriptedChatModel
from explainable_reranker.topa.client import HttpTopaPageClient, collect_snapshot


def _load_queries(args: argparse.Namespace) -> list[str]:
    if args.query:
        return [args.query]
    lines = Path(args.queries).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _scripted_teacher_response(response, sentence_index) -> str:
    """A deterministic, schema-valid label for --dummy runs (no API call).

    Ranks candidates by their topa retrieval score and cites each top book's
    first indexed sentence, so the offline pipeline produces real training rows.
    """

    sentences_by_book: dict[str, list] = {}
    for sentence in sentence_index:
        sentences_by_book.setdefault(sentence.book_id, []).append(sentence)
    ranked = sorted(response.candidates, key=lambda c: c.score or 0.0, reverse=True)
    n = len(ranked)
    ranking = [
        {"book": c.book_id, "score": round(3.0 * (n - i) / n, 4)} for i, c in enumerate(ranked)
    ]
    rationales = {}
    for c in ranked[:10]:
        sents = sentences_by_book.get(c.book_id, [])
        if sents:
            rationales[c.book_id] = {
                "sentence_ids": [sents[0].sentence_id],
                "reason": "retrieval-aligned stand-in rationale",
            }
    return json.dumps({"ranking": ranking, "rationales": rationales}, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect topa candidates + Opus teacher labels.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--queries", type=Path, help="file with one query per line")
    group.add_argument("--query", help="a single query string")
    parser.add_argument("--out", required=True, type=Path, help="output root dir")
    parser.add_argument("--base-url", default="https://www.topa.page")
    parser.add_argument("--path", default="/api/search/search-candidates")
    parser.add_argument("--top-k", type=int, default=None, help="omit to fetch the full set")
    parser.add_argument("--max-sentences", type=int, default=16, help="evidence cap per book (§1.5)")
    parser.add_argument("--top-k-rationale", type=int, default=10)
    parser.add_argument("--model-id", default=None, help="override the Opus model id")
    parser.add_argument("--dummy", action="store_true", help="offline scripted teacher (no API)")
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=0,
        help="relabel N>=2 times with shuffled order and report the §1.4 κ/NDCG/IoU gate "
        "(0 = off; saves the first run's label)",
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=None,
        help='mine hard negatives from a JSON file ({query: [neg,...]} or [neg,...]); '
        "mixed into the candidate pool before labeling (plan §3/§5.1.3)",
    )
    parser.add_argument(
        "--max-hard-negatives",
        type=int,
        default=None,
        help="cap injected hard negatives per query (default: no cap)",
    )
    parser.add_argument(
        "--evidence-fallback",
        type=Path,
        default=None,
        help="JSON {book_id: [sentence,...]} to backfill candidates with too few "
        "topa sentences (plan §1.5 book_chunks/Qdrant fallback)",
    )
    parser.add_argument(
        "--min-evidence",
        type=int,
        default=0,
        help="min sentences per book before the evidence fallback is queried (0 = off)",
    )
    args = parser.parse_args()

    queries = _load_queries(args)
    if not queries:
        raise SystemExit("no queries provided")

    snapshots_root = args.out / "snapshots"
    labels_dir = args.out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(snapshots_root)
    client = HttpTopaPageClient(args.base_url, path=args.path)
    teacher_config = GroundedTeacherConfig(
        top_k_rationale=args.top_k_rationale, max_sentences_per_book=args.max_sentences
    )
    hard_negative_source = (
        StaticHardNegativeSource.from_file(args.hard_negatives) if args.hard_negatives else None
    )
    evidence_fallback_source = (
        StaticEvidenceFallback.from_file(args.evidence_fallback) if args.evidence_fallback else None
    )

    labeled, failed, inconsistent = 0, 0, 0
    for i, query in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {query}")
        transform = None
        if hard_negative_source is not None or evidence_fallback_source is not None:
            def transform(payload, _query=query):
                # Mix hard negatives in first, then backfill thin candidates so the
                # fallback also reaches injected negatives that lack evidence.
                if hard_negative_source is not None:
                    negatives = hard_negative_source.fetch(_query, payload)
                    payload = inject_hard_negatives(
                        payload, negatives, max_negatives=args.max_hard_negatives
                    )
                if evidence_fallback_source is not None:
                    payload = augment_with_fallback(
                        payload, evidence_fallback_source, min_sentences=args.min_evidence
                    )
                return payload
        record, response = collect_snapshot(
            client, store, query, top_k=args.top_k, payload_transform=transform
        )
        sentence_index = build_sentence_index(response)
        injected = sum(1 for c in response.candidates if c.is_hard_negative)
        print(
            f"    candidates={len(response.candidates)} (hard_neg={injected}) "
            f"sentences={len(sentence_index)}"
        )

        if args.dummy:
            chat_model = ScriptedChatModel(
                _scripted_teacher_response(response, sentence_index)
            )
        else:
            chat_model = AnthropicClaudeChatModel(model_id=args.model_id)
        teacher = LLMGroundedTeacher(chat_model, teacher_config)
        try:
            if args.self_consistency >= 2:
                # plan §1.4: relabel with shuffled candidate order and gate on the
                # κ / NDCG@10 / rationale-IoU agreement before trusting the label.
                labels = teacher.label_with_self_consistency(
                    response, sentence_index, runs=args.self_consistency, seed=i
                )
                report = self_consistency_report(labels)
                status = "PASS" if report.passed else "FAIL"
                print(
                    f"    self-consistency: κ={report.weighted_kappa:.3f} "
                    f"ndcg@10={report.ndcg_at_10:.3f} iou={report.rationale_iou:.3f} [{status}]"
                )
                if not report.passed:
                    inconsistent += 1
                label = labels[0]
            else:
                label = teacher.label(response, sentence_index)
        except TeacherLabelingError as exc:
            failed += 1
            print(f"    teacher failed: {exc}")
            continue

        label_path = labels_dir / f"{response.response_id}.json"
        label_path.write_text(
            json.dumps(label.raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        labeled += 1
        print(f"    labeled → {label_path}")

    consistency_note = (
        f", {inconsistent} below self-consistency gate" if args.self_consistency >= 2 else ""
    )
    print(
        f"\ndone: {labeled} labeled, {failed} failed{consistency_note}; "
        f"snapshots under {snapshots_root}"
    )
    print(f"next: PYTHONPATH=src python3 scripts/train_neural.py "
          f"--snapshots {snapshots_root} --labels {labels_dir} --out checkpoints/neural-v1")
    return 0 if labeled else 1


if __name__ == "__main__":
    raise SystemExit(main())
