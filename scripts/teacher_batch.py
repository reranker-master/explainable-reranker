#!/usr/bin/env python3
"""Step-by-step Bedrock Batch teacher labeling.

This does not replace ``scripts/collect_and_label.py``. Use the existing sync
script for small pilots/debugging, and use this batch CLI for large runs where a
human reviews artifacts between stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from explainable_reranker.teacher.batch import (
    BatchModelConfig,
    BedrockBatchClient,
    approve_review_file,
    fetch_batch_stage,
    finalize_labels,
    prepare_ranking_batch,
    prepare_rationale_batch,
    read_jsonl,
    status_batch_stage,
    submit_batch_stage,
)


def _model_config(args: argparse.Namespace) -> BatchModelConfig:
    return BatchModelConfig(max_tokens=args.max_tokens, temperature=args.temperature)


def _client(args: argparse.Namespace) -> BedrockBatchClient:
    return BedrockBatchClient(region=args.region)


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _review_summary(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    summary: dict[str, int] = {"total": len(rows)}
    for row in rows:
        status = str(row.get("review_status", "missing"))
        summary[status] = summary.get(status, 0) + 1
        if row.get("errors"):
            summary["with_errors"] = summary.get("with_errors", 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, review, and finalize Bedrock teacher batches."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_ranking = sub.add_parser("prepare-ranking")
    prepare_ranking.add_argument("--batch-dir", required=True, type=Path)
    prepare_ranking.add_argument("--snapshots", required=True, type=Path)
    prepare_ranking.add_argument("--max-sentences", type=int, default=16)
    prepare_ranking.add_argument("--max-tokens", type=int, default=32000)
    prepare_ranking.add_argument("--temperature", type=float, default=0.0)
    prepare_ranking.add_argument("--limit", type=int, default=None)

    for name in ("submit-ranking", "submit-rationale"):
        submit = sub.add_parser(name)
        submit.add_argument("--batch-dir", required=True, type=Path)
        submit.add_argument("--role-arn", required=True)
        submit.add_argument("--model-id", required=True)
        submit.add_argument("--s3-input", required=True, help="S3 prefix for input JSONL upload")
        submit.add_argument("--s3-output", required=True, help="S3 prefix where Bedrock writes output")
        submit.add_argument("--region", default="us-east-1")
        submit.add_argument("--job-name", default=None)
        submit.add_argument("--timeout-hours", type=int, default=None)

    status = sub.add_parser("status")
    status.add_argument("--batch-dir", required=True, type=Path)
    status.add_argument("--stage", required=True, choices=("ranking", "rationale"))
    status.add_argument("--region", default="us-east-1")

    for name in ("fetch-ranking", "fetch-rationale"):
        fetch = sub.add_parser(name)
        fetch.add_argument("--batch-dir", required=True, type=Path)
        fetch.add_argument("--region", default="us-east-1")
        fetch.add_argument(
            "--results",
            type=Path,
            default=None,
            help="local results JSONL; omit to download from the job's S3 output URI",
        )
        if name == "fetch-rationale":
            fetch.add_argument("--top-k-rationale", type=int, default=10)

    review_ranking = sub.add_parser("review-ranking")
    review_ranking.add_argument("--batch-dir", required=True, type=Path)
    review_ranking.add_argument(
        "--approve-valid",
        action="store_true",
        help="mark rows without validation errors as approved after human inspection",
    )

    prepare_rationale = sub.add_parser("prepare-rationale")
    prepare_rationale.add_argument("--batch-dir", required=True, type=Path)
    prepare_rationale.add_argument("--top-k-rationale", type=int, default=10)
    prepare_rationale.add_argument("--max-sentences", type=int, default=16)
    prepare_rationale.add_argument("--max-tokens", type=int, default=32000)
    prepare_rationale.add_argument("--temperature", type=float, default=0.0)
    prepare_rationale.add_argument(
        "--include-pending",
        action="store_true",
        help="also use pending review rows; approved-only is the default",
    )

    review_labels = sub.add_parser("review-labels")
    review_labels.add_argument("--batch-dir", required=True, type=Path)
    review_labels.add_argument(
        "--approve-valid",
        action="store_true",
        help="mark preview rows without validation errors as approved after human inspection",
    )

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--batch-dir", required=True, type=Path)
    finalize.add_argument("--labels", required=True, type=Path)
    finalize.add_argument("--overwrite", action="store_true")
    finalize.add_argument(
        "--include-pending",
        action="store_true",
        help="write pending preview rows too; approved-only is the default",
    )

    args = parser.parse_args()

    if args.command == "prepare-ranking":
        result = prepare_ranking_batch(
            batch_dir=args.batch_dir,
            snapshots_dir=args.snapshots,
            model_config=_model_config(args),
            max_sentences_per_book=args.max_sentences,
            limit=args.limit,
        )
    elif args.command == "submit-ranking":
        result = submit_batch_stage(
            batch_dir=args.batch_dir,
            stage="ranking",
            client=_client(args),
            role_arn=args.role_arn,
            model_id=args.model_id,
            s3_input_prefix=args.s3_input,
            s3_output_uri=args.s3_output,
            job_name=args.job_name,
            timeout_hours=args.timeout_hours,
        )
    elif args.command == "status":
        result = status_batch_stage(
            batch_dir=args.batch_dir,
            stage=args.stage,
            client=_client(args),
        )
    elif args.command == "fetch-ranking":
        result = fetch_batch_stage(
            batch_dir=args.batch_dir,
            stage="ranking",
            client=_client(args),
            results_path=args.results,
        )
    elif args.command == "review-ranking":
        review_path = args.batch_dir / "ranking" / "review.jsonl"
        result = _review_summary(review_path)
        if args.approve_valid:
            result.update(approve_review_file(review_path))
    elif args.command == "prepare-rationale":
        result = prepare_rationale_batch(
            batch_dir=args.batch_dir,
            model_config=_model_config(args),
            top_k_rationale=args.top_k_rationale,
            max_sentences_per_book=args.max_sentences,
            include_pending=args.include_pending,
        )
    elif args.command == "submit-rationale":
        result = submit_batch_stage(
            batch_dir=args.batch_dir,
            stage="rationale",
            client=_client(args),
            role_arn=args.role_arn,
            model_id=args.model_id,
            s3_input_prefix=args.s3_input,
            s3_output_uri=args.s3_output,
            job_name=args.job_name,
            timeout_hours=args.timeout_hours,
        )
    elif args.command == "fetch-rationale":
        result = fetch_batch_stage(
            batch_dir=args.batch_dir,
            stage="rationale",
            client=_client(args),
            results_path=args.results,
            top_k_rationale=args.top_k_rationale,
        )
    elif args.command == "review-labels":
        review_path = args.batch_dir / "labels.preview.jsonl"
        result = _review_summary(review_path)
        if args.approve_valid:
            result.update(approve_review_file(review_path))
    elif args.command == "finalize":
        result = finalize_labels(
            batch_dir=args.batch_dir,
            labels_dir=args.labels,
            overwrite=args.overwrite,
            include_pending=args.include_pending,
        )
    else:  # pragma: no cover - argparse keeps this unreachable
        parser.error(f"unknown command: {args.command}")

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
