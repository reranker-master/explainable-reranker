#!/usr/bin/env python3
"""Step-by-step Bedrock Batch teacher labeling.

This does not replace ``scripts/collect_and_label.py``. Use the existing sync
script for small pilots/debugging, and use this batch CLI for large runs where a
human reviews artifacts between stages.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from explainable_reranker.config.env import load_project_dotenv
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


def _env_path(name: str, default: str | None = None) -> Path | None:
    value = os.environ.get(name, default)
    return Path(value) if value else None


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _aws_region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


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
    load_project_dotenv()
    parser = argparse.ArgumentParser(
        description="Prepare, submit, review, and finalize Bedrock teacher batches."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_ranking = sub.add_parser("prepare-ranking")
    prepare_ranking.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    prepare_ranking.add_argument("--snapshots", required=False, type=Path, default=_env_path("TEACHER_SNAPSHOTS_DIR", "data/snapshots"))
    prepare_ranking.add_argument("--max-sentences", type=int, default=_env_int("TEACHER_MAX_SENTENCES", 16))
    prepare_ranking.add_argument("--max-tokens", type=int, default=_env_int("BEDROCK_MAX_TOKENS", 32000))
    prepare_ranking.add_argument("--temperature", type=float, default=_env_float("BEDROCK_TEMPERATURE", 0.0))
    prepare_ranking.add_argument("--limit", type=int, default=None)

    for name in ("submit-ranking", "submit-rationale"):
        submit = sub.add_parser(name)
        submit.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
        submit.add_argument("--role-arn", required=os.environ.get("BEDROCK_ROLE_ARN") is None, default=os.environ.get("BEDROCK_ROLE_ARN"))
        submit.add_argument("--model-id", required=os.environ.get("BEDROCK_MODEL_ID") is None, default=os.environ.get("BEDROCK_MODEL_ID"))
        submit.add_argument("--s3-input", required=os.environ.get("BEDROCK_BATCH_INPUT_S3") is None, default=os.environ.get("BEDROCK_BATCH_INPUT_S3"), help="S3 prefix for input JSONL upload")
        submit.add_argument("--s3-output", required=os.environ.get("BEDROCK_BATCH_OUTPUT_S3") is None, default=os.environ.get("BEDROCK_BATCH_OUTPUT_S3"), help="S3 prefix where Bedrock writes output")
        submit.add_argument("--region", default=_aws_region())
        submit.add_argument("--job-name", default=None)
        submit.add_argument("--timeout-hours", type=int, default=_env_int("BEDROCK_TIMEOUT_HOURS"))

    status = sub.add_parser("status")
    status.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    status.add_argument("--stage", required=True, choices=("ranking", "rationale"))
    status.add_argument("--region", default=_aws_region())

    for name in ("fetch-ranking", "fetch-rationale"):
        fetch = sub.add_parser(name)
        fetch.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
        fetch.add_argument("--region", default=_aws_region())
        fetch.add_argument(
            "--results",
            type=Path,
            default=None,
            help="local results JSONL; omit to download from the job's S3 output URI",
        )
        if name == "fetch-rationale":
            fetch.add_argument("--top-k-rationale", type=int, default=10)

    review_ranking = sub.add_parser("review-ranking")
    review_ranking.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    review_ranking.add_argument(
        "--approve-valid",
        action="store_true",
        help="mark rows without validation errors as approved after human inspection",
    )

    prepare_rationale = sub.add_parser("prepare-rationale")
    prepare_rationale.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    prepare_rationale.add_argument("--top-k-rationale", type=int, default=10)
    prepare_rationale.add_argument("--max-sentences", type=int, default=_env_int("TEACHER_MAX_SENTENCES", 16))
    prepare_rationale.add_argument("--max-tokens", type=int, default=_env_int("BEDROCK_MAX_TOKENS", 32000))
    prepare_rationale.add_argument("--temperature", type=float, default=_env_float("BEDROCK_TEMPERATURE", 0.0))
    prepare_rationale.add_argument(
        "--include-pending",
        action="store_true",
        help="also use pending review rows; approved-only is the default",
    )

    review_labels = sub.add_parser("review-labels")
    review_labels.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    review_labels.add_argument(
        "--approve-valid",
        action="store_true",
        help="mark preview rows without validation errors as approved after human inspection",
    )

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--batch-dir", required=_env_path("TEACHER_BATCH_DIR") is None, type=Path, default=_env_path("TEACHER_BATCH_DIR"))
    finalize.add_argument("--labels", required=False, type=Path, default=_env_path("TEACHER_LABELS_DIR", "data/labels"))
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
