from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from explainable_reranker.data.sentence_index import IndexedSentence, build_sentence_index
from explainable_reranker.teacher.llm_client import (
    BedrockClaudeChatModel,
    extract_json_object,
)
from explainable_reranker.teacher.prompts import (
    SYSTEM_INSTRUCTIONS,
    build_listwise_prompt,
    build_rationale_prompt,
)
from explainable_reranker.teacher.schemas import (
    TeacherLabel,
    parse_teacher_label,
    validate_teacher_label,
)
from explainable_reranker.topa.adapter import TopaPageResponse, parse_topa_page_response

BatchStage = Literal["ranking", "rationale"]

STATUS_PREPARED = "prepared"
STATUS_SUBMITTED = "submitted"
STATUS_RECEIVED = "received"
STATUS_FAILED = "failed"

REVIEW_PENDING = "pending_review"
REVIEW_APPROVED = "approved"
REVIEW_REJECTED = "rejected"
REVIEW_AUTO_REJECTED = "auto_rejected"


@dataclass(frozen=True)
class BatchModelConfig:
    """Anthropic Messages body settings for Bedrock batch inference."""

    anthropic_version: str = BedrockClaudeChatModel.ANTHROPIC_VERSION
    max_tokens: int = 32000
    temperature: float = 0.0


@dataclass(frozen=True)
class ManifestEntry:
    response_id: str
    query_id: str
    snapshot_path: str
    stage: BatchStage
    record_id: str
    prompt_sha256: str
    status: str = STATUS_PREPARED
    provider_job_arn: str | None = None
    error: str | None = None
    review_status: str = REVIEW_PENDING

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pass"] = payload.pop("stage")
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ManifestEntry":
        data = dict(payload)
        data["stage"] = data.pop("pass", data.get("stage"))
        return cls(**data)


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str

    def as_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}" if self.key else f"s3://{self.bucket}"


class BedrockBatchClient:
    """Tiny AWS boundary for Bedrock Batch Inference and S3 artifacts.

    `boto3` is imported lazily so local tests and the dependency-free package path
    keep working without AWS libraries installed.
    """

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        bedrock_client: object | None = None,
        s3_client: object | None = None,
    ):
        self.region = region
        self._bedrock_client = bedrock_client
        self._s3_client = s3_client

    def _bedrock(self) -> object:
        if self._bedrock_client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - prod-only path
                raise RuntimeError(
                    "boto3 is required for Bedrock batch jobs; install the bedrock extra."
                ) from exc
            self._bedrock_client = boto3.client("bedrock", region_name=self.region)
        return self._bedrock_client

    def _s3(self) -> object:
        if self._s3_client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - prod-only path
                raise RuntimeError(
                    "boto3 is required for Bedrock batch jobs; install the bedrock extra."
                ) from exc
            self._s3_client = boto3.client("s3", region_name=self.region)
        return self._s3_client

    def upload_file(self, local_path: Path, s3_uri: str) -> str:
        parsed = parse_s3_uri(s3_uri)
        self._s3().upload_file(str(local_path), parsed.bucket, parsed.key)
        return parsed.as_uri()

    def submit_model_invocation_job(
        self,
        *,
        job_name: str,
        role_arn: str,
        model_id: str,
        input_s3_uri: str,
        output_s3_uri: str,
        timeout_hours: int | None = None,
        client_request_token: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "jobName": job_name,
            "roleArn": role_arn,
            "modelId": model_id,
            "modelInvocationType": "InvokeModel",
            "inputDataConfig": {
                "s3InputDataConfig": {"s3Uri": input_s3_uri},
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": output_s3_uri},
            },
        }
        if timeout_hours is not None:
            kwargs["timeoutDurationInHours"] = timeout_hours
        if client_request_token is not None:
            kwargs["clientRequestToken"] = client_request_token
        response = self._bedrock().create_model_invocation_job(**kwargs)
        return str(response["jobArn"])

    def get_job(self, job_arn: str) -> dict[str, Any]:
        return dict(self._bedrock().get_model_invocation_job(jobIdentifier=job_arn))

    def download_jsonl_results(self, output_s3_uri: str, local_path: Path) -> Path:
        parsed = parse_s3_uri(output_s3_uri)
        s3 = self._s3()
        keys = _list_result_keys(s3, parsed.bucket, parsed.key)
        if not keys:
            raise FileNotFoundError(f"no Bedrock JSONL result objects under {output_s3_uri}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as handle:
            for key in keys:
                body = s3.get_object(Bucket=parsed.bucket, Key=key)["Body"].read()
                handle.write(body)
                if body and not body.endswith(b"\n"):
                    handle.write(b"\n")
        return local_path


def ensure_batch_layout(batch_dir: Path) -> None:
    for relative in (
        "ranking",
        "rationale",
        "intermediate/ranking",
    ):
        (batch_dir / relative).mkdir(parents=True, exist_ok=True)


def prepare_ranking_batch(
    *,
    batch_dir: Path,
    snapshots_dir: Path,
    model_config: BatchModelConfig | None = None,
    max_sentences_per_book: int = 16,
    limit: int | None = None,
) -> dict[str, int]:
    ensure_batch_layout(batch_dir)
    model_config = model_config or BatchModelConfig()
    snapshot_paths = list(iter_snapshot_paths(snapshots_dir))
    if limit is not None:
        snapshot_paths = snapshot_paths[:limit]
    if not snapshot_paths:
        raise ValueError(f"no snapshot envelopes found under {snapshots_dir}")

    input_rows: list[dict[str, Any]] = []
    manifest_entries: list[ManifestEntry] = []
    for snapshot_path in snapshot_paths:
        response = load_snapshot_response(snapshot_path)
        sentence_index = build_sentence_index(response)
        prompt = build_listwise_prompt(
            response,
            sentence_index,
            max_sentences_per_book=max_sentences_per_book,
        )
        record_id = make_record_id(response.response_id, "ranking")
        input_rows.append(
            make_bedrock_record(
                record_id=record_id,
                prompt=prompt,
                model_config=model_config,
            )
        )
        manifest_entries.append(
            ManifestEntry(
                response_id=response.response_id,
                query_id=response.query_id,
                snapshot_path=str(snapshot_path),
                stage="ranking",
                record_id=record_id,
                prompt_sha256=sha256_text(prompt),
            )
        )

    write_jsonl(batch_dir / "ranking" / "input.jsonl", input_rows)
    upsert_manifest_entries(batch_dir, manifest_entries)
    write_summary(batch_dir, "prepare-ranking", {"prepared": len(input_rows)})
    return {"prepared": len(input_rows)}


def ingest_ranking_results(*, batch_dir: Path, results_path: Path) -> dict[str, int]:
    ensure_batch_layout(batch_dir)
    target_results = batch_dir / "ranking" / "results.jsonl"
    if results_path.resolve() != target_results.resolve():
        shutil.copyfile(results_path, target_results)

    manifest = read_manifest(batch_dir)
    entries_by_record = {
        entry.record_id: entry for entry in manifest if entry.stage == "ranking"
    }
    review_rows: list[dict[str, Any]] = []
    updates: list[ManifestEntry] = []
    failures = 0
    received = 0

    for row in read_jsonl(target_results):
        record_id = str(row.get("recordId", ""))
        entry = entries_by_record.get(record_id)
        if entry is None:
            failures += 1
            append_failure(batch_dir, "ranking", record_id, "", "result has no manifest entry")
            continue

        try:
            if "error" in row:
                raise ValueError(json.dumps(row["error"], ensure_ascii=False, sort_keys=True))
            text = extract_bedrock_output_text(row.get("modelOutput", {}))
            ranking_payload = extract_json_object(text)
            ranking_payload.setdefault("rationales", {})
            response = load_snapshot_response(Path(entry.snapshot_path))
            errors = validate_ranking_payload(ranking_payload, response)
        except Exception as exc:  # noqa: BLE001 - surfaced in failure artifacts
            failures += 1
            error = str(exc)
            append_failure(batch_dir, "ranking", record_id, entry.response_id, error)
            review_rows.append(
                review_row_for_failure(entry, error, stage="ranking")
            )
            updates.append(
                _replace_entry(
                    entry,
                    status=STATUS_FAILED,
                    error=error,
                    review_status=REVIEW_AUTO_REJECTED,
                )
            )
            continue

        review_status = REVIEW_PENDING if not errors else REVIEW_AUTO_REJECTED
        if errors:
            failures += 1
            append_failure(batch_dir, "ranking", record_id, entry.response_id, "; ".join(errors))
        else:
            received += 1
            write_json(
                batch_dir / "intermediate" / "ranking" / f"{entry.response_id}.json",
                ranking_payload,
            )
        review_rows.append(make_ranking_review_row(entry, response, ranking_payload, errors))
        updates.append(
            _replace_entry(
                entry,
                status=STATUS_RECEIVED if not errors else STATUS_FAILED,
                error="; ".join(errors) if errors else None,
                review_status=review_status,
            )
        )

    write_jsonl(batch_dir / "ranking" / "review.jsonl", review_rows)
    upsert_manifest_entries(batch_dir, updates)
    result = {"received": received, "failed": failures, "review_rows": len(review_rows)}
    write_summary(batch_dir, "fetch-ranking", result)
    return result


def approve_review_file(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    approved = 0
    rejected = 0
    for row in rows:
        if row.get("errors"):
            row["review_status"] = REVIEW_AUTO_REJECTED
            rejected += 1
        elif row.get("review_status") == REVIEW_PENDING:
            row["review_status"] = REVIEW_APPROVED
            approved += 1
    write_jsonl(path, rows)
    return {"approved": approved, "rejected": rejected}


def prepare_rationale_batch(
    *,
    batch_dir: Path,
    model_config: BatchModelConfig | None = None,
    top_k_rationale: int = 10,
    max_sentences_per_book: int = 16,
    include_pending: bool = False,
) -> dict[str, int]:
    ensure_batch_layout(batch_dir)
    model_config = model_config or BatchModelConfig()
    review_rows = read_jsonl(batch_dir / "ranking" / "review.jsonl")
    allowed_statuses = {REVIEW_APPROVED}
    if include_pending:
        allowed_statuses.add(REVIEW_PENDING)

    input_rows: list[dict[str, Any]] = []
    manifest_entries: list[ManifestEntry] = []
    for review in review_rows:
        if review.get("review_status") not in allowed_statuses:
            continue
        if review.get("errors"):
            continue
        response_id = str(review["response_id"])
        snapshot_path = Path(str(review["snapshot_path"]))
        ranking_path = batch_dir / "intermediate" / "ranking" / f"{response_id}.json"
        ranking_payload = read_json(ranking_path)
        response = load_snapshot_response(snapshot_path)
        sentence_index = build_sentence_index(response)
        ranked_book_ids = ranked_book_ids_from_payload(ranking_payload)
        prompt = build_rationale_prompt(
            response,
            sentence_index,
            ranked_book_ids=ranked_book_ids,
            top_k=top_k_rationale,
            max_sentences_per_book=max_sentences_per_book,
        )
        record_id = make_record_id(response.response_id, "rationale")
        input_rows.append(
            make_bedrock_record(
                record_id=record_id,
                prompt=prompt,
                model_config=model_config,
            )
        )
        manifest_entries.append(
            ManifestEntry(
                response_id=response.response_id,
                query_id=response.query_id,
                snapshot_path=str(snapshot_path),
                stage="rationale",
                record_id=record_id,
                prompt_sha256=sha256_text(prompt),
            )
        )

    if not input_rows:
        raise ValueError("no approved ranking rows available for rationale batch")
    write_jsonl(batch_dir / "rationale" / "input.jsonl", input_rows)
    upsert_manifest_entries(batch_dir, manifest_entries)
    result = {"prepared": len(input_rows)}
    write_summary(batch_dir, "prepare-rationale", result)
    return result


def ingest_rationale_results(
    *,
    batch_dir: Path,
    results_path: Path,
    top_k_rationale: int = 10,
) -> dict[str, int]:
    ensure_batch_layout(batch_dir)
    target_results = batch_dir / "rationale" / "results.jsonl"
    if results_path.resolve() != target_results.resolve():
        shutil.copyfile(results_path, target_results)

    manifest = read_manifest(batch_dir)
    entries_by_record = {
        entry.record_id: entry for entry in manifest if entry.stage == "rationale"
    }
    preview_rows: list[dict[str, Any]] = []
    updates: list[ManifestEntry] = []
    failures = 0
    received = 0

    for row in read_jsonl(target_results):
        record_id = str(row.get("recordId", ""))
        entry = entries_by_record.get(record_id)
        if entry is None:
            failures += 1
            append_failure(batch_dir, "rationale", record_id, "", "result has no manifest entry")
            continue
        try:
            if "error" in row:
                raise ValueError(json.dumps(row["error"], ensure_ascii=False, sort_keys=True))
            text = extract_bedrock_output_text(row.get("modelOutput", {}))
            rationale_payload = extract_json_object(text)
            ranking_payload = read_json(
                batch_dir / "intermediate" / "ranking" / f"{entry.response_id}.json"
            )
            label_payload = {
                "ranking": ranking_payload.get("ranking", []),
                "rationales": rationale_payload.get("rationales", {}),
            }
            response = load_snapshot_response(Path(entry.snapshot_path))
            sentence_index = build_sentence_index(response)
            label = parse_teacher_label(
                label_payload,
                query_id=response.query_id,
                response_id=response.response_id,
            )
            errors = validate_label_for_response(
                label,
                response,
                sentence_index,
                top_k_rationale=top_k_rationale,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in failure artifacts
            failures += 1
            error = str(exc)
            append_failure(batch_dir, "rationale", record_id, entry.response_id, error)
            preview_rows.append(review_row_for_failure(entry, error, stage="rationale"))
            updates.append(
                _replace_entry(
                    entry,
                    status=STATUS_FAILED,
                    error=error,
                    review_status=REVIEW_AUTO_REJECTED,
                )
            )
            continue

        review_status = REVIEW_PENDING if not errors else REVIEW_AUTO_REJECTED
        if errors:
            failures += 1
            append_failure(batch_dir, "rationale", record_id, entry.response_id, "; ".join(errors))
        else:
            received += 1
        preview_rows.append(make_label_preview_row(entry, response, label.raw, errors))
        updates.append(
            _replace_entry(
                entry,
                status=STATUS_RECEIVED if not errors else STATUS_FAILED,
                error="; ".join(errors) if errors else None,
                review_status=review_status,
            )
        )

    write_jsonl(batch_dir / "labels.preview.jsonl", preview_rows)
    upsert_manifest_entries(batch_dir, updates)
    result = {"received": received, "failed": failures, "preview_rows": len(preview_rows)}
    write_summary(batch_dir, "fetch-rationale", result)
    return result


def finalize_labels(
    *,
    batch_dir: Path,
    labels_dir: Path,
    overwrite: bool = False,
    include_pending: bool = False,
) -> dict[str, int]:
    labels_dir.mkdir(parents=True, exist_ok=True)
    allowed_statuses = {REVIEW_APPROVED}
    if include_pending:
        allowed_statuses.add(REVIEW_PENDING)

    written = 0
    skipped_review = 0
    skipped_existing = 0
    for row in read_jsonl(batch_dir / "labels.preview.jsonl"):
        if row.get("errors") or row.get("review_status") not in allowed_statuses:
            skipped_review += 1
            continue
        response_id = str(row["response_id"])
        label_path = labels_dir / f"{response_id}.json"
        if label_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        write_json(label_path, row["label"])
        written += 1

    result = {
        "written": written,
        "skipped_review": skipped_review,
        "skipped_existing": skipped_existing,
    }
    write_summary(batch_dir, "finalize", result)
    return result


def submit_batch_stage(
    *,
    batch_dir: Path,
    stage: BatchStage,
    client: BedrockBatchClient,
    role_arn: str,
    model_id: str,
    s3_input_prefix: str,
    s3_output_uri: str,
    job_name: str | None = None,
    timeout_hours: int | None = None,
) -> dict[str, str]:
    input_path = batch_dir / stage / "input.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"missing batch input file: {input_path}")
    batch_id = batch_dir.name
    input_s3_uri = join_s3_uri(s3_input_prefix, batch_id, stage, "input.jsonl")
    uploaded_uri = client.upload_file(input_path, input_s3_uri)
    final_job_name = job_name or f"teacher-{batch_id}-{stage}"
    token = sha256_text(f"{final_job_name}\n{uploaded_uri}\n{s3_output_uri}")[:64]
    job_arn = client.submit_model_invocation_job(
        job_name=final_job_name,
        role_arn=role_arn,
        model_id=model_id,
        input_s3_uri=uploaded_uri,
        output_s3_uri=s3_output_uri,
        timeout_hours=timeout_hours,
        client_request_token=token,
    )
    write_job_metadata(
        batch_dir,
        stage,
        {
            "job_name": final_job_name,
            "job_arn": job_arn,
            "input_s3_uri": uploaded_uri,
            "output_s3_uri": s3_output_uri,
            "model_id": model_id,
            "submitted_at": datetime.now(UTC).isoformat(),
        },
    )
    update_manifest_stage(batch_dir, stage, status=STATUS_SUBMITTED, provider_job_arn=job_arn)
    result = {"job_arn": job_arn, "input_s3_uri": uploaded_uri, "output_s3_uri": s3_output_uri}
    write_summary(batch_dir, f"submit-{stage}", result)
    return result


def fetch_batch_stage(
    *,
    batch_dir: Path,
    stage: BatchStage,
    client: BedrockBatchClient,
    results_path: Path | None = None,
    top_k_rationale: int = 10,
) -> dict[str, int]:
    target = batch_dir / stage / "results.jsonl"
    if results_path is None:
        job = read_job_metadata(batch_dir).get(stage)
        if not job:
            raise ValueError(f"no {stage} job metadata found")
        client.download_jsonl_results(str(job["output_s3_uri"]), target)
        results_path = target
    if stage == "ranking":
        return ingest_ranking_results(batch_dir=batch_dir, results_path=results_path)
    return ingest_rationale_results(
        batch_dir=batch_dir,
        results_path=results_path,
        top_k_rationale=top_k_rationale,
    )


def status_batch_stage(*, batch_dir: Path, stage: BatchStage, client: BedrockBatchClient) -> dict[str, Any]:
    job = read_job_metadata(batch_dir).get(stage)
    if not job:
        raise ValueError(f"no {stage} job metadata found")
    status = client.get_job(str(job["job_arn"]))
    write_summary(batch_dir, f"status-{stage}", status)
    return status


def make_bedrock_record(
    *,
    record_id: str,
    prompt: str,
    model_config: BatchModelConfig,
) -> dict[str, Any]:
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": model_config.anthropic_version,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "system": SYSTEM_INSTRUCTIONS,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        },
    }


def extract_bedrock_output_text(model_output: dict[str, Any]) -> str:
    blocks = model_output.get("content")
    if isinstance(blocks, list):
        return "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    output = model_output.get("output")
    if isinstance(output, dict):
        message = output.get("message", {})
        content = message.get("content", [])
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict)
            )
    raise ValueError("Bedrock result has no text content")


def validate_ranking_payload(payload: dict[str, Any], response: TopaPageResponse) -> list[str]:
    try:
        label = parse_teacher_label(
            {"ranking": payload.get("ranking", []), "rationales": {}},
            query_id=response.query_id,
            response_id=response.response_id,
        )
    except ValueError as exc:
        return [str(exc)]
    return validate_teacher_label(
        label,
        candidate_book_ids={candidate.book_id for candidate in response.candidates},
        sentence_ids_by_book={},
        require_rationales_for_top_k=0,
    )


def validate_label_for_response(
    label: TeacherLabel,
    response: TopaPageResponse,
    sentence_index: list[IndexedSentence],
    *,
    top_k_rationale: int,
) -> list[str]:
    sentence_ids_by_book: dict[str, set[str]] = {}
    for sentence in sentence_index:
        sentence_ids_by_book.setdefault(sentence.book_id, set()).add(sentence.sentence_id)
    return validate_teacher_label(
        label,
        candidate_book_ids={candidate.book_id for candidate in response.candidates},
        sentence_ids_by_book=sentence_ids_by_book,
        require_rationales_for_top_k=top_k_rationale,
    )


def make_ranking_review_row(
    entry: ManifestEntry,
    response: TopaPageResponse,
    ranking_payload: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    candidate_by_id = {candidate.book_id: candidate for candidate in response.candidates}
    ranking = ranking_payload.get("ranking", [])
    return {
        "response_id": entry.response_id,
        "query_id": entry.query_id,
        "query": response.query,
        "record_id": entry.record_id,
        "snapshot_path": entry.snapshot_path,
        "review_status": REVIEW_PENDING if not errors else REVIEW_AUTO_REJECTED,
        "errors": errors,
        "top": _ranking_slice(ranking, candidate_by_id, start=0, stop=5),
        "bottom": _ranking_slice(ranking, candidate_by_id, start=max(0, len(ranking) - 5), stop=len(ranking)),
        "ranking": ranking_payload,
    }


def make_label_preview_row(
    entry: ManifestEntry,
    response: TopaPageResponse,
    label_payload: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "response_id": entry.response_id,
        "query_id": entry.query_id,
        "query": response.query,
        "record_id": entry.record_id,
        "snapshot_path": entry.snapshot_path,
        "review_status": REVIEW_PENDING if not errors else REVIEW_AUTO_REJECTED,
        "errors": errors,
        "top_books": ranked_book_ids_from_payload(label_payload)[:10],
        "label": label_payload,
    }


def review_row_for_failure(entry: ManifestEntry, error: str, *, stage: BatchStage) -> dict[str, Any]:
    return {
        "response_id": entry.response_id,
        "query_id": entry.query_id,
        "record_id": entry.record_id,
        "snapshot_path": entry.snapshot_path,
        "review_status": REVIEW_AUTO_REJECTED,
        "errors": [error],
        "stage": stage,
    }


def ranked_book_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in payload.get("ranking", []):
        if not isinstance(item, dict):
            continue
        book_id = str(item.get("book") or item.get("book_id") or "")
        if not book_id:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scored.append((score, book_id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [book_id for _score, book_id in scored]


def iter_snapshot_paths(snapshots_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(snapshots_dir.glob("*/*.json")):
        if path.name.endswith(".json"):
            paths.append(path)
    return paths


def load_snapshot_response(snapshot_path: Path) -> TopaPageResponse:
    envelope = read_json(snapshot_path)
    payload = envelope.get("payload", envelope)
    return parse_topa_page_response(payload)


def make_record_id(response_id: str, stage: BatchStage) -> str:
    digest = hashlib.sha256(f"{stage}\n{response_id}".encode("utf-8")).hexdigest()[:20]
    prefix = "rnk" if stage == "ranking" else "rat"
    return f"{prefix}_{digest}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_s3_uri(uri: str) -> S3Uri:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[5:]
    bucket, sep, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri}")
    if not sep:
        key = ""
    return S3Uri(bucket=bucket, key=key)


def join_s3_uri(prefix: str, *parts: str) -> str:
    parsed = parse_s3_uri(prefix)
    clean_parts = [part.strip("/") for part in parts if part.strip("/")]
    key_parts = [parsed.key.strip("/")] if parsed.key.strip("/") else []
    key = "/".join(key_parts + clean_parts)
    return S3Uri(parsed.bucket, key).as_uri()


def read_manifest(batch_dir: Path) -> list[ManifestEntry]:
    path = batch_dir / "manifest.jsonl"
    if not path.exists():
        return []
    return [ManifestEntry.from_json(row) for row in read_jsonl(path)]


def write_manifest(batch_dir: Path, entries: list[ManifestEntry]) -> None:
    rows = [entry.to_json() for entry in sorted(entries, key=lambda item: (item.stage, item.record_id))]
    write_jsonl(batch_dir / "manifest.jsonl", rows)


def upsert_manifest_entries(batch_dir: Path, entries: list[ManifestEntry]) -> None:
    current = {(entry.stage, entry.record_id): entry for entry in read_manifest(batch_dir)}
    for entry in entries:
        current[(entry.stage, entry.record_id)] = entry
    write_manifest(batch_dir, list(current.values()))


def update_manifest_stage(
    batch_dir: Path,
    stage: BatchStage,
    *,
    status: str,
    provider_job_arn: str | None = None,
) -> None:
    updates: list[ManifestEntry] = []
    for entry in read_manifest(batch_dir):
        if entry.stage == stage:
            updates.append(_replace_entry(entry, status=status, provider_job_arn=provider_job_arn))
    upsert_manifest_entries(batch_dir, updates)


def read_job_metadata(batch_dir: Path) -> dict[str, Any]:
    path = batch_dir / "job.json"
    if not path.exists():
        return {}
    return read_json(path)


def write_job_metadata(batch_dir: Path, stage: BatchStage, payload: dict[str, Any]) -> None:
    metadata = read_job_metadata(batch_dir)
    metadata[stage] = payload
    write_json(batch_dir / "job.json", metadata)


def append_failure(batch_dir: Path, stage: BatchStage, record_id: str, response_id: str, error: str) -> None:
    path = batch_dir / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "stage": stage,
                    "record_id": record_id,
                    "response_id": response_id,
                    "error": error,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def write_summary(batch_dir: Path, command: str, payload: dict[str, Any]) -> None:
    summary = read_json(batch_dir / "summary.json") if (batch_dir / "summary.json").exists() else {}
    summary[command] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "result": payload,
    }
    write_json(batch_dir / "summary.json", summary)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _replace_entry(entry: ManifestEntry, **changes: Any) -> ManifestEntry:
    payload = entry.to_json()
    payload.update(changes)
    payload["pass"] = payload.pop("stage", payload.get("pass"))
    return ManifestEntry.from_json(payload)


def _ranking_slice(
    ranking: list[Any],
    candidate_by_id: dict[str, Any],
    *,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in ranking[start:stop]:
        if not isinstance(item, dict):
            continue
        book_id = str(item.get("book") or item.get("book_id") or "")
        candidate = candidate_by_id.get(book_id)
        rows.append(
            {
                "book": book_id,
                "title": candidate.title if candidate is not None else "",
                "score": item.get("score"),
            }
        )
    return rows


def _list_result_keys(s3_client: object, bucket: str, prefix: str) -> list[str]:
    clean_prefix = prefix.lstrip("/")
    keys: list[str] = []
    if hasattr(s3_client, "get_paginator"):
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=clean_prefix):
            keys.extend(_result_keys_from_page(page))
    else:
        page = s3_client.list_objects_v2(Bucket=bucket, Prefix=clean_prefix)
        keys.extend(_result_keys_from_page(page))
    return sorted(keys)


def _result_keys_from_page(page: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for item in page.get("Contents", []):
        key = str(item.get("Key", ""))
        if key.endswith("manifest.json.out"):
            continue
        if key.endswith(".jsonl.out") or key.endswith(".jsonl"):
            keys.append(key)
    return keys
