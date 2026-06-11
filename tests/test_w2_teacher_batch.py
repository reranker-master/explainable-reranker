from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.data.snapshot_store import SnapshotStore
from explainable_reranker.teacher.batch import (
    BatchModelConfig,
    BedrockBatchClient,
    approve_review_file,
    fetch_batch_stage,
    finalize_labels,
    make_record_id,
    prepare_ranking_batch,
    prepare_rationale_batch,
    read_jsonl,
    submit_batch_stage,
    write_jsonl,
)
from explainable_reranker.teacher.label_ranking import HeuristicRankingTeacher
from explainable_reranker.teacher.label_rationale import HeuristicRationaleTeacher
from explainable_reranker.topa.adapter import parse_topa_page_response


FIXTURE = Path(__file__).parent / "fixtures" / "topa_page_response.json"


class _FakeBedrockControl:
    def __init__(self) -> None:
        self.last_create: dict | None = None
        self.status = {"status": "Completed"}

    def create_model_invocation_job(self, **kwargs):
        self.last_create = kwargs
        return {"jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/demo"}

    def get_model_invocation_job(self, **kwargs):
        return dict(self.status, **kwargs)


class _FakeS3:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        self.uploads.append((local_path, bucket, key))


class TeacherBatchTest(unittest.TestCase):
    def _payload(self, response_id: str = "resp_demo_001") -> dict:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["response_id"] = response_id
        payload["query_id"] = response_id.replace("resp", "q")
        return payload

    def _save_snapshots(self, root: Path, count: int = 1) -> list:
        store = SnapshotStore(root)
        responses = []
        for idx in range(count):
            payload = self._payload(f"resp_batch_{idx + 1:03d}")
            store.save(payload, request_timestamp="2026-06-11T00:00:00+00:00")
            responses.append(parse_topa_page_response(payload))
        return responses

    def _ranking_payload(self, response) -> dict:
        index = build_sentence_index(response)
        return HeuristicRankingTeacher().label(response, index)

    def _rationale_payload(self, response, ranking_payload: dict) -> dict:
        index = build_sentence_index(response)
        ranked = [item["book"] for item in ranking_payload["ranking"]]
        return HeuristicRationaleTeacher().label(response, index, ranked)

    def _bedrock_line(self, record_id: str, payload: dict) -> dict:
        return {
            "recordId": record_id,
            "modelOutput": {
                "content": [
                    {
                        "type": "text",
                        "text": "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```",
                    }
                ]
            },
        }

    def test_prepare_ranking_writes_bedrock_jsonl_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            batch_dir = root / "teacher_batches" / "batch-a"
            response = self._save_snapshots(snapshots)[0]

            result = prepare_ranking_batch(
                batch_dir=batch_dir,
                snapshots_dir=snapshots,
                model_config=BatchModelConfig(max_tokens=1234, temperature=0.2),
            )

            self.assertEqual(result, {"prepared": 1})
            rows = read_jsonl(batch_dir / "ranking" / "input.jsonl")
            self.assertEqual(rows[0]["recordId"], make_record_id(response.response_id, "ranking"))
            self.assertEqual(rows[0]["modelInput"]["max_tokens"], 1234)
            self.assertIn("Task A: rank", rows[0]["modelInput"]["messages"][0]["content"][0]["text"])

            manifest = read_jsonl(batch_dir / "manifest.jsonl")
            self.assertEqual(manifest[0]["pass"], "ranking")
            self.assertEqual(manifest[0]["review_status"], "pending_review")
            self.assertTrue(manifest[0]["prompt_sha256"])

    def test_submit_uses_bedrock_batch_job_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            batch_dir = root / "teacher_batches" / "batch-a"
            self._save_snapshots(snapshots)
            prepare_ranking_batch(batch_dir=batch_dir, snapshots_dir=snapshots)

            fake_bedrock = _FakeBedrockControl()
            fake_s3 = _FakeS3()
            client = BedrockBatchClient(
                region="us-east-1",
                bedrock_client=fake_bedrock,
                s3_client=fake_s3,
            )
            result = submit_batch_stage(
                batch_dir=batch_dir,
                stage="ranking",
                client=client,
                role_arn="arn:aws:iam::123:role/bedrock-batch",
                model_id="anthropic.claude-opus-4-6-v1",
                s3_input_prefix="s3://bucket/input",
                s3_output_uri="s3://bucket/output",
            )

            self.assertEqual(result["job_arn"], "arn:aws:bedrock:us-east-1:123:model-invocation-job/demo")
            self.assertEqual(fake_s3.uploads[0][1:], ("bucket", "input/batch-a/ranking/input.jsonl"))
            self.assertEqual(fake_bedrock.last_create["modelInvocationType"], "InvokeModel")
            self.assertEqual(fake_bedrock.last_create["inputDataConfig"]["s3InputDataConfig"]["s3Uri"], result["input_s3_uri"])

            manifest = read_jsonl(batch_dir / "manifest.jsonl")
            self.assertEqual(manifest[0]["status"], "submitted")
            self.assertEqual(manifest[0]["provider_job_arn"], result["job_arn"])

    def test_out_of_order_results_flow_through_human_review_and_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            labels = root / "labels"
            batch_dir = root / "teacher_batches" / "batch-a"
            responses = self._save_snapshots(snapshots, count=2)
            prepare_ranking_batch(batch_dir=batch_dir, snapshots_dir=snapshots)

            ranking_lines = [
                self._bedrock_line(
                    make_record_id(responses[1].response_id, "ranking"),
                    self._ranking_payload(responses[1]),
                ),
                self._bedrock_line(
                    make_record_id(responses[0].response_id, "ranking"),
                    self._ranking_payload(responses[0]),
                ),
            ]
            ranking_results = root / "ranking-results.jsonl"
            write_jsonl(ranking_results, ranking_lines)
            fetch_batch_stage(
                batch_dir=batch_dir,
                stage="ranking",
                client=BedrockBatchClient(region="us-east-1", bedrock_client=_FakeBedrockControl(), s3_client=_FakeS3()),
                results_path=ranking_results,
            )

            ranking_review = batch_dir / "ranking" / "review.jsonl"
            self.assertEqual(approve_review_file(ranking_review)["approved"], 2)
            prepare_rationale_batch(batch_dir=batch_dir, top_k_rationale=3)

            rationale_lines = []
            for response in reversed(responses):
                ranking = self._ranking_payload(response)
                rationale_lines.append(
                    self._bedrock_line(
                        make_record_id(response.response_id, "rationale"),
                        self._rationale_payload(response, ranking),
                    )
                )
            rationale_results = root / "rationale-results.jsonl"
            write_jsonl(rationale_results, rationale_lines)
            fetch_batch_stage(
                batch_dir=batch_dir,
                stage="rationale",
                client=BedrockBatchClient(region="us-east-1", bedrock_client=_FakeBedrockControl(), s3_client=_FakeS3()),
                results_path=rationale_results,
                top_k_rationale=3,
            )

            # Preview rows are pending after validation; finalize is approval-gated.
            self.assertEqual(finalize_labels(batch_dir=batch_dir, labels_dir=labels)["written"], 0)
            self.assertEqual(approve_review_file(batch_dir / "labels.preview.jsonl")["approved"], 2)
            result = finalize_labels(batch_dir=batch_dir, labels_dir=labels)
            self.assertEqual(result["written"], 2)
            for response in responses:
                self.assertTrue((labels / f"{response.response_id}.json").exists())

    def test_invalid_label_preview_cannot_be_auto_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            labels = root / "labels"
            batch_dir = root / "teacher_batches" / "batch-a"
            response = self._save_snapshots(snapshots)[0]
            prepare_ranking_batch(batch_dir=batch_dir, snapshots_dir=snapshots)

            ranking_results = root / "ranking-results.jsonl"
            ranking = self._ranking_payload(response)
            write_jsonl(
                ranking_results,
                [self._bedrock_line(make_record_id(response.response_id, "ranking"), ranking)],
            )
            fetch_batch_stage(
                batch_dir=batch_dir,
                stage="ranking",
                client=BedrockBatchClient(region="us-east-1", bedrock_client=_FakeBedrockControl(), s3_client=_FakeS3()),
                results_path=ranking_results,
            )
            approve_review_file(batch_dir / "ranking" / "review.jsonl")
            prepare_rationale_batch(batch_dir=batch_dir, top_k_rationale=1)

            bad_rationale = {
                "ranking": [],
                "rationales": {
                    ranking["ranking"][0]["book"]: {
                        "sentence_ids": ["missing"],
                        "reason": "bad",
                    }
                },
            }
            rationale_results = root / "rationale-results.jsonl"
            write_jsonl(
                rationale_results,
                [self._bedrock_line(make_record_id(response.response_id, "rationale"), bad_rationale)],
            )
            fetch_batch_stage(
                batch_dir=batch_dir,
                stage="rationale",
                client=BedrockBatchClient(region="us-east-1", bedrock_client=_FakeBedrockControl(), s3_client=_FakeS3()),
                results_path=rationale_results,
                top_k_rationale=1,
            )

            self.assertEqual(approve_review_file(batch_dir / "labels.preview.jsonl")["approved"], 0)
            result = finalize_labels(batch_dir=batch_dir, labels_dir=labels)
            self.assertEqual(result["written"], 0)
            self.assertEqual(result["skipped_review"], 1)


if __name__ == "__main__":
    unittest.main()
