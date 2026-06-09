from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explainable_reranker.data.sentence_index import build_sentence_index
from explainable_reranker.topa.adapter import TopaPageResponse, parse_topa_page_response


@dataclass(frozen=True)
class SnapshotRecord:
    """Metadata for an immutable topa.page raw snapshot."""

    response_id: str
    query_id: str
    query: str
    topa_pipeline_version: str
    schema_version: str
    retrieval_params: dict[str, Any]
    request_timestamp: str
    payload_sha256: str
    path: str


class SnapshotStore:
    """Stores topa.page raw JSON snapshots under content-addressed metadata.

    A response_id is immutable. Re-saving the same payload is idempotent; saving a
    different payload under the same response_id raises an error so teacher labels
    cannot silently drift.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(
        self,
        payload: dict[str, Any],
        *,
        request_timestamp: str | None = None,
    ) -> SnapshotRecord:
        response = parse_topa_page_response(payload)
        payload_hash = canonical_json_sha256(payload)
        # plan §2.7.1/§3: persist the per-sentence text_hash alongside the raw payload
        # so reproducibility is anchored on the exact evidence text the labels cite,
        # not just the whole-blob hash. sentence_index assigns IDs/offsets only.
        sentence_hashes = {
            sentence.sentence_id: sentence.text_hash
            for sentence in build_sentence_index(response)
        }
        timestamp = request_timestamp or datetime.now(UTC).isoformat()
        snapshot_dir = self.root / response.schema_version
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{response.response_id}.json"

        envelope = {
            "metadata": {
                "response_id": response.response_id,
                "query_id": response.query_id,
                "query": response.query,
                "topa_pipeline_version": response.topa_pipeline_version,
                "schema_version": response.schema_version,
                "retrieval_params": response.retrieval_params,
                "request_timestamp": timestamp,
                "payload_sha256": payload_hash,
            },
            "sentence_hashes": sentence_hashes,
            "payload": payload,
        }

        if snapshot_path.exists():
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
            existing_hash = existing.get("metadata", {}).get("payload_sha256")
            if existing_hash != payload_hash:
                raise ValueError(
                    f"snapshot {response.response_id} already exists with different payload hash"
                )
        else:
            snapshot_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return SnapshotRecord(
            response_id=response.response_id,
            query_id=response.query_id,
            query=response.query,
            topa_pipeline_version=response.topa_pipeline_version,
            schema_version=response.schema_version,
            retrieval_params=response.retrieval_params,
            request_timestamp=timestamp,
            payload_sha256=payload_hash,
            path=str(snapshot_path),
        )

    def load(self, schema_version: str, response_id: str) -> TopaPageResponse:
        snapshot_path = self.root / schema_version / f"{response_id}.json"
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return parse_topa_page_response(envelope["payload"])

    def load_record(self, schema_version: str, response_id: str) -> SnapshotRecord:
        snapshot_path = self.root / schema_version / f"{response_id}.json"
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
        metadata = envelope["metadata"]
        return SnapshotRecord(path=str(snapshot_path), **metadata)

    def load_sentence_hashes(self, schema_version: str, response_id: str) -> dict[str, str]:
        """Return the persisted ``{sentence_id: text_hash}`` map for a snapshot.

        Empty for snapshots written before sentence-hash persistence (back-compat).
        """

        snapshot_path = self.root / schema_version / f"{response_id}.json"
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return envelope.get("sentence_hashes", {})

    def write_manifest(self, records: list[SnapshotRecord], name: str = "manifest.jsonl") -> Path:
        manifest_path = self.root / name
        self.root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        return manifest_path


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
