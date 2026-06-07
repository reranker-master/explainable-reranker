#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from explainable_reranker.serve.api import rerank_payload


def main() -> int:
    fixture = Path("tests/fixtures/topa_page_response.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    response = rerank_payload(payload)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
