#!/usr/bin/env python3
"""Build a leakage-free train/valid/test split (plan §4 라벨 분리 원칙).

Items are split by query family AND book cluster: items sharing either never
straddle splits, so the eval set stays genuinely independent. The split is
deterministic (hash-based) and reproducible given the same --salt.

--items JSON: a list of objects
  [{"item_id": "q1", "family": "mood", "book_clusters": ["c1", "c2"]}, ...]
(family and book_clusters are optional; an item with neither is its own group.)

Example:
  PYTHONPATH=src python3 scripts/make_splits.py \
      --items data/eval/items.json --ratios 0.8,0.1,0.1 --salt v1 --out data/eval/splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from explainable_reranker.data.splits import SplitItem, split_by_family_and_cluster


def _parse_ratios(text: str) -> tuple[float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must be 'train,valid,test', e.g. 0.8,0.1,0.1")
    return parts[0], parts[1], parts[2]


def _load_items(path: Path) -> list[SplitItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[SplitItem] = []
    for entry in raw:
        items.append(
            SplitItem(
                item_id=str(entry["item_id"]),
                family=(str(entry["family"]) if entry.get("family") is not None else None),
                book_clusters=frozenset(str(c) for c in entry.get("book_clusters", [])),
            )
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Family/cluster-aware train/valid/test split.")
    parser.add_argument("--items", required=True, type=Path, help="items JSON (see module docstring)")
    parser.add_argument("--ratios", type=_parse_ratios, default=(0.8, 0.1, 0.1))
    parser.add_argument("--salt", default="", help="changes the assignment deterministically")
    parser.add_argument("--out", type=Path, default=None, help="optional path to write splits JSON")
    args = parser.parse_args()

    items = _load_items(args.items)
    assignment = split_by_family_and_cluster(items, ratios=args.ratios, salt=args.salt)
    result = {"train": list(assignment.train), "valid": list(assignment.valid), "test": list(assignment.test)}
    print(
        f"split {len(items)} items → train={len(assignment.train)} "
        f"valid={len(assignment.valid)} test={len(assignment.test)}"
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
