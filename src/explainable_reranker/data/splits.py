from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

# plan §4/§1.5 "라벨 분리 원칙": train/valid/test는 무작위가 아니라 query family·book
# cluster 단위로 나눠 누수를 막아야 한다. 같은 family(동일 템플릿/축에서 파생된 질의)나
# 같은 book cluster(세트책/개정판 등으로 묶인 책)를 공유하는 항목이 split 경계를 넘으면
# 사실상 같은 정보가 train과 test에 모두 들어가 평가가 부풀려진다.
#
# 두 차원을 동시에 만족시키기 위해 "family 또는 cluster를 공유하면 연결"로 보는 union-find
# 연결요소를 만들고, 각 연결요소 전체를 하나의 split에 결정론적으로 배정한다(해시 기반 →
# 재현 가능, 무작위 없음).


@dataclass(frozen=True)
class SplitItem:
    item_id: str
    family: str | None = None
    book_clusters: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SplitAssignment:
    train: tuple[str, ...]
    valid: tuple[str, ...]
    test: tuple[str, ...]

    def split_of(self, item_id: str) -> str | None:
        for name in ("train", "valid", "test"):
            if item_id in getattr(self, name):
                return name
        return None


def split_by_family_and_cluster(
    items: list[SplitItem],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "",
) -> SplitAssignment:
    """Group-aware deterministic train/valid/test split (no leakage across groups).

    Items that share a family or any book cluster are unioned into one component;
    each component lands wholly in a single split, chosen by hashing the component's
    stable key so the assignment is reproducible across runs.
    """

    train_ratio, valid_ratio = _normalized_thresholds(ratios)

    parent: dict[str, str] = {item.item_id: item.item_id for item in items}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # attach to the lexicographically smaller root for deterministic structure
            low, high = sorted((left_root, right_root))
            parent[high] = low

    family_rep: dict[str, str] = {}
    cluster_rep: dict[str, str] = {}
    for item in items:
        if item.family is not None:
            existing = family_rep.setdefault(item.family, item.item_id)
            union(item.item_id, existing)
        for cluster in item.book_clusters:
            existing = cluster_rep.setdefault(cluster, item.item_id)
            union(item.item_id, existing)

    components: dict[str, list[str]] = defaultdict(list)
    for item in items:
        components[find(item.item_id)].append(item.item_id)

    buckets: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for member_ids in components.values():
        component_key = min(member_ids)  # stable across runs regardless of input order
        bucket = _bucket(component_key, train_ratio, valid_ratio, salt=salt)
        buckets[bucket].extend(member_ids)

    return SplitAssignment(
        train=tuple(sorted(buckets["train"])),
        valid=tuple(sorted(buckets["valid"])),
        test=tuple(sorted(buckets["test"])),
    )


def _normalized_thresholds(ratios: tuple[float, float, float]) -> tuple[float, float]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must be three non-negative numbers (train, valid, test)")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("ratios must sum to a positive number")
    train_ratio = ratios[0] / total
    valid_ratio = ratios[1] / total
    return train_ratio, train_ratio + valid_ratio


def _bucket(key: str, train_ratio: float, valid_cumulative: float, *, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / float(0x100000000)  # deterministic value in [0, 1)
    if fraction < train_ratio:
        return "train"
    if fraction < valid_cumulative:
        return "valid"
    return "test"
