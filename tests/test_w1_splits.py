from __future__ import annotations

import unittest

from explainable_reranker.data.splits import (
    SplitItem,
    split_by_family_and_cluster,
)


class SplitByFamilyAndClusterTest(unittest.TestCase):
    def test_shared_family_stays_in_one_split(self) -> None:
        items = [SplitItem(item_id=f"q{i}", family="mood") for i in range(20)]
        assignment = split_by_family_and_cluster(items)
        splits = {assignment.split_of(item.item_id) for item in items}
        self.assertEqual(len(splits), 1)  # all 20 land together

    def test_shared_cluster_unions_transitively(self) -> None:
        # q1-q2 share book "B1"; q2-q3 share book "B2" → all three are one component.
        items = [
            SplitItem(item_id="q1", book_clusters=frozenset({"B1"})),
            SplitItem(item_id="q2", book_clusters=frozenset({"B1", "B2"})),
            SplitItem(item_id="q3", book_clusters=frozenset({"B2"})),
            SplitItem(item_id="q4", book_clusters=frozenset({"B9"})),
        ]
        assignment = split_by_family_and_cluster(items)
        self.assertEqual(
            assignment.split_of("q1"),
            assignment.split_of("q2"),
        )
        self.assertEqual(assignment.split_of("q2"), assignment.split_of("q3"))

    def test_no_group_spans_two_splits(self) -> None:
        # Many groups; assert the leakage invariant holds for every family and cluster.
        items = []
        for i in range(60):
            items.append(
                SplitItem(
                    item_id=f"q{i}",
                    family=f"fam{i % 12}",
                    book_clusters=frozenset({f"c{i % 7}"}),
                )
            )
        assignment = split_by_family_and_cluster(items)

        by_family: dict[str, set[str]] = {}
        by_cluster: dict[str, set[str]] = {}
        for item in items:
            split = assignment.split_of(item.item_id)
            by_family.setdefault(item.family, set()).add(split)
            for cluster in item.book_clusters:
                by_cluster.setdefault(cluster, set()).add(split)
        for splits in by_family.values():
            self.assertEqual(len(splits), 1)
        for splits in by_cluster.values():
            self.assertEqual(len(splits), 1)

    def test_deterministic_and_partitions_all_items(self) -> None:
        items = [SplitItem(item_id=f"q{i}", family=f"f{i}") for i in range(50)]
        first = split_by_family_and_cluster(items, salt="v1")
        second = split_by_family_and_cluster(list(reversed(items)), salt="v1")
        self.assertEqual(first, second)  # order-independent + reproducible

        all_ids = set(first.train) | set(first.valid) | set(first.test)
        self.assertEqual(all_ids, {item.item_id for item in items})
        # disjoint
        self.assertEqual(len(all_ids), len(first.train) + len(first.valid) + len(first.test))

    def test_salt_changes_assignment(self) -> None:
        items = [SplitItem(item_id=f"q{i}", family=f"f{i}") for i in range(50)]
        a = split_by_family_and_cluster(items, salt="a")
        b = split_by_family_and_cluster(items, salt="b")
        self.assertNotEqual(a, b)

    def test_invalid_ratios_raise(self) -> None:
        items = [SplitItem(item_id="q1")]
        with self.assertRaises(ValueError):
            split_by_family_and_cluster(items, ratios=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            split_by_family_and_cluster(items, ratios=(0.8, -0.1, 0.3))


if __name__ == "__main__":
    unittest.main()
