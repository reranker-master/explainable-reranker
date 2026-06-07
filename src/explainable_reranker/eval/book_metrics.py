from __future__ import annotations


def categorical_diversity(values: list[str]) -> float:
    if not values:
        return 0.0
    return len(set(values)) / len(values)


def duplicate_family_rate(book_family_ids: list[str]) -> float:
    if not book_family_ids:
        return 0.0
    duplicate_count = len(book_family_ids) - len(set(book_family_ids))
    return duplicate_count / len(book_family_ids)
