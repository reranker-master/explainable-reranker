from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import cycle


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    text: str
    family: str
    facets: tuple[str, ...]
    source: str = "synthetic"


FAMILY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "mood": (
        "{mood} 분위기의 {genre} 책 추천해줘",
        "{mood} 정서가 오래 남는 {genre}를 찾고 있어",
    ),
    "relationship": (
        "{relationship} 관계가 중심이고 {mood} 느낌인 책",
        "{relationship} 사이의 갈등과 회복이 잘 드러나는 작품",
    ),
    "trope": (
        "{trope} 설정이 있지만 {avoid} 느낌은 피하고 싶어",
        "{trope} 전개에 {mood} 감정선이 있는 책",
    ),
    "negative_preference": (
        "{avoid} 분위기는 싫고 {mood} 쪽으로 추천해줘",
        "{genre} 중에서 {avoid} 요소가 적고 {relationship} 이야기가 있는 책",
    ),
    "composite": (
        "{genre}이면서 {mood}, {trope}, {relationship} 요소가 같이 있는 책",
        "{mood} 무드에 {trope}가 들어가고 {avoid}는 약한 작품",
    ),
}

DEFAULT_FACETS = {
    "mood": ("잔잔한", "위로되는", "서늘한", "몽환적인", "긴장감 있는", "따뜻한"),
    "genre": ("소설", "에세이", "미스터리", "청소년 문학", "로맨스", "판타지"),
    "relationship": ("가족", "친구", "연인", "스승과 제자", "이웃", "자매"),
    "trope": ("성장", "재회", "비밀", "상실 이후의 회복", "작은 마을", "시간 여행"),
    "avoid": ("폭력적인", "지나치게 어두운", "급하게 끝나는", "설명이 많은", "자극적인"),
}


def generate_synthetic_queries(count: int, *, seed: int = 7) -> list[QuerySpec]:
    """Generate deterministic synthetic query families for teacher-label pilots."""

    rng = random.Random(seed)
    families = list(FAMILY_TEMPLATES)
    family_iter = cycle(families)
    queries: list[QuerySpec] = []
    seen: set[str] = set()

    while len(queries) < count:
        family = next(family_iter)
        template = rng.choice(FAMILY_TEMPLATES[family])
        values = {facet: rng.choice(options) for facet, options in DEFAULT_FACETS.items()}
        text = template.format(**values)
        if text in seen:
            continue
        seen.add(text)
        facets = tuple(f"{key}:{value}" for key, value in sorted(values.items()) if "{" + key + "}" in template)
        queries.append(
            QuerySpec(
                query_id=f"syn_{len(queries) + 1:05d}",
                text=text,
                family=family,
                facets=facets,
            )
        )

    return queries
