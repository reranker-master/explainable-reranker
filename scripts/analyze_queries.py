#!/usr/bin/env python3
"""Quality analysis for the book-recommendation query pool.

Two layers:
  1. Quantitative (free, over EVERY query): length distribution vs the 8-30 char
     target, exact duplicates, near-duplicates (char-3gram Jaccard via an inverted
     index so it stays well under O(n^2)), FOCUS-category coverage, and non-Korean
     / degenerate flags.
  2. Qualitative (DeepSeek, on a stratified sample): each sampled query is rated for
     realism / specificity / natural-Korean and any issue is flagged, batched to
     keep the call count tiny.

  PYTHONPATH=src python3 scripts/analyze_queries.py \
      --sources existing=data/query_sets/existing_1000.txt new=data/query_sets/reranker_pilot_1000_v2.txt \
      --out .tmp/query_quality_report.md --sample 120

Pass --no-llm to skip the DeepSeek layer (pure stats, zero cost).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Loading / normalization
# --------------------------------------------------------------------------- #
HANGUL = re.compile(r"[가-힣]")


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip()).strip("·-•\"'　 ").lower()


def load_sources(specs: list[str]) -> list[tuple[str, str]]:
    """Return [(source_tag, query), ...] from 'tag=path' specs."""
    rows: list[tuple[str, str]] = []
    for spec in specs:
        tag, _, path = spec.partition("=")
        if not path:
            tag, path = "queries", tag
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append((tag, line))
    return rows


# --------------------------------------------------------------------------- #
# Near-duplicate detection (char 3-gram Jaccard, inverted-index blocked)
# --------------------------------------------------------------------------- #
def shingles(s: str, k: int = 3) -> set[str]:
    s = re.sub(r"\s+", "", _norm(s))
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def near_duplicate_pairs(queries: list[str], threshold: float = 0.6) -> list[tuple[int, int, float]]:
    sh = [shingles(q) for q in queries]
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)
    # candidate pairs: share at least one shingle (drop ultra-common shingles to keep it cheap)
    candidates: set[tuple[int, int]] = set()
    for ids in index.values():
        if len(ids) > 200:  # too common to be discriminative
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                candidates.add((ids[a], ids[b]))
    pairs: list[tuple[int, int, float]] = []
    for i, j in candidates:
        inter = len(sh[i] & sh[j])
        if not inter:
            continue
        jac = inter / len(sh[i] | sh[j])
        if jac >= threshold:
            pairs.append((i, j, round(jac, 3)))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# FOCUS-category coverage (keyword-based, multi-label)
# --------------------------------------------------------------------------- #
CATEGORIES: dict[str, list[str]] = {
    "장르": ["추리", "미스터리", "스릴러", "sf", "공상과학", "판타지", "로맨스", "공포", "호러",
            "역사소설", "무협", "라이트노벨", "라노벨", "느와르", "디스토피아"],
    "무드/감정": ["위로", "무기력", "번아웃", "설렘", "긴장", "먹먹", "통쾌", "힐링", "우울",
               "감동", "따뜻", "잔잔", "씁쓸", "슬픈", "웃긴", "유쾌"],
    "삶의상황": ["이직", "육아", "이별", "취업", "은퇴", "병문안", "이사", "첫 직장", "퇴사",
              "결혼", "임신", "수험", "방학", "군대", "노후"],
    "주제/소재": ["노동", "기후", "인공지능", "ai", "가족", "정체성", "예술", "과학사", "철학",
               "역사", "경제", "심리", "종교", "여성", "전쟁", "환경", "죽음", "사랑"],
    "비교작": ["처럼", "같은 느낌", "같은", "비슷한", "스타일", "작가", "수상", "베스트셀러", "원작"],
    "독서맥락/형식": ["출퇴근", "잠들기", "완독", "짧게", "짧은", "문장이", "오디오북", "두꺼운",
                 "가볍게", "한 권", "단편", "장편", "쉽게 읽", "술술"],
    "선물/대상": ["청소년", "10대", "부모님", "연인", "친구에게", "선물", "아이", "어린이",
               "남자친구", "여자친구", "초등", "중학생", "고등학생"],
    "비소설/교양": ["에세이", "교양", "입문", "인문", "자기계발", "경제경영", "역사책", "과학책",
                "심리학", "철학책", "비문학", "교양서"],
}


def categorize(q: str) -> list[str]:
    low = q.lower()
    hits = [cat for cat, kws in CATEGORIES.items() if any(k in low for k in kws)]
    return hits


# --------------------------------------------------------------------------- #
# Quantitative report
# --------------------------------------------------------------------------- #
def quant_report(rows: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    queries = [q for _, q in rows]
    n = len(queries)
    sources = sorted({t for t, _ in rows})

    # exact duplicates (normalized)
    norm_counts = Counter(_norm(q) for q in queries)
    exact_dups = {k: c for k, c in norm_counts.items() if c > 1}
    dup_extra = sum(c - 1 for c in exact_dups.values())

    # cross-source overlap
    by_src: dict[str, set[str]] = defaultdict(set)
    for t, q in rows:
        by_src[t].add(_norm(q))
    cross = []
    for a in range(len(sources)):
        for b in range(a + 1, len(sources)):
            inter = by_src[sources[a]] & by_src[sources[b]]
            cross.append((sources[a], sources[b], len(inter)))

    # length
    lengths = [len(q) for q in queries]
    in_target = sum(1 for L in lengths if 8 <= L <= 30)
    too_short = sum(1 for L in lengths if L < 8)
    too_long = sum(1 for L in lengths if L > 30)

    # language / degenerate
    no_hangul = [q for q in queries if not HANGUL.search(q)]
    very_generic = [q for q in queries if len(re.sub(r"\s+", "", q)) <= 5]

    # category coverage
    cat_counter: Counter = Counter()
    uncategorized = 0
    for q in queries:
        cats = categorize(q)
        if cats:
            cat_counter.update(cats)
        else:
            uncategorized += 1

    out.append("# Query pool quality report\n")
    out.append(f"- Total queries: **{n}** (sources: " + ", ".join(f"{s}={len(by_src[s])}" for s in sources) + ")\n")

    out.append("## Duplicates\n")
    out.append(f"- Exact (normalized) duplicate rows: **{dup_extra}** "
               f"({len(exact_dups)} distinct strings repeated)\n")
    for a, b, c in cross:
        out.append(f"- Cross-source overlap `{a}`∩`{b}`: **{c}**\n")

    out.append("## Length (chars)\n")
    out.append(f"- min {min(lengths)} / median {int(statistics.median(lengths))} / "
               f"mean {statistics.mean(lengths):.1f} / max {max(lengths)}\n")
    out.append(f"- in 8–30 target: **{in_target} ({100*in_target/n:.1f}%)**; "
               f"too short (<8): {too_short} ({100*too_short/n:.1f}%); "
               f"too long (>30): {too_long} ({100*too_long/n:.1f}%)\n")

    out.append("## Degenerate / language flags\n")
    out.append(f"- No Hangul at all: **{len(no_hangul)}**"
               + (f" — e.g. {no_hangul[:5]}" if no_hangul else "") + "\n")
    out.append(f"- Very generic (≤5 non-space chars): **{len(very_generic)}**"
               + (f" — e.g. {very_generic[:5]}" if very_generic else "") + "\n")

    out.append("## FOCUS-category coverage (multi-label, keyword-based)\n")
    out.append("| category | queries | % of pool |")
    out.append("|---|---|---|")
    for cat in CATEGORIES:
        c = cat_counter.get(cat, 0)
        out.append(f"| {cat} | {c} | {100*c/n:.1f}% |")
    out.append(f"| (uncategorized) | {uncategorized} | {100*uncategorized/n:.1f}% |")
    out.append("")

    # near-dups
    pairs = near_duplicate_pairs(queries, threshold=0.6)
    out.append("## Near-duplicates (char-3gram Jaccard ≥ 0.6)\n")
    out.append(f"- Near-duplicate pairs: **{len(pairs)}**\n")
    for i, j, jac in pairs[:15]:
        out.append(f"  - {jac}: `{queries[i]}`  ⟷  `{queries[j]}`")
    out.append("")
    return out


# --------------------------------------------------------------------------- #
# Qualitative (DeepSeek) — stratified sample, batched
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = "너는 한국 도서 추천 검색 쿼리의 품질을 평가하는 심사자다. JSON만 출력."
JUDGE_TMPL = (
    "아래는 도서 추천 서비스의 검색 쿼리 목록이다. 각 쿼리를 평가하라.\n"
    "평가 기준:\n"
    "- realistic: 실제 사용자가 입력할 법한가 (1~5)\n"
    "- specific: 책을 좁히기에 충분히 구체적인가 (1~5)\n"
    "- natural_korean: 자연스러운 한국어인가 (true/false)\n"
    "- issue: 문제가 있으면 한 줄로, 없으면 null\n"
    "출력은 다음 형식의 JSON 배열만 (설명 금지):\n"
    '[{"i":1,"realistic":4,"specific":3,"natural_korean":true,"issue":null}, ...]\n\n'
    "쿼리:\n{numbered}"
)


def _parse_json_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def llm_report(rows: list[tuple[str, str]], sample: int, batch: int, model_id: str) -> list[str]:
    import random

    from explainable_reranker.config.env import load_project_dotenv
    from explainable_reranker.teacher.llm_client import BedrockConverseChatModel

    load_project_dotenv()
    rng = random.Random(13)
    # stratified sample across sources
    by_src: dict[str, list[str]] = defaultdict(list)
    for t, q in rows:
        by_src[t].append(q)
    per_src = max(1, sample // max(1, len(by_src)))
    sampled: list[tuple[str, str]] = []
    for t, qs in by_src.items():
        pick = rng.sample(qs, min(per_src, len(qs)))
        sampled.extend((t, q) for q in pick)
    rng.shuffle(sampled)

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    model = BedrockConverseChatModel(model_id=model_id, region=region, max_tokens=8192, temperature=0.0)

    scores: list[dict] = []
    issues: list[tuple[str, str]] = []  # (query, issue)
    for start in range(0, len(sampled), batch):
        chunk = sampled[start : start + batch]
        numbered = "\n".join(f"{idx+1}. {q}" for idx, (_, q) in enumerate(chunk))
        try:
            text = model.generate(system=JUDGE_SYSTEM, user=JUDGE_TMPL.replace("{numbered}", numbered))
        except Exception as exc:  # noqa: BLE001
            print(f"  judge batch {start//batch} failed: {type(exc).__name__}: {exc}")
            continue
        for rec in _parse_json_array(text):
            i = rec.get("i")
            if not isinstance(i, int) or not (1 <= i <= len(chunk)):
                continue
            src, q = chunk[i - 1]
            rec["_src"] = src
            scores.append(rec)
            if rec.get("issue"):
                issues.append((q, str(rec["issue"])))
        print(f"  judged batch {start//batch+1}: cumulative {len(scores)} rated")

    out: list[str] = ["## Qualitative judging (DeepSeek sample)\n"]
    if not scores:
        out.append("_No ratings returned._\n")
        return out

    def avg(key: str, src: str | None = None) -> float:
        vals = [r[key] for r in scores if isinstance(r.get(key), (int, float))
                and (src is None or r.get("_src") == src)]
        return statistics.mean(vals) if vals else 0.0

    srcs = sorted({r["_src"] for r in scores})
    out.append(f"- Sampled & rated: **{len(scores)}** queries\n")
    out.append("| source | realistic | specific | %natural_korean | %with issue |")
    out.append("|---|---|---|---|---|")
    for s in srcs + ["(all)"]:
        sub = [r for r in scores if s == "(all)" or r.get("_src") == s]
        nat = 100 * sum(1 for r in sub if r.get("natural_korean") is True) / max(1, len(sub))
        iss = 100 * sum(1 for r in sub if r.get("issue")) / max(1, len(sub))
        sk = None if s == "(all)" else s
        out.append(f"| {s} | {avg('realistic', sk):.2f} | {avg('specific', sk):.2f} | {nat:.0f}% | {iss:.0f}% |")
    out.append("")
    if issues:
        out.append(f"### Flagged queries ({len(issues)})\n")
        for q, iss in issues[:25]:
            out.append(f"- `{q}` — {iss}")
        out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="tag=path specs")
    ap.add_argument("--out", type=Path, default=Path(".tmp/query_quality_report.md"))
    ap.add_argument("--sample", type=int, default=120, help="total queries to LLM-judge (stratified)")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--model-id", default=os.environ.get("BEDROCK_MODEL_ID", "deepseek.v3.2"))
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    rows = load_sources(args.sources)
    report = quant_report(rows)
    if not args.no_llm:
        report += llm_report(rows, args.sample, args.batch, args.model_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
