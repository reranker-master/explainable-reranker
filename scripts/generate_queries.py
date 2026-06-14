#!/usr/bin/env python3
"""Generate N additional, de-duplicated Korean book-recommendation queries.

Matches the style/diversity of data/query_sets/reranker_pilot_500.txt (genre,
mood, situation, theme, comparison-to-a-known-book, gift/reading context) using
the same Bedrock teacher backend (DeepSeek via Converse). Output is de-duplicated
against the existing set and within itself, then written one-per-line.

  PYTHONPATH=src python3 scripts/generate_queries.py \
      --existing data/query_sets/reranker_pilot_500.txt \
      --out data/query_sets/reranker_pilot_500_extra.txt --target 500
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from explainable_reranker.config.env import load_project_dotenv
from explainable_reranker.teacher.llm_client import BedrockConverseChatModel

# Rotating focus hints so successive calls cover different parts of the space and
# overlap less (dedup still catches the rest).
FOCUS = [
    "장르 중심: 추리/미스터리, SF, 판타지, 로맨스, 스릴러, 공포, 역사소설, 무협, 라이트노벨의 세부 취향",
    "무드/감정 상태: 위로, 무기력, 번아웃, 설렘, 긴장, 먹먹함, 통쾌함 등에 맞는 책",
    "삶의 상황/맥락: 이직, 육아, 이별, 취업준비, 은퇴, 병문안, 이사, 첫 직장",
    "주제/소재: 노동, 기후위기, 인공지능, 가족, 정체성, 예술, 과학사, 철학적 질문",
    "비교작 기준: 유명한 책/작가/수상작 '~처럼', '~같은 느낌'의 추천",
    "독서 맥락/형식: 출퇴근에 짧게, 잠들기 전, 완독 쉬운, 문장이 아름다운, 오디오북으로 좋은",
    "선물/대상: 10대 청소년, 부모님, 연인, 책 안 읽는 친구에게 줄 책",
    "비소설/교양: 에세이, 과학교양, 역사교양, 경제, 심리학 입문, 인문 에세이",
]

PROMPT_TMPL = (
    "너는 한국 도서 추천 서비스의 검색 쿼리를 만드는 도우미다.\n"
    "사용자가 책을 찾을 때 실제로 입력할 법한 자연스러운 한국어 추천 쿼리를 {n}개 생성하라.\n"
    "초점: {focus}\n"
    "규칙:\n"
    "- 한 줄에 하나, 번호/불릿/따옴표 없이 쿼리 문장만.\n"
    "- 8~30자 내외의 구체적이고 다양한 표현. 뻔한 반복 금지.\n"
    "- 아래 '이미 있는 예시'와 의미가 겹치는 쿼리는 만들지 말 것.\n"
    "이미 있는 예시(피하라):\n{avoid}\n"
)


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip()).strip("·-•\"'　 ").lower()


def _parse(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip("\"'")
        if 4 <= len(line) <= 60 and not line.endswith(":"):
            out.append(line)
    return out


def main() -> int:
    load_project_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--model-id", default=os.environ.get("BEDROCK_MODEL_ID", "deepseek.v3.2"))
    ap.add_argument("--per-call", type=int, default=120)
    args = ap.parse_args()

    existing = [l.strip() for l in args.existing.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {_norm(q) for q in existing}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    model = BedrockConverseChatModel(model_id=args.model_id, region=region, max_tokens=8192)

    import random  # deterministic avoid-sample without affecting reproducibility elsewhere
    rng = random.Random(7)
    collected: list[str] = []
    i = 0
    while len(collected) < args.target and i < 60:
        focus = FOCUS[i % len(FOCUS)]
        avoid = "\n".join(rng.sample(existing, min(40, len(existing))))
        prompt = PROMPT_TMPL.format(n=args.per_call, focus=focus, avoid=avoid)
        try:
            text = model.generate(system="한국어로만 답하라. 쿼리 목록만 출력.", user=prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"  call {i} failed: {type(exc).__name__}: {exc}")
            i += 1
            continue
        added = 0
        for q in _parse(text):
            k = _norm(q)
            if k and k not in seen:
                seen.add(k)
                collected.append(q)
                added += 1
        i += 1
        print(f"  call {i}: +{added} new (total {len(collected)}/{args.target})")

    collected = collected[: args.target]
    args.out.write_text("\n".join(collected) + "\n", encoding="utf-8")
    print(f"wrote {len(collected)} unique new queries -> {args.out}")
    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
