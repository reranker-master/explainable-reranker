from __future__ import annotations

from explainable_reranker.models.select_predict.model import Span


def build_reason(query: str, spans: tuple[Span, ...] | list[Span], *, max_sentences: int = 2) -> str:
    """Render an extractive explanation from selected spans only."""

    selected = [span.text.strip() for span in spans if span.text.strip()][:max_sentences]
    if not selected:
        return "선택된 근거 문장이 없어 추천 사유를 생성하지 않았습니다."
    if len(selected) == 1:
        return f"'{query}'에 대한 근거는 \"{selected[0]}\" 문장입니다."
    evidence = " / ".join(f"\"{text}\"" for text in selected)
    return f"'{query}'에 대한 근거는 {evidence} 문장들입니다."
