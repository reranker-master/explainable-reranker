from __future__ import annotations

from collections import defaultdict

from explainable_reranker.data.sentence_index import IndexedSentence
from explainable_reranker.topa.adapter import TopaPageResponse


SYSTEM_INSTRUCTIONS = """You are a grounded ranking teacher for Korean book recommendation.
Use only the provided sentence IDs as evidence. Do not invent sentence IDs or quote unsupported facts.
The candidate pool may contain hard negatives: books that look plausible but are wrong for the query
(same genre but the opposite mood, or box-set/revised-edition duplicates of another book). Judge each
book on its evidence sentences, not on surface genre or title similarity, and score such distractors low.
Return strict JSON only."""


def build_listwise_prompt(
    response: TopaPageResponse,
    sentence_index: list[IndexedSentence],
    *,
    max_sentences_per_book: int = 16,
) -> str:
    evidence_by_book = _evidence_by_book(sentence_index)
    blocks = [
        SYSTEM_INSTRUCTIONS,
        "",
        "Task A: rank candidate books by relevance to the query.",
        "Return JSON with ranking sorted by descending score in the 0..3 range.",
        "Some candidates are hard negatives: plausible but wrong for this query — same genre but the",
        "opposite mood/intent, or a duplicate edition (box-set/revised) of another candidate. Score",
        "those low (0..0.5) rather than be fooled by surface similarity, AND list each one under",
        '"hard_negatives" keyed by book_id with a reason. Use reason "same_genre_diff_mood" (looks',
        'relevant by genre/title but the mood or intent is wrong), "title_variant" (duplicate edition',
        'of another candidate), or "other". Only flag books that look relevant but are traps — do not',
        "flag books that are simply unrelated. Leave hard_negatives empty if the pool has no such traps.",
        "",
        f"[QUERY] {response.query}",
    ]
    for candidate in response.candidates:
        blocks.append(f"[BOOK {candidate.book_id}] title: {candidate.title}")
        for sentence in evidence_by_book[candidate.book_id][:max_sentences_per_book]:
            blocks.append(f"  {sentence.sentence_id}) {sentence.text}")
    blocks.append(
        'Schema: {"ranking":[{"book":"book_id","score":0.0}],'
        '"hard_negatives":{"book_id":{"reason":"same_genre_diff_mood","note":"one short reason"}},'
        '"rationales":{}}'
    )
    return "\n".join(blocks)


def build_rationale_prompt(
    response: TopaPageResponse,
    sentence_index: list[IndexedSentence],
    *,
    ranked_book_ids: list[str],
    top_k: int = 10,
    max_sentences_per_book: int = 16,
) -> str:
    evidence_by_book = _evidence_by_book(sentence_index)
    candidate_by_id = {candidate.book_id: candidate for candidate in response.candidates}
    blocks = [
        SYSTEM_INSTRUCTIONS,
        "",
        "Task B: choose grounded rationale sentence IDs for each top-ranked book.",
        "Select 1 to 3 sentence IDs per book. Use only IDs shown below.",
        "",
        f"[QUERY] {response.query}",
    ]
    for book_id in ranked_book_ids[:top_k]:
        candidate = candidate_by_id[book_id]
        blocks.append(f"[BOOK {candidate.book_id}] title: {candidate.title}")
        for sentence in evidence_by_book[book_id][:max_sentences_per_book]:
            blocks.append(f"  {sentence.sentence_id}) {sentence.text}")
    blocks.append(
        'Schema: {"ranking":[],"rationales":{"book_id":{"sentence_ids":["..."],"reason":"one grounded sentence"}}}'
    )
    return "\n".join(blocks)


def _evidence_by_book(sentence_index: list[IndexedSentence]) -> dict[str, list[IndexedSentence]]:
    grouped: dict[str, list[IndexedSentence]] = defaultdict(list)
    for sentence in sentence_index:
        grouped[sentence.book_id].append(sentence)
    return grouped
