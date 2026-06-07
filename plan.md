# Explainable Reranker — 학습(증류) 중심 실행 계획 (plan.md)

> **본질: 리랭커를 "학습"시키는 것.**
> 고성능 LLM(teacher)으로 랭킹 + 근거를 만들고, 그것을 **bge-reranker-v2-m3(student)** 에 **증류(distillation)** 한다.
> 학습된 리랭커는 단일 forward에서 **(1) 책을 제대로 랭킹**하고 **(2) 추천 사유에 근접한 토큰(rationale span)을 뿌려준다.**
> 토파(`topa_service`)는 후보·코퍼스·서빙·평가앵커를 제공하는 **도구로 재사용**한다(=새로 안 만든다). 학습 설계가 이 문서의 중심.

---

## 0. 핵심 아이디어 (왜 증류인가)

보고서가 지목한 최대 병목 = **"rationale 라벨이 달린 대규모 데이터 부재"**.
원래 계획의 해법(쿼리-문장 cosine weak label)은 약하다. → **LLM teacher로 ranking + rationale을 둘 다 생성해서 증류**하면 이 병목을 정면 돌파한다.

```
[고성능 LLM teacher]  query + 후보책(제목/시놉시스/리뷰문장에 번호 부여)
        │  프롬프트
        ▼
  ① 랭킹/관련도 점수 (listwise)      → 랭킹 증류 타깃
  ② 근거 문장 ID 인용 (grounded)     → 근거 증류 타깃 (환각 불가: 실제 문장만 가리킴)
        │  distillation
        ▼
[student = bge-reranker-v2-m3 + LoRA]  단일 forward로
  ① relevance score  ② 토큰별 rationale 확률(=근거 근접 토큰)
```

- **RankGPT급 품질을 cross-encoder 속도로**: teacher는 느리지만(5~30s) 학습 때만 쓰고, 추론은 student 단독(≤1s).
- **구조적 faithfulness**: teacher가 "자유 텍스트"가 아니라 **번호 매긴 실제 문장 ID**를 인용 → span이 항상 실제 코퍼스 토큰에 매핑됨 → 환각 원천 차단.

---

## 1. ⭐ Teacher 라벨 생성 (학습 데이터의 핵심)

### 1.1 Teacher 모델 선택
| 역할 | 후보 | 비고 |
|---|---|---|
| 메인 teacher (대량) | Gemini 3 Flash (토파에 이미 연결됨) / Qwen3-Reranker-8B | 비용·속도 균형, 대량 라벨링 |
| 강 teacher (캘리브레이션·검수) | **Claude Opus 4.8** (`claude-opus-4-8`, 토파 Bedrock 경유 가능) | 고난도 쿼리·근거 품질 기준, agreement 체크 |
| 기존 약라벨 (warm-up) | Qdrant cosine (쿼리↔청크) | teacher 호출 전 student 사전적응용 |

> 토파가 이미 Gemini(google-generativeai)·Bedrock Claude를 쓰고 있어 teacher 호출 인프라를 그대로 재사용. provider 불명 시 최신·고성능 Claude를 기본값으로.

### 1.2 입력 구성 (grounding이 핵심)
후보책의 모든 근거 문장에 **고유 ID를 부여**해서 teacher에 넣는다. teacher는 span 텍스트를 "쓰지" 않고 **ID만 고른다** → 항상 실제 토큰에 매핑.
```
[QUERY] 잔잔하고 위로되는 가족 이야기
[BOOK b1] 제목: ...
  s1) (시놉시스 문장)
  s2) (리뷰 문장)
  s3) (mood/trope 태그 문장)
  ...
[BOOK b2] ...
```

### 1.3 Teacher 프롬프트 (2-pass)
- **Pass A — listwise 랭킹**: 후보 N권을 query 관련도로 정렬 + graded score(0~3 또는 0~1). (RankGPT 슬라이딩 윈도우로 N>윈도우 처리)
- **Pass B — 근거 인용**: 상위 책마다 "왜 추천?"의 **근거 문장 ID들**과 한 줄 사유.

출력 스키마(예):
```jsonc
{
  "ranking": [{"book":"b1","score":0.93}, {"book":"b2","score":0.41}, ...],
  "rationales": {
    "b1": {"sentence_ids":["s2","s7"], "reason":"가족의 빈자리를 잔잔히 다룸"},
    ...
  }
}
```

### 1.4 비용/품질 통제
- 문장 ID 인용 → 출력 토큰 최소화. 근거는 top-k 책만.
- 메인=Gemini Flash 대량, 강=Opus 4.8 샘플(예: 5~10%)로 **agreement(κ)** 측정 → 라벨 신뢰도 게이트.
- 쿼리 dedup·캐시·배치. teacher 호출은 학습 1회성 비용.

**산출물:** `(query, candidate, teacher_score, rationale_sentence_ids)` 라벨셋.

---

## 2. ⭐ 증류 학습 설계 (student = bge-reranker + LoRA)

teacher 신호 → student 손실. 세 아키텍처가 **같은 teacher 라벨**을 소비한다.

### 2.1 손실 함수
**(a) 랭킹 증류** — teacher 점수 분포를 student에 이식
```
P_t = softmax(teacher_scores / τ)        # 쿼리별 후보 분포
P_s = softmax(student_scores / τ)
L_rank = KL(P_t || P_s)                   # listwise KD (RankDistil 계열)
```
- 보조: 실제 피드백 하드라벨로 앵커  `L_hard = BCE(student, feedback_label)` (있는 쿼리만)
- teacher가 순위만 주면 ListMLE / pairwise(RankNet)로 대체.

**(b) 근거 증류** — teacher 인용 문장 → 토큰 라벨 `y_t∈{0,1}` (인용 문장에 속한 토큰=1)
- **B(Multi-task)**: span head 토큰 확률 `p_t`. `L_span = weighted BCE(p_t, y_t)` (BIO 가능). → 이 `p_t`가 곧 "뿌려주는 근거 토큰".
- **C(Select-then-Predict)**: generator가 문장 선택 `z_t`(Gumbel-Softmax). teacher 문장으로 선택 지도 `L_select = BCE(z,y) + λ_sp·‖z‖₁ + λ_co·continuity`. predictor는 **선택된 토큰만으로** score → 거기에 `L_rank`. → 점수가 근거에만 의존 = 구조적 faithfulness.

**(c) 총 손실**
```
L = L_rank  +  α·L_hard  +  β·L_rationale  (+ reg)
```
α, β, τ는 스윕. 근거가 랭킹을 해치지 않는 trade-off 지점 탐색.

### 2.2 학습 절차
1. **Warm-up**: Qdrant cosine 약라벨로 LoRA 사전적응(teacher 호출 절약).
2. **Distill**: teacher 라벨로 본 학습 (`L_rank + L_rationale`).
3. **Refine(self-distill)**: student가 어려워하는 쿼리만 강 teacher(Opus)로 재라벨 → 2~3라운드.
4. LoRA target=q/k/v/o proj, rank·alpha·τ·α·β ablation, W&B 추적.

### 2.3 세 아키텍처 = 같은 증류, 다른 head
| 설계 | 구조 | 랭킹 학습 | 근거 학습 | faithfulness |
|---|---|---|---|---|
| A baseline | score only | L_rank | — | post-hoc만 |
| B multi-task | score + BIO span head | L_rank | L_span(토큰 BCE) | 중간 |
| C select-predict | generator→predictor | predictor L_rank | L_select(+sparsity/continuity) | 구조적 보장 |

**산출물:** A/B/C 체크포인트, ablation(τ/α/β·LoRA), 랭킹×근거×레이턴시 trade-off 표.

---

## 3. 토파 재사용 (학습을 위한 도구 — 새로 안 만듦)
학습에 필요한 재료만 빠르게 끌어온다. (상세 파일 경로는 부록)
- **후보 풀**: 검색 파이프라인(Qdrant+Memgraph→RRF) Top-50 = teacher 입력 후보. 로깅으로 수집.
- **근거 코퍼스(문장 단위)**: `topa_raw.book_chunks`(review/synopsis/mood) + Qdrant 3개 컬렉션 → 문장 ID 부여 대상.
- **하드라벨/평가 앵커**: `book_feedback`/`block_feedback`/`question_history` → `L_hard` + 오프라인 평가.
- **하드 네거티브**: Memgraph theme/mood/trope("동일 장르·다른 무드"), books 제목정규화(세트책/개정판) → teacher 입력에 섞어 난이도↑.
- **서빙(drop-in)**: 기존 `/rerank` HTTP 계약에 `spans` 필드만 추가해 그대로 꽂음(하위호환).
- **teacher 호출 인프라**: 토파의 Gemini/Bedrock 접속 재사용.

---

## 4. 평가
- **랭킹**: NDCG@{1,5,10}, MRR, Recall@K (피드백/teacher 합의 라벨 기준)
- **도서특화**: 장르/저자/출판사 다양성, 세트책·개정판 혼입률, 세렌디피티
- **근거 충실도(ERASER)**: sufficiency, comprehensiveness + teacher 인용 대비 token-F1/IoU
- **사람평가**: 팀원6+베타20, 블라인드 A/B, 근거 타당성 5점(≥4.0)
- **서비스 A/B**: 토파 2주, CTR(+3%)·체류·재방문
- **teacher vs student 격차**: 증류 효율(품질 보존율) 리포트

### KPI
| 품질 세트책혼입 50%↓ | 근거 4.0↑ | CTR +3%↑ | 레이턴시 ≤1s | 증류 품질보존 측정 |

---

## 5. 리포 구조 (학습 중심)
```
explainable-reranker/
├── plan.md  pyproject.toml  configs/
├── src/
│   ├── teacher/                 # ⭐ LLM teacher 라벨링
│   │   ├── prompts.py           # listwise rank + rationale-by-ID 프롬프트
│   │   ├── label_ranking.py     # Pass A
│   │   ├── label_rationale.py   # Pass B
│   │   └── agreement.py         # Gemini vs Opus κ
│   ├── distill/                 # ⭐ 증류 학습
│   │   ├── dataset.py           # teacher 라벨 → 학습 샘플(토큰 정렬)
│   │   ├── losses.py            # L_rank(KL/ListMLE)+L_hard+L_rationale
│   │   └── trainer.py           # LoRA, warm-up→distill→refine
│   ├── models/{baseline,multitask,select_predict}.py
│   ├── topa/{db,qdrant_client,memgraph_client,log_collector}.py  # 재사용 어댑터
│   ├── eval/{ir_metrics,book_metrics,faithfulness,run_eval}.py
│   └── serve/{api,export_onnx,quantize}.py   # /rerank + spans
├── benchmark/  scripts/  notebooks/  tests/
```

---

## 6. 타임라인 (학습에 집중, 16주)
| 주차 | 마일스톤 |
|---|---|
| W1 | 토파 연결(어댑터) + 후보/코퍼스/피드백 추출 + baseline(현 cross-encoder) 로깅 |
| W2–W3 | **Teacher 라벨링 파이프라인** (프롬프트·문장ID·랭킹+근거), agreement 검증, 라벨셋 v1 |
| W4–W6 | **증류 학습 B(multi-task)**: L_rank+L_span, warm-up→distill, span 품질 |
| W7–W9 | **증류 학습 C(select-predict)** + A/B/C ablation(τ/α/β·LoRA), refine 라운드 |
| W10–W11 | 평가·벤치마크 확정 + 사람평가(6+20) |
| W12–W13 | 서빙 drop-in(/rerank+spans), FP16/INT8·ONNX/TensorRT(≤1s), Two-tier 설명 |
| W14–W15 | 토파 실트래픽 A/B 2주 (CTR·체류·재방문) |
| W16 | HF 체크포인트·벤치마크 공개, 논문/발표 정리 |

---

## 7. 팀 역할 (6인)
| 인원 | 트랙 | 담당 |
|---|---|---|
| 2명 | teacher·데이터 | `src/teacher/*`+`src/topa/*` (라벨링·후보·코퍼스·하드네거티브·agreement) |
| 2명 | 증류 학습 | `src/distill/*`+`src/models/*` (A/B/C·손실·LoRA·ablation·refine) |
| 1명 | 평가·벤치마크 | `src/eval/*`+`benchmark/` (메트릭·사람평가·token-F1) |
| 1명 | 서빙 | `src/serve/*`+`explain/*` (/rerank+spans·최적화·A/B 운영) |

---

## 8. 리스크 & 대응
| 리스크 | 대응 |
|---|---|
| teacher 라벨 비용 | Flash 대량 + Opus 샘플, ID인용으로 출력↓, 캐시·dedup, 1회성 |
| teacher 랭킹 노이즈 | agreement κ 게이트, 하드피드백 앵커(L_hard), refine 라운드 |
| 근거 인용 부정확 | 문장ID 강제(자유텍스트 금지), 사람검수 샘플, IoU 모니터 |
| 랭킹↔근거 trade-off | β·τ 스윕, C안으로 구조적 분리 |
| C selection 불안정 | Gumbel temp 스케줄, B fallback |
| 레이턴시 미달 | INT8/distill, span head 경량화, Top-K 축소 |

---

## 9. 즉시 시작 작업
1. `src/topa/`: DB/Qdrant/Memgraph 접속 smoke test + 후보/청크/피드백 건수 실측
2. `src/teacher/prompts.py`: 문장ID 기반 listwise+rationale 프롬프트 1차 + 10쿼리 파일럿
3. teacher 출력 스키마 검증 + Gemini↔Opus agreement 측정
4. `src/distill/losses.py`: KL 랭킹증류 + 토큰 BCE 근거증류 골격, 작은 셋 overfit 테스트
5. baseline 로깅으로 평가 파이프라인(ir_metrics) 먼저 가동

---

### 부록 — 토파 파일 레퍼런스
| 용도 | 경로 |
|---|---|
| `/rerank` HTTP 계약(서빙 drop-in) | `topa/backend/app/modules/search/runtime/ranking.py:798` |
| cross-encoder rerank (baseline A) | `…/ranking.py:1086` |
| 세트책/개정판 필터 | `…/ranking.py:1273` |
| rerank 파이프라인 | `topa/backend/app/modules/search/pipeline/rerank.py` |
| Qdrant 컬렉션(COL_REVIEW/SYNOPSIS/MOOD) | `…/search/runtime/settings.py:199` |
| 근거 코퍼스/ETL | `topa-data-refine/`(book_chunks, workflows) |
| 텍스트 정제 | `topa-review-crawler/sanitizer.py` |
| DB/서비스 접속 | `topa-data-refine/.env.example` |

### 참고문헌
Lei+2016(Rationalizing), DeYoung+2020(ERASER), Sun+2023(RankGPT, teacher 근거), Xiao+2023(BGE), Hu+2022(LoRA), Zhuang+2023(RankT5, 랭킹손실)
