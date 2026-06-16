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
[student = bge-reranker-v2-m3 + LoRA, select-then-predict]
  Generator: 근거 문장 선택(z)  →  Predictor: 선택된 근거만으로 ① relevance score
  → ② 선택된 문장/토큰 = 근거(span). 점수가 근거에만 의존 = 구조적 faithfulness.
```

- **RankGPT급 품질을 cross-encoder 속도로**: teacher는 느리지만(5~30s) 학습 때만 쓰고, 추론은 student 단독(≤1s).
- **구조적 faithfulness**: teacher가 "자유 텍스트"가 아니라 **번호 매긴 실제 문장 ID**를 인용 → span이 항상 실제 코퍼스 토큰에 매핑됨 → 환각 원천 차단.

---

## 1. ⭐ Teacher 라벨 생성 (학습 데이터의 핵심)

### 1.1 Teacher 모델 선택
**예산 제약이 없으므로 teacher는 전부 Claude Opus 4.8 단일로 통일한다.** (Gemini Flash 등 저비용 대량 teacher 미사용 → provider 혼재로 인한 라벨 분포 불일치 제거)

| 역할 | 모델 | 비고 |
|---|---|---|
| 메인 teacher (대량 + 검수) | **Claude Opus 4.8** (`claude-opus-4-8`, 토파 Bedrock 경유) | 랭킹·근거 라벨 전량 생성 |
| 라벨 신뢰도 게이트 | 동일 Opus 4.8 **self-consistency**(재샘플) + 사람 spot-check | §1.4 κ 게이트 |
| 기존 약라벨 (warm-up) | Qdrant cosine (쿼리↔청크) | teacher 호출 전 student 사전적응용 |

> 토파가 이미 Bedrock Claude를 쓰고 있어 teacher 호출 인프라를 그대로 재사용. provider 불명 시 최신·고성능 Claude를 기본값으로.

### 1.2 입력 구성 (grounding이 핵심)
topa.page JSON에 **동봉돼 온 후보 문장**에 **고유 ID를 부여**해서 teacher에 넣는다. teacher는 span 텍스트를 "쓰지" 않고 **ID만 고른다** → 항상 실제 토큰에 매핑.
```
[QUERY] 잔잔하고 위로되는 가족 이야기
[BOOK b1] 제목: ...
  s1) (시놉시스 문장)
  s2) (리뷰 문장)
  s3) (리뷰 문장)
  ...
[BOOK b2] ...
```

### 1.3 Teacher 프롬프트 (2-pass)
- **Pass A — listwise 랭킹 + 풀 내 하드 네거티브 표시**: 후보 N권을 query 관련도로 정렬 + graded score(0~3 또는 0~1). (RankGPT 슬라이딩 윈도우로 N>윈도우 처리) 같은 호출에서 **풀 안에 섞인 함정(plausible-but-wrong) 책을 `hard_negatives`로 표시**한다 → §3 참조(v1 하드 네거티브 기본 경로).
- **Pass B — 근거 인용**: 상위 책마다 "왜 추천?"의 **근거 문장 ID들**과 한 줄 사유.

출력 스키마(예):
```jsonc
{
  "ranking": [{"book":"b1","score":0.93}, {"book":"b2","score":0.41}, ...],
  "rationales": {
    "b1": {"sentence_ids":["s2","s7"], "reason":"가족의 빈자리를 잔잔히 다룸"},
    ...
  },
  // 풀 안의 함정만 표시(단순 무관 책은 제외). reason: same_genre_diff_mood | title_variant | other.
  "hard_negatives": {
    "b9": {"reason":"same_genre_diff_mood", "note":"같은 가족소설이지만 무드가 반대"}
  }
}
```

### 1.4 비용/품질 통제
- 문장 ID 인용 → 출력 토큰 최소화. 근거는 top-k 책만.
- teacher는 Opus 4.8 단일이므로 신뢰도 게이트는 **(i) Opus self-consistency**(동일 쿼리를 후보 순서 셔플·재샘플로 2~3회 라벨링 → 런 간 **agreement κ**)와 **(ii) 사람 spot-check 샘플**로 측정한다. 어려운 쿼리는 self-consistency 전수.
- **게이트 기준(v1)**:
  - relevance 0~3 grade 기준 weighted κ ≥ 0.60
  - teacher ranking agreement NDCG@10 ≥ 0.85
  - rationale sentence-F1/IoU 평균 ≥ 0.45
  - 존재하지 않는 sentence ID, 빈 rationale, score-ranking 불일치 출력은 자동 폐기/재라벨
- **사람 spot-check 기준(v1)**:
  - pilot 100쿼리는 전수 검수.
  - v1 train은 무작위 3~5% 또는 최소 300쿼리 + self-consistency 낮은 쿼리 전수 검수.
  - 통과 기준: teacher 근거 타당성 평균 ≥ 4.0/5.0, 치명 오류(근거 문장이 query와 무관/반대/존재하지 않음) ≤ 5%, ranking 상위 Top-10 중 명백한 오판 ≤ 10%.
  - 기준 미달 query family는 synthetic 생성 규칙/teacher 프롬프트를 수정하고 재라벨한다.
- κ 미달 배치는 학습 제외 + self-consistency 재샘플/사람검수 샘플 비율 상향.
- 쿼리 dedup·캐시·배치. teacher 호출은 학습 1회성 비용.

**산출물:** `(query, candidate, teacher_score, rationale_sentence_ids)` 라벨셋.

### 1.5 쿼리 소스와 라벨셋 규모
토파 검색 로그·피드백이 **거의 없으므로** synthetic query가 사실상 전부다. 실데이터는 비중을 잡기보다 "있으면 앵커로 섞는" 보조로만 쓴다.

- 쿼리 비율(v1): **synthetic ~95%**, 실제 로그·피드백·팀 작성 쿼리는 확보되는 만큼(목표치로 강제하지 않음). 실데이터 부재가 default 가정.
- synthetic query는 도서 메타/시놉시스/리뷰에서 다음 축으로 만든다: mood, trope, 관계, 상황, 회피조건, 독자 취향, 장르+정서 조합, "이런 건 싫고 이런 건 좋다" 식의 복합 질의. → 실데이터 부족을 메우는 핵심 레버이므로 family 다양성·검수에 비중을 더 둔다.
- 라벨셋 단계 (**예산 제약이 없으므로 규모·검수 비율을 키운다**):
  - pilot: 100 queries × Top-50~60 candidates. 프롬프트/스키마/agreement 검증.
  - v1 train: 10~15k queries × Top-50~60 candidates, rationale은 상위 Top-10 책만.
  - v2/refine: 품질 게이트 통과 후 +30k queries 확장. 어려운 쿼리/teacher-student disagreement 중심.
- split은 랜덤이 아니라 **query family와 book cluster 기준**으로 나눈다. train/valid/test = 80/10/10, 최종 평가는 §4의 독립 평가셋을 별도로 둔다.
- teacher context window는 충분하다고 가정하되, 비용·노이즈 관리를 위해 책당 evidence는 제한한다: v1은 **시놉시스 + 리뷰 문장** 중심, 책당 최대 12~16문장. 초과 시 query-aware cosine/Qdrant preselect로 줄인다.

---

## 2. ⭐ 증류 학습 설계 — Select-then-Predict (C) 단일 아키텍처

본 프로젝트의 student는 **select-then-predict(C) 하나**다. multi-task(B)는 폐기.
> **B를 버린 이유:** B는 score head와 span head가 **병렬**이라, 모델이 내놓은 span이 점수 계산에 실제로 쓰였다는 보장이 없다(설명과 예측의 인과 분리 불가 = "그럴듯한 사후 설명"에 머묾). C는 **점수를 선택된 근거 문장만으로 계산**하므로 "이 근거 때문에 이 점수"가 구조적으로 성립한다. explainable이 목표라면 C가 정답.

### 2.1 아키텍처: Generator → Predictor (bge backbone 공유 + 별도 LoRA 2벌)

```
                query + 후보책의 근거 문장들 s1..sN (각 문장에 ID 부여)
                              │
        ┌─────────────────────┴──────────────────────┐
        │ Generator G  (bge backbone + LoRA_g)         │
        │  단일 forward(query+전체 문장) → 문장별 표현  │
        │  (offset 범위 mean-pool) → Linear → logit π_i │
        │  → 미분가능 게이트 z_i∈[0,1] (Hard-Concrete)  │   ← "무엇이 근거인가"
        └─────────────────────┬──────────────────────┘
                              │  선택 z  (G 학습=soft, P 입력/추론=hard 0/1)
                              ▼
        ┌─────────────────────────────────────────────┐
        │ Predictor P  (bge backbone + LoRA_p)          │
        │  입력 = query + (z_i=1 인 문장만)              │   ← 비선택 토큰은
        │  physically pack → [CLS] → relevance score    │     입력 시퀀스에서 제거
        └─────────────────────────────────────────────┘
```

- backbone(bge-reranker-v2-m3)은 **frozen**, LoRA 2벌(`LoRA_g`/`LoRA_p`)만 학습 → 메모리·속도 절약, 한 backbone을 두 번 forward.
- **선택 단위 = 문장(sentence)**. teacher가 문장 ID로 인용하므로 지도신호가 문장 단위로 깨끗하게 정렬됨. (토큰 단위 선택은 ablation으로만 비교)
- **faithfulness의 물리적 근거:** Predictor 입력은 `z_i=1`인 문장만 **physically pack**해서 만든다. `z_i=0` 문장의 토큰은 P 입력 시퀀스에 존재하지 않음 → 점수가 비선택 토큰에 의존하는 게 불가능. (post-hoc 해석과의 결정적 차이)
- **G의 문장별 logit 추출 방식 (v1 확정):** query+후보책 전체 문장을 **단일 forward**로 인코딩(문장 간 cross-attention 허용)하고, §2.7에서 만든 char/token offset 범위로 **문장별 mean-pool**(또는 문장 앞 경계 토큰 hidden) → Linear → π_i. **faithfulness는 P가 보장**하므로 G가 문장끼리 문맥을 봐도 안전하고, 오히려 "어느 문장이 더 나은 근거인가"를 비교 선택할 수 있어 유리하다. 문장별 개별 인코딩(N forward)은 §2.8 ablation으로만 비교.
  - **전제 검증(W1):** 책당 12~16문장 + query가 bge-reranker-v2-m3 `max_length` 안에 들어가는지 실측 확정(m3 계열은 long-context 지원). 초과 시 §1.5의 evidence 상한/preselect로 줄인다.

#### LoRA 기본값
**LoRA(Low-Rank Adaptation)** 는 frozen backbone의 큰 weight `W`를 직접 바꾸지 않고, 작은 저랭크 행렬 `A/B`만 학습해서 `W + BA` 형태의 보정값을 더하는 방식이다. 즉 bge 전체를 재학습하지 않고 작은 adapter만 학습/저장한다.

- v1 기본: backbone frozen, `LoRA_g`와 `LoRA_p`는 **분리된 adapter**로 둔다. Generator와 Predictor가 하는 일이 다르기 때문이다.
- target modules는 코드에서 **실제 module name을 inspect한 뒤 확정**한다. `q_proj/k_proj/v_proj/o_proj`는 후보일 뿐이며, bge-reranker-v2-m3가 XLM-R/Roberta 계열 naming을 쓰면 `query/key/value/dense`류로 잡는다.
- W1에서 `named_modules()`를 출력해 attention projection module만 regex allowlist로 고정하고, `lora_target_modules.yaml`에 저장한다. linear classifier/head와 pooling head는 별도 ablation 전까지 LoRA target에서 제외한다.
- 시작값: rank `r=16`, alpha `32`, dropout `0.05`. underfit이면 `r=32`, 메모리 부족이면 `r=8` 또는 QLoRA.
- 저장물: base model ID + generator adapter + predictor adapter + tokenizer/sentence-index metadata.

### 2.2 선택 메커니즘 — 미분가능 이산선택

문장 선택은 이산(0/1)이라 그대로면 미분 불가 → **Hard-Concrete gate(L0 정규화 계열)** 로 완화. (Gumbel-Softmax는 ablation 대안)
```
π_i = G(query, s_i)                  # 문장별 선택 logit
z_i = HardConcrete(π_i, temp)        # 학습: (0,1) 연속이되 0/1로 쏠리게
                                      # 추론: z_i = 1[π_i > 0]  (hard, straight-through)
```
- **v1 구현 원칙:** P에는 soft attention mask를 넣지 않는다. 선택된 문장만 실제로 pack해서 P에 넣는다. 따라서 `L_rank` gradient를 P에서 G로 억지로 흘리지 않는다.
- G는 `L_select + L_sparsity + L_continuity`로 직접 학습하고, P는 `L_rank + L_hard`로 학습한다. teacher가 ranking과 rationale을 동시에 주기 때문에 두 신호를 같은 라벨셋에서 정렬한다.
- joint 학습에서 P 입력 mask는 schedule로 섞는다: 초반 teacher rationale mask 중심 → 후반 G hard mask 비율 증가. 이렇게 P가 G의 선택 오류에도 견디게 한다.
- `L_rank → G` gradient가 필요한 방식은 v1 범위 밖 ablation이다. 이 경우 HF 기본 binary `attention_mask`가 아니라 custom additive attention bias 또는 REINFORCE/straight-through surrogate를 별도 구현해야 한다.

### 2.3 손실 함수 (teacher 라벨 = 분포 증류 + 선택 지도)

teacher가 주는 두 신호 — **순위 점수(Pass A)** + **인용 문장 ID(Pass B)** — 가 각각 P와 G로 직결된다.

**(a) 랭킹 증류 — Predictor 출력에**
```
P_t = softmax(teacher_scores / τ)        # 쿼리별 후보책 분포
P_s = softmax(predictor_scores / τ)      # ★ 선택된 근거만으로 계산된 점수
L_rank = KL(P_t ‖ P_s)                    # listwise KD (teacher가 순위만 주면 ListMLE/RankNet 대체)
```
> v1에서 `L_rank`는 Predictor/`LoRA_p`를 업데이트한다. Generator/`LoRA_g`는 아래 `L_select`와 정규화 손실로 업데이트한다.

**(b) 선택 증류 — Generator 게이트에  (★ C 안정화의 핵심)**
```
y_i = 1  if  문장 s_i ∈ teacher 인용 ID,  else 0
L_select = BCE(z_i, y_i)
```
> Lei+2016 원본 select-then-predict는 selection이 **비지도**(예측손실만으로 유도)라 학습이 불안정했다. 우리는 **teacher 인용으로 z를 직접 지도**하므로 그 불안정성을 원천 제거한다 — 증류가 이 아키텍처를 실용화하는 지점.

**(c) 정규화 — 근거의 형태 제어**
```
L_sparsity   = λ_sp · E[ Σ_i z_i / N ]          # 근거는 적게 (몇 문장만)
L_continuity = λ_co · Σ_i |z_i − z_{i−1}|         # 흩어지지 말고 인접하게
```

**(d) 하드 앵커 — known 비관련(0) 라벨 (in-pool 하드 네거티브 + 있으면 피드백)**
```
L_hard = α · BCE(predictor_score, hard_label=0)   # teacher가 풀 안에서 표시한 함정 책
       (+ 실제 feedback 라벨이 있으면 같은 항에 합류)
```
> 핵심 공급원은 **§3의 in-pool 하드 네거티브**: teacher가 Pass A에서 표시한 함정 책이 `hard_label=0`으로 들어와 앵커가 실제로 작동한다(`train_neural.load_batches` → `build_training_batch(..., hard_labels=...)`). 실제 feedback 라벨은 **거의 없으므로** 있으면 같은 항에 보조로 합류시키되(없으면 클릭/저장 proxy), 학습이 feedback에 의존하지 않게 설계한다. α는 작게 둔다.

**(e) 총 손실**
```
L = L_rank + α·L_hard + β·L_select + λ_sp·L_sparsity + λ_co·L_continuity
```
스윕: τ, α, β, λ_sp, λ_co, HardConcrete temp. **trade-off 곡선**(sparsity↑ → 설명 간결 ↔ 랭킹 손실) 탐색이 평가의 중심.

### 2.4 학습 절차 (3단계 + LoRA 설정) + collapse 방지

1. **Warm-up (Predictor 먼저, 부분입력 강건화)**: 마스킹 끄고(z≡1) + **랜덤 부분선택**으로 P만 LoRA 사전적응. bge는 "문서 전체"로만 학습됐는데 C의 P는 **잘려나간 부분 문서**를 채점하므로, 랜덤 마스크로 "조각 입력에서도 점수 내는 법"을 먼저 가르쳐 distribution shift를 흡수한다. (G·P 동시 난수 시작 → 양쪽 붕괴)
2. **Joint distill**: teacher 라벨로 G+P 동시 학습. G는 `L_select` 중심으로 빠르게 정렬하고, P는 teacher mask와 G hard mask를 schedule로 섞은 packed evidence만 보고 `L_rank`를 맞춘다. HardConcrete temp annealing(고→저)으로 점진적 이산화.
3. **Refine (self-distill)**: teacher-student agreement 낮은(어려운) 쿼리만 선별 → Opus 4.8 재라벨(추론예산↑·self-consistency 다수결) → 2~3라운드.
4. LoRA 설정은 §2.1 기본값에서 시작하고, W&B로 rank/alpha·τ·α·β·λ 추적.

**알려진 실패모드 대응 (C 핵심 리스크):**
| 실패모드 | 대응 |
|---|---|
| 전부 선택 (z→1) | `L_sparsity` + sparsity 타깃(후보 문장의 10~30%) |
| 아무것도 선택 안 함 (z→0) | `L_select` 지도(teacher가 ≥1문장 인용) + 최소 1문장 강제 |
| Predictor가 마스크 우회 | 비선택 문장을 P 입력에서 physically pack 단계에 제거 → 우회 경로 구조적 차단 |

### 2.5 추론 & faithfulness

단일 파이프라인(G→P, backbone 2회 forward ≈ baseline 2배 비용). 출력:
```jsonc
{
  "score": 0.91,
  "rationale_sentence_ids": ["s2","s7"],
  "spans": [/* 토큰 오프셋 */],
  "reason": "선택된 근거 문장만 사용해 만든 자연어 추천 사유"
}
```
- **구조적 faithfulness**: 반환된 근거 문장이 점수 계산에 쓰인 **유일한 입력**. 어텐션/IG 같은 사후 추정이 아니라 **인과적으로 보장**.
- ERASER sufficiency/comprehensiveness가 정의상 높게 나와야 정상 → 이를 검증 지표로 사용.

### 2.6 자연어 추천 사유 생성
사용자에게는 span ID만 보여주기보다 자연어 추천 사유가 필요하다. 단, 자유 생성 head를 student에 붙이면 근거 밖 주장을 만들 수 있으므로 v1은 **span-grounded renderer**로 간다.

- 모델 출력: score + 선택 sentence IDs + token/char offsets.
- `reason_builder`: 선택된 문장만 입력으로 받아 1~2문장 추천 사유를 만든다. 원칙은 **추출형/템플릿형**이다.
- 예: `"이 책은 '{근거문장1}'라는 대목처럼 {query facet}에 맞고, '{근거문장2}'에서 {정서/관계}가 드러납니다."`
- teacher의 one-line reason은 renderer 템플릿/평가 참고용으로만 사용한다. inference 시 별도 LLM 호출은 하지 않는다.
- 고급화는 v2: 선택 span만 조건으로 받는 작은 reason generator를 학습하되, unsupported claim 검출을 붙인다.

### 2.7 문장 ID와 토큰 오프셋 매핑
입력 evidence는 **topa.page 통짜 JSON에 동봉된 후보 문장**(시놉시스/리뷰에서 이미 추려진 상태)을 그대로 쓴다. 우리 쪽은 문장을 *추출*하지 않고 **받은 문장에 안정적 ID·offset만 부여**해, teacher가 고른 sentence ID를 학습 label·추론 highlight·자연어 사유까지 같은 좌표계로 유지한다.

1. topa.page 원본 응답은 **immutable raw snapshot**으로 저장한다: `response_id`, `query_id`, `topa_pipeline_version`, retrieval params, request timestamp, response JSON, schema version.
2. JSON의 각 후보 문장을 `book_id, source_type(synopsis/review), source_id, text, text_hash`로 정규화해 보관한다(이미 분리돼 오므로 신규 sentence split 불필요).
3. 화면 표시용 canonical evidence text = 동봉 문장 텍스트 자체. highlight도 이 좌표계로 처리해 raw text 역매핑 복잡도를 없앤다.
4. JSON이 문장 단위가 아니라 문단/통문자로 오는 경우에만 한국어 splitter(`kss` 우선, 실패 시 punctuation fallback)로 분리한다(예외 경로).
5. 각 문장에 안정적인 ID를 부여한다: `{response_id}:{book_id}:{source_type}:{source_id}:{sent_idx}:{text_hash8}`.
6. tokenizer 호출 시 `return_offsets_mapping=True`로 token offset을 받고, 각 token을 sentence char range에 매핑한다.
7. teacher label은 sentence-level BCE에 쓰고, inference 결과는 sentence ID + char offset + token offset으로 반환한다.
8. topa.page pipeline/schema가 바뀌면 기존 라벨을 재매핑하지 않는다. 새 snapshot/dataset version을 만들고 기존 라벨셋은 원본 snapshot으로만 재현한다.

### 2.8 비교군 & ablation

| 구분 | 역할 | 랭킹 | 근거 | faithfulness |
|---|---|---|---|---|
| **Baseline** (off-the-shelf bge) | 비교 기준 | score only | post-hoc(어텐션/IG) | 보장 없음 |
| **Full-input KD bge** | 설명 없는 증류 비교군 | teacher score 증류 | 없음 | 보장 없음 |
| **C select-then-predict** (본 모델) | 산출물 | predictor `L_rank` | `L_select`(+sparsity/continuity) | **구조적 보장** |
| **Opus 4.8 teacher** | 품질 upper bound | listwise teacher | sentence ID 인용 | grounded but 느림 |

ablation: τ/α/β/λ_sp/λ_co, 선택 단위(문장 vs 토큰), 게이트(HardConcrete vs Gumbel), warm-up 유무, LoRA rank/alpha, **G 인코딩 방식(단일 forward pooling[v1] vs 문장별 개별 인코딩)**.

**산출물:** C 체크포인트, ablation 표, **랭킹×근거충실도×sparsity×레이턴시 trade-off 곡선**, baseline 대비 faithfulness 정량 비교.

### 2.9 심화 설계 결정 (load-bearing — 한 줄로 넘기면 안 되는 곳)

C를 실제로 굴리면 아래 5개가 모델의 성패를 좌우한다. 각각을 명시적 결정으로 고정한다.

**(1) 2-패스 레이턴시 = 재인코딩이라 불가피**
G(query+전체 문장 인코딩) → P(query+선택 문장 **재**인코딩). P가 비선택 토큰의 hidden을 재사용하면 faithfulness가 깨지므로 **두 번의 full forward는 구조상 필수**(baseline 대비 ≈2×).
→ 결정: **비대칭 2-패스** — Generator는 1차 검색 **상위 후보(예: Top-20)에만**, 하위는 baseline 점수 유지. 그래도 ≤1s 미달 시 **C를 cheaper student로 2차 증류**. (W12–13 최적화에서 실측 게이트)

**(2) Predictor는 "부분 문서 채점"을 새로 배워야 함 (distribution shift)**
bge는 `(query, 전체 문서)`로 학습됨 → 선택돼 잘린 입력은 분포 밖. → 결정: warm-up을 **랜덤 부분선택 강건화**로 정의(§2.4-1). 이게 실패하면 C가 baseline보다 약해짐.

**(3) 배치 = 쿼리 단위 (listwise ↔ per-book 충돌)**
`L_rank`는 한 쿼리의 **모든 후보책 점수를 동시에** 필요(listwise softmax), `L_select`는 책별. → 한 스텝에 `1 query × N books × ~M sentences × 2 pass`를 올려야 함 = **메모리가 진짜 제약**.
→ 결정: 배치 단위=쿼리, **gradient accumulation + 후보 N 샘플링(sampled listwise)** 로 메모리 타협. N·M을 GPU에 맞춰 스윕.

**(4) faithfulness가 보장/불가한 것 — 평가를 분리하라**
구조적 faithfulness = "점수가 선택 토큰에만 의존" (✅ 보장). "선택이 사람이 납득하는 이유" (❌ 비보장 — teacher 지도로 *teacher 근거에* 정렬될 뿐).
→ 결정: **ERASER sufficiency/comprehensiveness는 정의상 높게 나오므로 *아키텍처 sanity check*로만** 쓴다. 진짜 품질 지표는 **① baseline 대비 랭킹 손실(NDCG 델타) ② 사람-근거 일치도** 둘로 분리(§4 반영).

**(5) 라벨 품질 게이트 = C의 필수 의존성**
B와 달리 C는 **선택이 틀리면 점수의 입력 자체가 틀림** → 랭킹 상한이 선택 품질에 묶임. teacher 근거-라벨 노이즈가 직접 모델을 망친다.
→ 결정: §1.4 agreement(κ) 게이트를 "리스크"가 아니라 **필수 통과조건**으로 격상. κ 미달 라벨은 학습 제외 + 사람검수 샘플 상향.

---

## 3. 토파 재사용 (학습을 위한 도구 — 새로 안 만듦)
학습에 필요한 재료만 빠르게 끌어온다. (상세 파일 경로는 부록)
- **후보 풀(데이터 수집 진입점)**: 쿼리를 **topa.page에 요청**하면 검색 파이프라인(Qdrant+Memgraph→RRF)이 **Top-50~60권 + 각 책의 근거 후보 문장(시놉시스/리뷰에서 추린 문장들)까지 다 붙은 통짜 JSON**으로 반환한다 → 이게 teacher 입력 후보이자 근거 코퍼스 그 자체. **별도 ETL·문장 추출·코퍼스 조인이 불필요**하고, 이 JSON을 그대로 라벨링 파이프라인 입력으로 쓴다. 단, 응답은 반드시 raw snapshot으로 저장한다: `response_id`, `topa_pipeline_version`, retrieval params, schema version, request/response JSON, 문장 `text_hash`. 우리 쪽 sentence_index는 "추출"이 아니라 **받은 문장에 ID·offset만 부여**하는 역할로 축소된다.
- **근거 코퍼스(문장 단위)**: v1은 **topa.page 응답 JSON에 동봉된 후보 문장이 곧 근거 코퍼스** → 문장 ID 부여 대상. `book_chunks`/Qdrant review·synopsis 컬렉션은 JSON에 문장이 부족하거나 보강이 필요할 때만 fallback으로 조회. mood/trope/tag는 hard negative와 synthetic query 생성에 우선 사용하고, 근거 evidence에는 옵션으로만 둔다.
- **하드라벨/평가 앵커**: `book_feedback`/`block_feedback`/`question_history` → `L_hard` + 오프라인 평가. **단 실데이터가 거의 없어 보조 신호로만** 기대(있으면 앵커, 없으면 클릭/저장 proxy 또는 생략).
- **하드 네거티브** — ⭐ **v1 기본 = 풀 안(in-pool) teacher 표시**. 리랭커는 추론 때 **topa 풀 안에서만** 순위를 매기므로, 네거티브도 그 풀에서 와야 train/serve 분포가 정확히 일치한다. RRF 검색은 이미 "같은 장르·다른 무드"류 함정을 Top-50 안에 끌어오므로, teacher에게 풀 밖에서 끌어오게 하는 대신 **풀 안의 함정을 직접 지목**하게 한다.
  - **메커니즘(기본)**: teacher가 Pass A 랭킹과 **같은 호출**에서 `hard_negatives`(book_id별 reason: `same_genre_diff_mood`/`title_variant`/`other`)를 emit한다. 프롬프트는 "풀에 그럴듯하지만 틀린 distractor가 있으니 표면 유사성 말고 근거 문장으로 낮게 채점하고, **함정만** 표시(단순 무관 책은 제외, 없으면 비움)"라고 안내 → teacher soft score는 정직하게 유지되고, 표시된 책은 구성상 **known hard_label=0**으로 §2.3(d) `L_hard` 앵커에 공급된다(soft KD + hard anchor 이중 신호). 라벨(teacher label)에 담기므로 **스냅샷은 불변으로 유지**된다.
  - **학습 연결**: `train_neural.load_batches`가 teacher label의 `hard_negatives`를 `hard_labels`로 복원해 `build_training_batch(..., hard_labels=...)`에 전달 → `_hard_anchor`(weight=α) 손실이 실제 작동. 풀에 함정이 없으면 그 쿼리는 본래 쉬운 쿼리이므로 손실 없음.
  - **선택적 보강 — Memgraph 풀 밖(out-of-pool) 주입**: `teacher.hard_negatives` seam(`HardNegativeSource` Protocol; 오프라인 `StaticHardNegativeSource`(`--hard-negatives <json>`), 운영 `MemgraphHardNegativeSource`(동일 장르·다른 무드 / 제목정규화 변형, cypher executor 주입))으로 풀 밖 후보를 `collect_snapshot(..., payload_transform=inject_hard_negatives)`(fetch 직후·save 직전, `is_hard_negative` 마커)로 섞을 수 있다. 단 **풀 밖 네거티브는 모델이 실제로 볼 일 없는 책을 거부하게 가르치는 분포 불일치**라 v1 필수 아님 — 쉬운 풀의 하드 예시 밀도를 의도적으로 높일 때만 쓴다. 로더는 teacher in-pool `hard_negatives`와 스냅샷 주입 마커를 합집합으로 사용.
- **서빙(drop-in)**: 기존 `/rerank` HTTP 계약에 `spans`/`reason` 필드만 추가해 그대로 꽂음(하위호환). v1은 별도 fallback 정책을 두지 않는다.
- **teacher 호출 인프라**: 토파의 Bedrock Claude(Opus 4.8) 접속 재사용.

---

## 4. 평가
- **라벨 분리 원칙**: teacher 라벨은 train/valid에 쓰되, 최종 성능 평가는 teacher와 독립된 평가셋으로 한다. teacher-labeled test는 디버깅/증류 효율 측정용으로만 사용.
- **독립 평가셋(v1)**: 300 queries × Top-20 candidates. **실데이터가 거의 없으므로 사람이 직접 작성/검수한 쿼리 위주**(실 로그는 확보되는 만큼만 섞음), train query family와 겹치지 않게 구성. relevance 0~3, 근거 타당성 1~5를 사람 2인 이상이 라벨링하고 disagreement는 adjudication. 평가셋 자체의 inter-annotator agreement 목표는 **relevance weighted κ ≥ 0.60, rationale validity κ ≥ 0.60**. 미달 시 라벨 가이드 수정 후 재라벨한다.
- **랭킹**: NDCG@{1,5,10}, MRR, Recall@K (독립 평가셋 + 피드백 held-out 기준)
- **도서특화**: 장르/저자/출판사 다양성, 세트책·개정판 혼입률, 세렌디피티
- **근거 충실도**: ERASER sufficiency/comprehensiveness는 C에선 정의상 높음 → **아키텍처 sanity check**용. 진짜 품질은 **① baseline 대비 NDCG 델타(근거 강제의 랭킹 비용) ② 사람-근거 일치도**로 분리 측정 + teacher 인용 대비 token-F1/IoU
- **사람평가**: 팀원6+베타20, 블라인드 A/B, 근거 타당성 5점(≥4.0)
- **서비스 A/B**: 토파 2주, CTR(+3%)·체류·재방문. 트래픽이 부족하면 정성 세션 로그 + 검색 후 클릭/저장 proxy로 보조.
- **teacher vs student 격차**: 증류 효율(품질 보존율) 리포트
- **비교 대상**: off-the-shelf bge, full-input KD bge, C select-then-predict, Opus 4.8 teacher upper bound.

### KPI
| 항목 | 목표 |
|---|---|
| 랭킹 품질 | full-input KD 대비 NDCG@10 하락 ≤ 5%p |
| 세트책/개정판 혼입률 | baseline 대비 50%↓ |
| 근거 타당성 | 사람평가 평균 4.0/5.0↑ |
| 서비스 지표 | CTR +3%↑ |
| 레이턴시 | p95 ≤ 1s |
| 증류 효율 | Opus 4.8 teacher 대비 NDCG 보존율 리포트 |

---

## 5. 리포 구조 (학습 중심)
```
explainable-reranker/
├── plan.md  pyproject.toml  configs/
│   └── lora_target_modules.yaml   # W1 module inspection 후 확정
├── src/
│   ├── data/
│   │   ├── query_synth.py          # synthetic query 생성(로그 부족 보완)
│   │   ├── snapshot_store.py       # topa.page raw JSON snapshot/version/hash 저장
│   │   └── sentence_index.py       # topa.page JSON 동봉 문장에 ID/offset 부여(분리는 예외경로)
│   ├── teacher/                 # ⭐ LLM teacher 라벨링
│   │   ├── prompts.py           # listwise rank + rationale-by-ID 프롬프트
│   │   ├── label_ranking.py     # Pass A
│   │   ├── label_rationale.py   # Pass B
│   │   └── agreement.py         # Opus self-consistency + 사람 spot-check κ
│   ├── distill/                 # ⭐ 증류 학습 (select-then-predict)
│   │   ├── dataset.py           # snapshot+teacher 라벨 → 학습 샘플(문장ID·토큰 정렬)
│   │   ├── losses.py            # L_rank(KL/ListMLE)+L_hard+L_select+sparsity+continuity
│   │   ├── gates.py             # HardConcrete/Gumbel 미분가능 게이트
│   │   └── trainer.py           # LoRA 2벌, warm-up(P)→joint distill(G+P)→refine
│   ├── models/baseline.py                       # 비교용 off-the-shelf bge
│   ├── models/full_input_kd.py                  # 설명 없는 teacher-score 증류 비교군
│   ├── models/select_predict/{generator,predictor,model}.py  # ⭐ 본 모델
│   ├── topa/{db,qdrant_client,memgraph_client,log_collector}.py  # 재사용 어댑터
│   ├── eval/{ir_metrics,book_metrics,faithfulness,run_eval}.py
│   ├── explain/reason_builder.py              # span-grounded 자연어 추천 사유
│   └── serve/{api,export_onnx,quantize}.py   # /rerank + spans + reason
├── benchmark/  scripts/  notebooks/  tests/
```

---

## 6. 타임라인 (학습에 집중, 16주)
| 주차 | 마일스톤 |
|---|---|
| W1 | 토파 연결(어댑터, topa.page JSON 파싱) + raw snapshot store + 동봉 문장 sentence index + LoRA module inspection + synthetic query generator + baseline 로깅 |
| W2–W3 | **Teacher 라벨링 파이프라인** (프롬프트·문장ID·랭킹+근거), 100쿼리 pilot, agreement 검증, 라벨셋 v1 |
| W4–W6 | **C 학습 1차**: Predictor warm-up → Generator 결합 → joint distill, **collapse 잡기**(sparsity 타깃·temp annealing), full-input KD 비교군 |
| W7–W9 | **C ablation**(τ/α/β/λ_sp/λ_co·게이트·선택단위·LoRA) + refine 라운드 + ranking/faithfulness trade-off 비교 |
| W10–W11 | 독립 평가셋 확정 + 사람평가(6+20) + reason_builder 품질 평가 |
| W12–W13 | 서빙 drop-in(/rerank+spans+reason), FP16/INT8·ONNX/TensorRT(≤1s), Two-tier 설명 |
| W14–W15 | 토파 실트래픽 A/B 2주 (CTR·체류·재방문) |
| W16 | HF 체크포인트·벤치마크 공개, 논문/발표 정리 |

### 단계별 Go/No-Go 기준
| 시점 | 통과 기준 | 미달 시 조치 |
|---|---|---|
| W1 | Top-50 후보 생성, raw snapshot/version/hash 저장, sentence ID/offset 매핑, LoRA target module 확정, baseline metric 로깅 성공 | 토파 어댑터/문장 인덱스/LoRA target부터 고정하고 teacher 호출 보류 |
| W3 | pilot schema valid ≥ 98%, weighted κ ≥ 0.60, NDCG@10 agreement ≥ 0.85, 사람 spot-check 통과 | 프롬프트/문장 preselect 재설계, Opus self-consistency/검수 비율 상향 |
| W6 | 작은 셋 overfit 성공, z→0/1 collapse 없음, packed-evidence P 학습 성공, C NDCG@10이 baseline 대비 -10%p 이내 | warm-up/β/λ/temp/mask schedule 재스윕, full-input KD로 상한 재확인 |
| W9 | C가 full-input KD 대비 NDCG@10 -5%p 이내 + 근거 타당성 pilot ≥ 3.8/5 | C 구조 유지 여부 재판단, 선택 문장 수/teacher 라벨 품질 재점검 |
| W13 | p95 latency ≤ 1s, `/rerank` 응답에 spans/reason 안정 반환 | Top-K 축소, INT8/ONNX, cheaper student 2차 증류 |

---

## 7. 팀 역할 (6인)
| 인원 | 트랙 | 담당 |
|---|---|---|
| 2명 | teacher·데이터 | `src/data/*`+`src/teacher/*`+`src/topa/*` (synthetic query·sentence index·라벨링·후보·코퍼스·agreement) |
| 2명 | 증류 학습 | `src/distill/*`+`src/models/select_predict/*` (G/P·게이트·손실·LoRA·collapse·ablation·refine) |
| 1명 | 평가·벤치마크 | `src/eval/*`+`benchmark/` (메트릭·사람평가·token-F1) |
| 1명 | 서빙 | `src/serve/*`+`src/explain/*` (/rerank+spans+reason·최적화·A/B 운영) |

---

## 8. 리스크 & 대응
| 리스크 | 대응 |
|---|---|
| teacher 라벨 비용 | 전량 Opus 4.8(예산 무제약 → 우선순위 낮음), ID인용으로 출력↓, 캐시·dedup, 1회성 |
| synthetic query 편향 (실데이터 거의 없어 의존도↑) | query family 다양화·검수 강화, 사람작성 독립 평가셋(train family와 분리)으로 일반화 검증, 확보되는 실로그는 앵커로 |
| teacher 랭킹 노이즈 | self-consistency κ 게이트, refine 라운드(하드피드백 앵커 L_hard는 실데이터 빈약으로 보조만) |
| 근거 인용 부정확 | 문장ID 강제(자유텍스트 금지), 사람검수 샘플, IoU 모니터 |
| topa.page 응답 재현 불가 | raw snapshot + pipeline/schema version + retrieval params + text_hash 저장. pipeline 변경 시 기존 라벨 재매핑 금지, 새 dataset version 생성 |
| 문장/토큰 offset 불일치 | canonical evidence text 기준으로 sentence split/highlight 통일, tokenizer offset test 추가 |
| LoRA target module 오지정 | W1 `named_modules()` inspection으로 attention projection allowlist 확정, `lora_target_modules.yaml`에 저장 후 학습 시작 |
| `L_rank`가 G까지 미분된다는 착각 | v1은 P packed-evidence 학습과 G supervised selection을 분리. `L_rank→G`는 custom attention bias/REINFORCE ablation에서만 구현 |
| 자연어 사유가 근거 밖 주장 | v1은 span-grounded 추출/템플릿 renderer만 사용, 별도 inference LLM 호출 금지 |
| 랭킹↔근거 trade-off | β·τ·λ 스윕으로 trade-off 곡선 탐색, sparsity 타깃 조정 |
| C selection collapse (z→0/1, 마스크 우회) | **teacher 인용으로 selection 지도**(비지도 아님), Predictor warm-up 선행, sparsity 타깃, HardConcrete temp annealing, 입력 제거로 우회 차단 |
| 레이턴시 미달 (G→P 2패스) | INT8/distill, Generator 경량화(상위 후보만), Top-K 축소, P warm캐시 |
| 서빙 fallback 없음 = 단일 장애점 | v1은 의도적으로 fallback 미설치(수용 리스크). 대신 모니터링/헬스체크 + 장애 시 수동 롤백 트리거를 운영 안전장치로 둔다 |

---

## 9. 즉시 시작 작업
1. `src/topa/`: **topa.page 요청→통짜 JSON 응답 스키마 확정**(Top-50~60 + 동봉 문장 필드) + 파싱/로깅, DB/Qdrant/Memgraph는 fallback 보강용 smoke test + 후보/문장/피드백 건수 실측
2. `src/data/snapshot_store.py`: `response_id/topa_pipeline_version/retrieval_params/schema_version/text_hash` 포함 raw snapshot 저장
3. `src/data/sentence_index.py`: JSON 동봉 문장 정규화 + sentence ID + char/token offset 매핑 test (신규 분리는 예외경로만)
4. `scripts/inspect_lora_targets.py`: bge-reranker-v2-m3 `named_modules()` 출력 + `configs/lora_target_modules.yaml` 확정
5. `src/data/query_synth.py`: synthetic query family 정의 + 100쿼리 pilot 생성
6. `src/teacher/prompts.py`: 문장ID 기반 listwise+rationale 프롬프트 1차 + 100쿼리 pilot
7. teacher 출력 스키마 검증 + Opus self-consistency(재샘플) κ + 사람 spot-check 측정
8. `src/models/full_input_kd.py`: teacher score만 증류하는 비교군 구현
9. `src/distill/`: G(HardConcrete 게이트)+P packed-evidence 골격 + `L_rank(KL)+L_select(BCE)+sparsity` 손실, 작은 셋 overfit + collapse 여부 확인
10. `src/explain/reason_builder.py`: 선택 span 기반 자연어 추천 사유 템플릿 v1
11. baseline 로깅으로 평가 파이프라인(ir_metrics) 먼저 가동

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
Lei+2016(Rationalizing, select-then-predict 원형), Bastings+2019(Differentiable Binary Variables, Hard-Concrete 근거선택), Louizos+2018(L0 정규화), Paranjape+2020(Information Bottleneck rationale), Jain&Wallace+2019(Attention is not Explanation, post-hoc 한계 근거), DeYoung+2020(ERASER 충실도 평가), Sun+2023(RankGPT, teacher 근거), Xiao+2023(BGE), Hu+2022(LoRA), Zhuang+2023(RankT5, 랭킹손실)
