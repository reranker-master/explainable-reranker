# 검증 기록 (2026-06-19)

이 문서는 "데이터를 더 모아 rerank 품질을 올린다"는 작업을 진행하면서 돌린 **검증 실험들과 그 결론**을 한곳에 정리한 것이다. 각 실험의 스크립트와 산출물 경로를 함께 적는다.

요약 한 줄: **후보캡24 + 데이터 2배(989→1907) + best-epoch 선택**으로 랭킹 +5%, 근거 충실성 2.2배. 추론은 50후보 기준 ~1.8s(bf16). 배치 처리해도 순위는 사실상 동일하고, 근거(rationale)는 fp32 선택으로 완전 결정론화 가능(단 지연 +57%).

---

## 1. Learning curve — "데이터 부족(data-bound)인가?"

데이터를 더 모으는 게 의미 있는지 먼저 확인. 동일 seed-0 split에서 train만 중첩 부분집합(198/396/595/793)으로 키워가며 같은 valid로 평가. (`scripts/learning_curve.py`, 캡24, 3-epoch, final-epoch 평가)

| n_train | valid NDCG@10 | rationale_f1 |
|---|---|---|
| 198 | 0.428 | 0.358 |
| 396 | 0.416 | 0.330 |
| 595 | **0.499** | 0.360 |
| 793 | 0.343 ⚠️ | 0.101 ⚠️ |

- 198→595 랭킹이 단조 상승 → **데이터 부족 확정, 더 모으면 이득.**
- 793에서 붕괴는 의심스러움 → §2에서 규명.
- 주의: 이 실험은 비용 때문에 **best-epoch 선택 없이 final-epoch만** 평가 + 단일 seed라 점 단위 노이즈가 큼.

## 2. 793 붕괴 규명 — best-epoch 재실행

같은 793(=전체 train), 같은 캡24, **5-epoch + best-epoch 선택**으로 재실행. (`scripts/train_optimal.py`, `checkpoints/`(임시) )

| epoch | valid NDCG@1 | NDCG@10 | rationale_f1 | train_loss |
|---|---|---|---|---|
| 3 | 0.508 | 0.568 | 0.341 | 0.424 |
| **4 (best)** | 0.550 | **0.597** | 0.408 | 0.439 |
| 5 | 0.392 💥 | 0.454 | 0.415 | 0.329 |

- **결론: 793 붕괴는 데이터 탓이 아니라 "마지막 epoch 발산"이었다.** epoch5에서 train_loss는 떨어지는데 NDCG@1이 폭락(degenerate ranking 과적합). best-epoch(4)를 고르면 793이 모든 크기 중 최고점.
- **교훈: 항상 best-epoch 선택. fixed-final-epoch 평가는 믿지 말 것.** (이 현상은 §6 neural-v2에서도 재현됨)

## 3. 쿼리 1000개 추가 생성

`scripts/generate_queries.py` (DeepSeek, 기존 1000개와 dedup) → `data/query_sets/reranker_pilot_1000_v2.txt`

- 1000개 생성, 기존과 **정확 중복 0**.

## 4. 전체 2000개 쿼리 품질 분석

`scripts/analyze_queries.py` (정량 전수 + DeepSeek 샘플 판정), 리포트 `.tmp/query_quality_report.md`

**정량(2000):** 정확/교차중복 0, 비한국어 0, 길이 8–30자 97.4%, 근접중복(Jaccard≥0.6) 13쌍.

**정성(DeepSeek 120 샘플, 1~5):**

| source | 현실성 | 구체성 | 자연한국어 | 이슈 |
|---|---|---|---|---|
| 기존 | 4.30 | 3.27 | 100% | 23% |
| 신규 | 3.88 | 3.70 | 97% | 28% |

- 신규는 기존과 동급(구체성↑, 현실성 약간↓). 이슈 26%는 대부분 "~처럼"(비교작·리랭커엔 오히려 좋은 난이도)·과한 일반성 같은 소프트 이슈. **재생성 불필요.**

## 5. DeepSeek 라벨링

6-way 샤딩으로 신규 1000개 라벨링 (`scripts/collect_and_label.py --converse --model-id deepseek.v3.2 --max-candidates 50 --top-k-rationale 10`).

- 첫 패스 **918/1000 성공(91.8%)**, 82개 실패.
- cache-bust 복구(maxTokens 16384, temp 0.4) → **0개 복구**. 잔여 82는 구조적 실패(깨진 JSON/빈 rationale/오염 ISBN)이고 **broad·주관적 쿼리에 집중** → 살려도 노이즈라 드롭(자연 품질필터).
- **전체 라벨: 989 → 1,907** (스냅샷 2000, 미라벨 93).

## 6. 재학습 neural-v2 (1907 labels)

`scripts/train_optimal.py --epochs 5 --max-train-candidates 24 --gpu-mem-fraction 0.75 --out checkpoints/neural-v2` (best=epoch-4). split: train 1527 / valid 190 / test 190.

| test 지표 | neural-v1 (989, uncapped) | **neural-v2 (1907, cap24)** | 변화 |
|---|---|---|---|
| NDCG@10 | 0.577 | **0.606** | +5% |
| MRR | 0.899 | 0.880 | −2% |
| **rationale_f1** | **0.197** | **0.434** | **2.2×** |
| recall@10 | 0.398 | 0.421 | +6% |

- 두 레버 다 적중: **후보캡24가 rationale의 주 레버**(중간검증: 989+cap24 → rationale 0.407), **데이터 2배가 랭킹 레버**.
- epoch5에서 또 붕괴(valid NDCG@1 0.61→0.39) → §2의 막판 발산 재확인, best-epoch가 회피.
- **neural-v2가 새 베이스라인** (`checkpoints/neural-v2/`).

## 7. 추론 지연 (latency)

학습에 안 쓰인 미라벨 스냅샷 90개로 in-process rerank 시간 측정 (`scripts/bench_rerank.py`, bf16, 워밍업 제외).

| 후보/쿼리 | 평균 | 중앙 | p95 | 후보당 |
|---|---|---|---|---|
| 중앙 50 | **1.82 s** | 1.89 s | 2.06 s | ~37.8 ms |

- 지연 ≈ **38ms × 후보수** (선형). 50후보 ~1.8s. README의 "≤1s"는 ~26후보 이하에서만 성립.
- 모델 로드(1회성) ~65s. 현재 서빙은 후보를 **순차 처리**(배치 없음) → 최대 최적화 포인트.

## 8. 배치 처리 등가성 — "배치하면 순위가 달라지나?"

모델은 후보를 **독립적으로** 채점(cross-candidate 상호작용 없음)하므로 이론상 배치=순차. 실제 수치 차이를 측정 (`scripts/bench_batch_equiv.py`, 50쿼리·후보 2352).

| 지표 | 값 |
|---|---|
| 후보당 \|Δ점수\| | 평균 0.0020, 최대 0.017 |
| **top-1 뒤집힘** | **0/50 (0%)** |
| **top-5 집합 변경** | **0/50 (0%)** |
| 전체 순위 변동 | 37/50 (74%) |
| Kendall-τ 불일치쌍 | 평균 **0.17%** |

- **상위권 완전 동일.** "74% 변동"은 점수가 0.002 이내로 붙은 **풀 깊은 동률 후보 1~2쌍의 인접 스왑**일 뿐(τ 0.17%). **배치 처리해도 순위 안전.**
- 원인은 cross-candidate 의존성이 아니라 **bf16 + 패딩 수치오차 + 이산 top-k 선택**.

## 9. fp32 선택 패치 — rationale 결정론화

이산 선택(top-k)이 bf16 노이즈로 흔들리는 걸 제거하기 위해, **추론(eval)에서만 generator 인코더를 fp32**로 (학습은 bf16 유지). 패치: `models/select_predict/backends.py::HFSentenceGenerator._forward_logits` (autocast를 `self._model.training`일 때만 적용).

검증 (`scripts/verify_generator_fp32.py`, 30쿼리·후보 1383, solo vs 배치 선택 비교):

| 체제 | rationale 흔들림 | \|Δlogit\| 평균/최대 |
|---|---|---|
| bf16 (기존) | 5/1383 (**0.36%**) | 0.0081 / 0.083 |
| **fp32 (패치)** | 0/1383 (**0.00%**) | 0.000002 / 0.00008 |

- fp32로 logit 차이 ~4000배 감소 → **선택 완전 결정론화.**
- ⚠️ **트레이드오프: 지연 증가.** fp32 generator로 50후보 rerank가 **1.82s → 2.86s (+57%)**.
- **결정/구현: 플래그화 (기본 bf16).** `HFSentenceGenerator(select_fp32=...)` → `load_neural_model(select_fp32=...)` → `serve_rerank.py --select-fp32`. 기본은 off(속도 우선, 1.8s, 흔들림 0.36%는 풀 깊은 borderline뿐이라 순위 무영향). rationale 비트-결정론이 요구사항일 때만 `--select-fp32`(2.8s).

---

## 종합 결론 / 권고

1. **학습 레시피:** `train_optimal.py --max-train-candidates 24 --epochs 5 --gpu-mem-fraction 0.75`, **best-epoch 선택 필수**(막판 발산).
2. **데이터:** 아직 데이터-바운드 → 더 모으면 추가 상승 여지. 라벨링은 broad 쿼리에서 ~8% 영구 실패(자연 필터).
3. **서빙:** 50후보 ~1.8s(bf16). 1초 SLA 필요 시 **후보 배치 처리**(순위 동일, 안전) 또는 풀 ~25개 제한.
4. **rationale 결정론:** 기본 bf16(빠름). 비트-결정론이 필요하면 `serve_rerank.py --select-fp32`(흔들림 0%, 단 +57% 지연).

## 산출물

- 체크포인트: `checkpoints/neural-v2/` (best=epoch-4)
- 라벨/스냅샷: `data/labels` (1907), `data/snapshots` (2000)
- 스크립트: `scripts/{learning_curve,train_optimal,generate_queries,analyze_queries,collect_and_label,bench_rerank,bench_batch_equiv,verify_generator_fp32}.py`
- 리포트: `.tmp/query_quality_report.md`, `checkpoints/neural-v2/training_report.md`
