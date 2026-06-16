# Explainable Reranker

질문에 맞게 책 후보를 **다시 줄 세우면서(rerank)**, 동시에 **"왜 이 책이 상위인지"를 실제 코퍼스 문장으로 짚어주는** 설명 가능한 리랭커입니다.

느리지만 똑똑한 **Teacher LLM**(Claude Opus 4.8)이 `랭킹 + 근거 문장 ID`를 모범답안으로 만들고, 빠른 **Student**(`bge-reranker-v2-m3` + LoRA)가 이를 **증류(distillation)** 로 베껴 배웁니다. 단, Student를 **select-then-predict** 구조로 설계해 점수가 항상 "고른 근거 문장에만" 의존하도록 — 즉 **구조적 faithfulness**를 — 보장합니다.

> 📖 비전공자용 설명은 [explain-for-everyone.md](explain-for-everyone.md), 정확한 수식·아키텍처·손실함수·실행계획은 [plan.md](plan.md)를 참고하세요.

---

## 핵심 아이디어

기존 리랭커는 "이 책 87점"만 말하고 *왜인지는* 알려주지 않습니다. 이 프로젝트는 이유까지 말하되, 그 이유가 **진짜 점수의 근거**가 되도록 만듭니다.

```
[Teacher: Claude Opus 4.8]   query + 후보책(문장에 번호 부여 s1, s2, ...)
        │
        ├─ ① 랭킹/관련도 점수 (listwise)   → 랭킹 증류 타깃
        └─ ② 근거 문장 ID 인용 (grounded)  → 근거 증류 타깃 (환각 불가: 실제 문장만 가리킴)
        │  distillation
        ▼
[Student: bge-reranker-v2-m3 + LoRA, select-then-predict]
  Generator: 근거 문장 선택(z)  →  Predictor: 선택된 근거만 보고 relevance score
  → 점수가 선택된 문장에만 의존 = 구조적 faithfulness
```

- **RankGPT급 품질을 cross-encoder 속도로**: Teacher는 학습 때만 쓰고(5~30s), 추론은 Student 단독(≤1s).
- **환각 원천 차단**: Teacher가 자유 텍스트가 아니라 **번호 매긴 실제 문장 ID**를 인용 → 근거 span이 항상 실제 코퍼스 토큰에 매핑됩니다.

---

## 파이프라인

```
topa.page 후보 수집 → Teacher 라벨링 → 학습(증류) → 평가 → 서빙
   (snapshot)        (ranking+rationale)  (LoRA)    (IR+faithfulness)  (/rerank)
```

1. **수집(collect)** — `topa.page`의 `search-candidates`에서 질문별 후보책을 받아 **불변(immutable) 스냅샷**으로 저장하고, 문장 ID·char/token 오프셋을 인덱싱합니다.
2. **라벨링(label)** — Teacher가 2-pass(랭킹 → 근거)로 모범답안을 생성합니다. 동기 API 경로와, 사람이 단계마다 검수하는 **AWS Bedrock Batch** 경로 둘 다 지원합니다.
3. **학습(train)** — joint distillation: `listwise KD + select BCE + sparsity + continuity + hard anchor`. warmup→anneal 스케줄을 따릅니다.
4. **평가(evaluate)** — 독립 qrels 기반 IR 지표 + rationale 충실성/겹침 지표.
5. **서빙(serve)** — 드롭인 `POST /rerank` (`score`, `rationale_sentence_ids`, `spans`, `reason` 반환).

---

## 설치

코어 패키지는 **의존성이 없습니다(dependency-free)**. 외부/GPU 스택은 선택적 extra로 분리되어 있습니다.

```bash
pip install -e .              # 코어 (오프라인 더미 파이프라인까지 전부 실행 가능)
pip install -e '.[teacher]'   # Anthropic SDK (동기 Opus 라벨 생성)
pip install -e '.[bedrock]'   # boto3 (Bedrock Batch 라벨 생성)
pip install -e '.[gpu]'       # torch/transformers/peft/accelerate (DGX Spark/GB10 학습)
```

- Python ≥ 3.11
- GPU 학습 기본값은 GB10/DGX Spark에 맞춰져 있습니다(CUDA 자동 감지, bf16 autocast, `max_length=8192`). GPU가 없으면 CPU/fp32로 폴백합니다.

환경 변수는 `.env.example`를 `.env.local`로 복사해 채웁니다(스크립트가 `.env` → `.env.local` 순으로 자동 로드).

---

## 빠른 시작

```bash
# 1) 오프라인 더미로 전체 체인 검증 (API 비용 0)
PYTHONPATH=src python3 scripts/run_dummy_pipeline.py
PYTHONPATH=src python3 -m unittest discover -s tests

# 2) 실제 후보 수집 + Opus 라벨 생성 (--dummy 빼고 키 설정 시 실제 호출)
pip install -e '.[teacher]'
PYTHONPATH=src ANTHROPIC_API_KEY=... python3 scripts/collect_and_label.py \
    --queries data/queries.txt --out data --max-sentences 16
# → data/snapshots/<schema>/<response_id>.json, data/labels/<response_id>.json
```

### 대량 라벨링 (Bedrock Batch, 단계별 사람 검수)

스냅샷을 고정한 뒤, 각 단계 산출물을 사람이 검수하며 진행합니다:

```bash
PYTHONPATH=src python3 scripts/teacher_batch.py prepare-ranking  --batch-dir data/teacher_batches/pilot-001 --snapshots data/snapshots
PYTHONPATH=src python3 scripts/teacher_batch.py submit-ranking   --batch-dir ... --role-arn ... --model-id ... --s3-input ... --s3-output ...
PYTHONPATH=src python3 scripts/teacher_batch.py fetch-ranking    --batch-dir ...
PYTHONPATH=src python3 scripts/teacher_batch.py review-ranking   --batch-dir ... --approve-valid
PYTHONPATH=src python3 scripts/teacher_batch.py prepare-rationale --batch-dir ...
PYTHONPATH=src python3 scripts/teacher_batch.py submit-rationale  --batch-dir ... --role-arn ... ...
PYTHONPATH=src python3 scripts/teacher_batch.py fetch-rationale   --batch-dir ...
PYTHONPATH=src python3 scripts/teacher_batch.py review-labels     --batch-dir ... --approve-valid
PYTHONPATH=src python3 scripts/teacher_batch.py finalize          --batch-dir ... --labels data/labels
```

### GPU 학습 (GB10 / DGX Spark)

```bash
pip install -e '.[gpu]'

# (1회) 실제 bge-reranker-v2-m3 트리에서 LoRA 타깃 모듈 추출
PYTHONPATH=src python3 scripts/inspect_lora_targets.py \
    --model-id BAAI/bge-reranker-v2-m3 --output configs/lora_target_modules.yaml

# joint distillation: snapshots + labels → adapter checkpoints
PYTHONPATH=src python3 scripts/train_neural.py \
    --snapshots data/snapshots --labels data/labels \
    --lora-config configs/lora_target_modules.yaml \
    --out checkpoints/neural-v1 --epochs 3 --device cuda --compute-dtype bfloat16

# per-epoch 체크포인트 + validation 기반 모델 선택
PYTHONPATH=src python3 scripts/train_optimal.py ...
```

### 서빙

```bash
PYTHONPATH=src python3 scripts/serve_rerank.py     # POST /rerank, GET /healthz
```

학습된 체크포인트는 `load_neural_model("checkpoints/neural-v1", "configs/lora_target_modules.yaml")`로 로드해 `serve.api.rerank_payload` / `serve.http_app.RerankApp`에 `model=`로 넘깁니다.

---

## 저장소 구조

```
src/explainable_reranker/
├── config/        # 환경 변수 로딩 (.env / .env.local)
├── data/          # topa 응답 파싱, 불변 스냅샷, 문장 인덱싱, 쿼리 합성, train/valid/test 분할
├── teacher/       # Teacher LLM 클라이언트, grounded 프롬프트/라벨링, 배치, hard-negative, self-consistency
├── distill/       # 데이터셋 계약, 손실(KL/BCE/sparsity/continuity), HardConcrete gate, joint 학습 루프
├── models/
│   ├── baseline.py
│   ├── full_input_kd.py          # full-input KD 비교군
│   └── select_predict/           # Generator/Predictor 백엔드 (lexical 스탠드인 + bge+LoRA), 서빙 모델
├── explain/       # span 기반 reason 렌더러
├── eval/          # IR 지표, rationale faithfulness/겹침, qrels 평가 러너
├── serve/         # /rerank, /healthz (stdlib http.server)
├── topa/          # topa.page 클라이언트(더미 + HTTP)와 응답 어댑터
└── io_cache.py    # I/O 캐시 + 비용 원장

scripts/           # collect_and_label, teacher_batch, train_neural, train_optimal, evaluate, serve_rerank, ...
configs/           # lora_target_modules.yaml
tests/             # W1~W11 단계별 단위 테스트 + fixtures
docs/              # implementation-status.md (단계별 구현 현황)
```

### 어댑터 심(seam) 설계

모든 외부/GPU 의존성은 **프로토콜 뒤에 격리**되어 있어, 오프라인 더미로 전체 파이프라인이 돌고 테스트됩니다. 프로덕션 전환은 더미를 실제 구현으로 교체하는 것뿐입니다.

| 외부 시스템 | 프로토콜 | 더미 | 프로덕션 |
|---|---|---|---|
| Teacher LLM | `teacher.llm_client.ChatModel` | `ScriptedChatModel` | `AnthropicClaudeChatModel`, `BedrockClaudeChatModel` (Opus 4.8) |
| topa.page 검색 | `topa.client.TopaPageClient` | `DummyTopaPageClient` | `HttpTopaPageClient` |
| Hard negative | `teacher.hard_negatives.HardNegativeSource` | `StaticHardNegativeSource` | `MemgraphHardNegativeSource` |
| Generator/Predictor | `models.select_predict.backends` | lexical 스탠드인 | `HFSentenceGenerator`, `HFPackedEvidencePredictor` (bge+LoRA) |

---

## 테스트

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
# 또는
pytest        # pyproject.toml에 pythonpath=src, testpaths=tests 설정됨
```

---

## 현재 상태

W1~W11 전 단계의 **로컬 실행 가능한 구현 경로**가 완성되어 있으며, 모든 외부 의존성은 더미 뒤에서 테스트됩니다. live `topa.page` 엔드포인트 대상 수집은 end-to-end 검증되었습니다(134 후보 → 스냅샷 → 라벨 → 학습 배치). 남은 것은 실제 GPU 학습 실행, Bedrock Opus 대량 라벨링, 독립 human-eval 수집입니다. 자세한 현황은 [docs/implementation-status.md](docs/implementation-status.md)를 참고하세요.
