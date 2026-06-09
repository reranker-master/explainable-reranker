# Implementation Status

This repository now contains a local, testable implementation path through the
parts of `plan.md` that can be completed before external teacher calls, GPU
training, and human evaluation.

## Implemented locally

- W1 data path: topa.page response parsing, immutable raw snapshots, sentence ID
  and char/token offset indexing, synthetic query generation, LoRA target config.
- W2-W3 teacher path: grounded prompts, teacher label schema validation,
  heuristic dummy teacher, self-consistency agreement metrics.
- W4-W9 distillation path: query-batch dataset contract, KL/BCE/sparsity/
  continuity losses, HardConcrete gate, select-then-predict model contract,
  full-input KD comparison contract.
- W10-W11 evaluation path: independent qrels evaluation, ranking metrics,
  rationale overlap metrics, dummy human-eval fixture, span-grounded reason
  renderer.
- Serving smoke path: function-level `/rerank` response contract with
  `score`, `rationale_sentence_ids`, `spans`, and `reason`.

## Adapter seams (external systems isolated behind protocols)

Every external/GPU dependency now sits behind a protocol with an offline dummy
and a production skeleton, so the whole pipeline runs and is tested locally:

- Teacher LLM: `teacher.llm_client.ChatModel` protocol with `ScriptedChatModel`
  (dummy), `AnthropicClaudeChatModel` (first-party Opus 4.8 via the `anthropic`
  SDK — streaming + adaptive thinking; reads `ANTHROPIC_API_KEY`), and
  `BedrockClaudeChatModel` (Opus 4.8, lazy `boto3`).
  `teacher.grounded_teacher.LLMGroundedTeacher` orchestrates the 2-pass labeling,
  validation, retries, and self-consistency.
- topa.page retrieval: `topa.client.TopaPageClient` protocol with
  `DummyTopaPageClient` and `HttpTopaPageClient` (stdlib `urllib`, injectable
  opener); `collect_snapshot` ties fetch → immutable raw snapshot → parse.
  `HttpTopaPageClient` defaults to the live `https://www.topa.page/api/search/
  search-candidates` endpoint, sends a browser UA (the endpoint is behind
  Cloudflare), and omits `top_k` by default so the full candidate set is
  reranked. `topa.adapter.parse_topa_page_response` handles the live schema
  (nested `book.isbn`/`book.title`, `retrieval_debug.rrf_score`, and the
  `chunks` dict `{synopsis, review}`) as well as the flat compat schema.
- Hard-negative mining: `teacher.hard_negatives.HardNegativeSource` protocol with
  `StaticHardNegativeSource` (dummy/`--hard-negatives <json>`) and
  `MemgraphHardNegativeSource` (production skeleton; injectable cypher executor for
  the plan §3 strategies — same-genre/opposite-mood and 제목정규화 변형). Plan §5.1.3
  distractors are mixed into the pool via `collect_snapshot(payload_transform=
  inject_hard_negatives)` (between fetch and save, so the snapshot = the pool the
  teacher saw), marked on `TopaBookCandidate.is_hard_negative`, and recovered at
  train time by `train_neural` via `hard_label_map` → `build_training_batch(...,
  hard_labels=...)` so the `_hard_anchor` loss actually fires. The teacher prompt is
  distractor-aware (score plausible-but-wrong books low) but is NOT told which
  candidates are injected, keeping its soft scores honest alongside the known
  hard_label=0 anchor.
- Neural model: `models.select_predict.backends` defines the Generator/Predictor
  backend protocols (lexical stand-ins satisfy them) plus `HFSentenceGenerator`/
  `HFPackedEvidencePredictor` (bge + LoRA, lazy torch/transformers/peft) and
  `load_lora_config` for `configs/lora_target_modules.yaml`. The neural forward
  passes are now implemented: G does a single-forward encode + per-sentence
  mean-pool over char offsets → linear head → π_i; P runs the (query, packed
  evidence) cross-encoder → score. Both expose `trainable_parameters`,
  `train_mode`/`eval_mode`, and `save_pretrained`/`from_pretrained`, default to
  CUDA + bf16 autocast (GB10/DGX Spark), and fall back to CPU/fp32 off-GPU.
- Training: `distill.training.TrainableSelectionGenerator` + `train_selection`
  run a real analytic-gradient loop (BCE + sparsity) that overfits a separable
  set without collapse; `save_checkpoint`/`load_checkpoint` persist the adapter.
  `distill.neural_training.train_joint` is the GPU counterpart: a torch autograd
  joint-distillation loop (listwise KD + select BCE + sparsity + continuity +
  hard anchor) that follows the warmup→anneal schedule (teacher citations early,
  generator selections late). `models.select_predict.neural_model.load_neural_model`
  reloads a trained checkpoint straight into the serving model.
- Serving: `serve.http_app.RerankApp` exposes drop-in `POST /rerank` and
  `GET /healthz` with pure, testable routing over stdlib `http.server`.

## Human or external-system gates remaining

- Point `HttpTopaPageClient` at the real topa.page endpoint and store production
  raw snapshots.
- Run `scripts/inspect_lora_targets.py` in the training environment with
  `transformers` and the actual `BAAI/bge-reranker-v2-m3` weights.
- Swap `ScriptedChatModel` for `BedrockClaudeChatModel` and run pilot
  self-consistency plus spot-check labeling.
- Install the GPU extras on the DGX Spark and train the LoRA Generator/Predictor:
  `pip install -e '.[gpu]'` then `scripts/train_neural.py` (see "GPU training on
  GB10 / DGX Spark" below). The forward passes are implemented; what remains is
  running the actual training with real labels and weights.
- Collect the independent human evaluation set and adjudicate disagreements.

## Local verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 scripts/run_dummy_pipeline.py
```

## End-to-end: collect → label → train

```bash
# 0. Collect topa candidates + generate Opus "best case" labels.
#    --dummy runs the whole chain offline (no API cost); drop it + set
#    ANTHROPIC_API_KEY to generate real Opus 4.8 labels.
pip install -e '.[teacher]'
PYTHONPATH=src ANTHROPIC_API_KEY=... python3 scripts/collect_and_label.py \
    --queries data/queries.txt --out data --max-sentences 16
# → data/snapshots/<schema>/<response_id>.json  and  data/labels/<response_id>.json
```

The live `/api/search/search-candidates` endpoint returns the full candidate set
(no `top_k`), and the whole candidate list is reranked. The collect step was
verified end-to-end against the live endpoint (134 candidates → snapshot → label
→ training batch with gold rationale rows).

## GPU training on GB10 / DGX Spark

The core package is dependency-free; the neural stack is an optional extra.

```bash
# 1. Install the GPU extras (torch/transformers/peft/accelerate).
pip install -e '.[gpu]'

# 2. (Once) inspect real LoRA target modules from the bge-reranker-v2-m3 tree.
PYTHONPATH=src python3 scripts/inspect_lora_targets.py \
    --model-id BAAI/bge-reranker-v2-m3 --output configs/lora_target_modules.yaml

# 3. Joint distillation: snapshots + teacher labels -> adapter checkpoints.
PYTHONPATH=src python3 scripts/train_neural.py \
    --snapshots data/snapshots --labels data/labels \
    --lora-config configs/lora_target_modules.yaml \
    --out checkpoints/neural-v1 --epochs 3 \
    --device cuda --compute-dtype bfloat16

# 4. Serve the trained model (drop-in /rerank).
#    load_neural_model("checkpoints/neural-v1", "configs/lora_target_modules.yaml")
#    -> pass as `model=` to serve.api.rerank_payload / serve.http_app.RerankApp.
```

Defaults target the GB10: CUDA device auto-detect, bf16 autocast over fp32 master
weights (LoRA adapters + heads), `max_length=8192` to fit ~12-16 evidence
sentences per book (plan §1.5/§2.1). Sentences that exceed `max_length` fall back
to the CLS state and are counted in `HFSentenceGenerator.truncated_sentences`.
