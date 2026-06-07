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
  (dummy) and `BedrockClaudeChatModel` (Opus 4.8, lazy `boto3`).
  `teacher.grounded_teacher.LLMGroundedTeacher` orchestrates the 2-pass labeling,
  validation, retries, and self-consistency.
- topa.page retrieval: `topa.client.TopaPageClient` protocol with
  `DummyTopaPageClient` and `HttpTopaPageClient` (stdlib `urllib`, injectable
  opener); `collect_snapshot` ties fetch → immutable raw snapshot → parse.
- Neural model: `models.select_predict.backends` defines the Generator/Predictor
  backend protocols (lexical stand-ins satisfy them) plus `HFSentenceGenerator`/
  `HFPackedEvidencePredictor` (bge + LoRA, lazy torch/transformers/peft) and
  `load_lora_config` for `configs/lora_target_modules.yaml`.
- Training: `distill.training.TrainableSelectionGenerator` + `train_selection`
  run a real analytic-gradient loop (BCE + sparsity) that overfits a separable
  set without collapse; `save_checkpoint`/`load_checkpoint` persist the adapter.
- Serving: `serve.http_app.RerankApp` exposes drop-in `POST /rerank` and
  `GET /healthz` with pure, testable routing over stdlib `http.server`.

## Human or external-system gates remaining

- Point `HttpTopaPageClient` at the real topa.page endpoint and store production
  raw snapshots.
- Run `scripts/inspect_lora_targets.py` in the training environment with
  `transformers` and the actual `BAAI/bge-reranker-v2-m3` weights.
- Swap `ScriptedChatModel` for `BedrockClaudeChatModel` and run pilot
  self-consistency plus spot-check labeling.
- Implement the lazy `logits`/`score` forward passes in the HF backends and train
  the LoRA Generator and Predictor on GPU.
- Collect the independent human evaluation set and adjudicate disagreements.

## Local verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 scripts/run_dummy_pipeline.py
```
