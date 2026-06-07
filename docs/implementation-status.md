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

## Human or external-system gates remaining

- Connect the real topa.page endpoint and store production raw snapshots.
- Run `scripts/inspect_lora_targets.py` in the training environment with
  `transformers` and the actual `BAAI/bge-reranker-v2-m3` weights.
- Replace the heuristic dummy teacher with Bedrock Claude Opus calls and run
  pilot self-consistency plus spot-check labeling.
- Train the HF/LoRA Generator and Predictor backends on GPU.
- Collect the independent human evaluation set and adjudicate disagreements.

## Local verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 scripts/run_dummy_pipeline.py
```
