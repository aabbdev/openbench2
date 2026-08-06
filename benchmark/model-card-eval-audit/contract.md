# Model-card eval coverage audit

- Candidates: Qwen/Qwen3.6-27B, LiquidAI/LFM2.5-2.6B,
  Qwen/Qwen3.5-4B, Nanbeige/Nanbeige4.2-3B,
  microsoft/Phi-4-mini-instruct, google/gemma-4-E4B-it.
- Sources: the official Hugging Face model-card README for each candidate.
- Task matrix: every benchmark explicitly reported in a result table or
  evaluation-results passage. Training-only dataset mentions are excluded.
- Primary classification: exact, alias-backed, partial/variant, or missing in
  the current openbench2 registry and implementation.
- Preserved qualifiers: benchmark spelling, suite/subtask, metric, shot count,
  prompting/reasoning mode, modality, language, and model-card context.
- Fairness budget: official card content only; no third-party leaderboard
  substitutions. Every card receives the same extraction and classification
  treatment.
- Stopping condition: every extracted model-card eval has a source and an
  openbench2 support classification, with unresolved ambiguities called out.
- Baseline: clean `main` worktree at audit start; registry in
  `src/openbench/config.py` and implementations under `src/openbench/evals/`.
