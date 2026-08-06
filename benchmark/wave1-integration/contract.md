# Wave 1 benchmark integration contract

## Goal

Integrate the 13 Wave 1 evaluations requested on 2026-08-06, in the supplied
order, while preserving official dataset, prompt, sampling, and scoring
semantics.

## Frozen candidates

1. TruthfulQA MC2
2. C-Eval
3. BigBench Hard (all 23 tasks and aggregate)
4. OCRBench v1
5. Global PIQA
6. BrowseComp+
7. AIME 2026
8. HMMT November 2025
9. HMMT February 2026
10. IMOAnswerBench
11. GSM8K Hard
12. HumanEval+
13. MBPP+

## Acceptance matrix

Every candidate is checked against the same requirements: authoritative public
source, redistributable or remotely loadable data, immutable revision when the
host supports it, faithful prompt and scoring, distinct registry identity,
offline unit tests, import smoke test, and documented limitations.

The primary metric is the number of candidates that satisfy every applicable
requirement and pass their focused tests. Secondary metrics are unsupported or
blocked candidates, protocol deviations, network-bound smoke tests, and added
dependencies.

## Fairness and evidence rules

- No benchmark is represented by a similarly named substitute.
- A missing or non-public 2025/2026 dataset is recorded as blocked rather than
  reconstructed from memory or unofficial questions.
- Existing implementations are reused only when semantics remain identical.
- Generated code is never executed directly on the host as part of scoring.
- Research failures and implementation failures remain visible in `matrix.tsv`.

## Environment

- Repository: `openbench2`
- Date frozen: 2026-08-06
- Python commands run after `source .venv/bin/activate`
- Existing uncommitted LiveCodeBench work is out of scope and must be preserved.

## Stopping condition

Stop after all 13 candidates are either implemented and verified or have a
source-backed blocking rationale. Do not begin Wave 2.
