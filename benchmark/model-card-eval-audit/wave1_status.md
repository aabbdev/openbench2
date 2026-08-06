# Wave 1 implementation status

Status after the source/protocol audit and the current OpenBench integration pass.
"Implemented" means the public artifact is pinned and the local protocol is tested;
"adapted" means the local score is useful but is not directly comparable with the
historical leaderboard protocol.

| Evaluation | Status | Reproducibility note |
|---|---|---|
| TruthfulQA MC2 | Blocked | MC2 requires probability mass over every true/false continuation. Inspect's provider-neutral generation interface does not expose portable forced-continuation log-likelihoods; generated-choice accuracy would be a different metric. |
| C-Eval / C-Eval Hard | Implemented | Pinned `ceval/ceval-exam`; all 52 subjects, official five-shot Chinese prompt, eight-subject hard subset, grouped subject/category metrics. |
| BIG-Bench Hard | Implemented | Registry now covers all 23 conceptual tasks / 27 physical configurations. The nine formerly absent configurations use full free-response targets instead of letter-only truncation. |
| OCRBench v1 | Implemented | Pinned 1,000-example `echo840/OCRBench`; multimodal transport and historical substring scorer, kept separate from OCRBenchV2. |
| Global PIQA v1 | Implemented (generation) | Both pinned parallel/non-parallel corpora, official 2-choice/4-choice prompts and sampling, and hierarchical macro aggregation. Cloze `acc_bytes` remains unsupported because it needs continuation log-likelihoods. |
| BrowseComp-Plus | Blocked | Requires the fixed 100,195-document corpus, a versioned retriever/search tool, multi-step agent traces, evidence recall, and a Qwen3-32B judge. It is not a BrowseComp dataset alias. LiquidAI's OpenClaw scaffold/configuration is unpublished. |
| AIME 2026 | Implemented | Pinned 30-problem MathArena artifact, four runs, boxed final answers. This is a third-party republication; original MAA redistribution rights were not independently established. |
| HMMT November 2025 | Implemented subset | Pinned 30-problem MathArena subset, not the complete 66-problem human contest. |
| HMMT February 2026 | Implemented subset | Pinned 33-problem MathArena subset, not the complete 76-problem contest. Fraction/power answers now use a non-AIME scorer. |
| IMO-AnswerBench | Blocked for faithful integration | The maintained v2 CSV changes 29 rows and still has a malformed row; candidate-generation settings and the Gemini 2.5 Pro judge snapshot are missing. HF mirrors contain deprecated v1. |
| GSM8K Hard / PAL GSM-Hard | Implemented (adapted) | Pinned canonical 1,319-row artifact and strict `<1e-3` numeric scoring. The local task is direct-answer generation; canonical PAL instead generates and securely executes Python with an eight-shot prompt. |
| HumanEval+ | Implemented | Pinned EvalPlus HumanEval+ v0.1.10; base and augmented differential tests run in a fail-closed, network-disabled Docker sandbox. Hidden tests stay out of Inspect logs and are unlinked before candidate execution. |
| MBPP+ | Implemented | Pinned EvalPlus MBPP+ v0.2.0 with the same isolated base/plus execution, adaptive limits, special oracles, and hidden-test protections. |

## Remaining architecture work

1. Add a provider capability for forced-continuation log-likelihoods; then implement
   TruthfulQA MC2 and Global PIQA cloze/`acc_bytes` without metric substitution.
2. Add a versioned corpus/retriever/tool-trace layer; then implement BrowseComp-Plus.
3. Revisit IMO-AnswerBench only after upstream repairs the v2 row and a reproducible
   judge snapshot/protocol is available, or expose an explicitly named OpenBench
   adaptation rather than claiming historical-score equivalence.
