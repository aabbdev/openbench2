# openbench2 coverage of model-card evaluations

## Implementation progress

Wave 1 adds C-Eval, the complete BBH aggregate, OCRBench v1, AIME 2026,
HMMT November 2025, HMMT February 2026, GSM-Hard, HumanEval+, MBPP+, and
LiveCodeBench v6. Global PIQA generation is also available, but remains partial
for card-reproduction purposes because cloze `acc_bytes` is unsupported.

Current exact scored-row coverage is **66/186 (35.5%)**, with 6 partial and 114
missing rows. This is an increase of 16 exact rows over the 50/186 audit
baseline. LiveCodeBench accounts for five rows; C-Eval for two; current
competition math for six; and completing BBH plus adding OCRBench v1 for three.
The detailed per-card tables below are retained as the pre-implementation
baseline so the original gap analysis remains auditable. Current Wave 1 protocol
status is recorded in [`wave1_status.md`](wave1_status.md).

BFCL v4 single-turn is now runnable across all 13 official single-turn
categories through provider-native tools or safely parsed prompted calls. It is
kept explicitly partial: the model-card `BFCLv4` overall score also weights
multi-turn and agentic web-search/memory sections, which are not represented by
the `bfcl_v4_single_turn` registry ID.

The four official multi-turn categories are now available as
`bfcl_v4_multi_turn`, backed by the pinned upstream state checker in Docker.
Memory and web-search are available as `bfcl_v4_agentic_offline`, and the
10/10/10/30/40 weighted composition as `bfcl_v4_offline`. These remain partial
for historical-score reproduction because the agentic environment uses frozen
evidence rather than live SERPAPI results and precomputed memory prerequisites.

## Classification rule

- **Have**: the same public dataset/subset/version is runnable in openbench2,
  possibly under a normalized registry ID.
- **Partial**: openbench2 has a related family, but the card's subset, version,
  metric, tool condition, or aggregate cannot currently be reproduced.
- **Missing**: no matching implementation is registered.
- These labels measure task coverage, not score reproducibility. Most Qwen,
  LiquidAI, Nanbeige, and Gemma rows omit at least one of metric, shots, prompt,
  harness revision, or sampling details.

The registry evidence is in [`src/openbench/config.py`](../../src/openbench/config.py),
with implementations under [`src/openbench/evals`](../../src/openbench/evals).

At the pre-implementation audit baseline, across 186 scored-row occurrences,
50 map to the same benchmark
dataset/variant, 9 map only to a related or incompatible variant, and 127 are
missing. These counts retain duplicate rows and distinct configurations such as
VideoMME with versus without subtitles.

## Coverage by card

| Card | Have | Partial | Missing |
|---|---|---|---|
| Qwen3.6-27B | MMLU-Pro; MMLU-Redux; SuperGPQA; GPQA Diamond; HLE (no tools); HMMT Feb 25; MMMU; MMMU-Pro; MathVista mini; MMStar | OCRBench (only OCRBenchV2 exists) | SWE-bench Verified/Pro/Multilingual; Terminal-Bench 2.0; SkillsBench; QwenWebBench; NL2Repo; both Claw-Eval aggregates; QwenClawBench; C-Eval; LiveCodeBench v6; HMMT Nov 25/Feb 26; IMOAnswerBench; AIME26; DynaMath; VlmsAreBlind; RealWorldQA; MMBench; SimpleVQA; CharXiv; CC-OCR; ERQA; CountBench; RefCOCO; EmbSpatialBench; RefSpatialBench; VideoMME; VideoMMMU; MLVU; MVBench; V*; AndroidWorld |
| LFM2.5-2.6B | AIME25; IFBench | BrowseComp+ (base BrowseComp only) | AA Omniscience; LiveCodeBench v6; Multi-IF; IFStruct; BFCLv4; ToolSandbox; τ³-Bench Banking; Claw-Eval average (EN); PinchBench |
| Qwen3.5-4B | MMLU-Pro; MMLU-Redux; SuperGPQA; GPQA Diamond; IFEval; IFBench; MultiChallenge; HMMT Feb 25; TAU2-Bench; MMMLU; MMMU; MMMU-Pro; MathVista mini; MMStar | Global PIQA (English PIQA only); OCRBench (V2 only) | C-Eval; AA-LCR; LongBench v2; HMMT Nov 25; LiveCodeBench v6; OJBench; BFCL-V4; VITA-Bench; DeepPlanning; MMLU-ProX; NOVA-63; INCLUDE; PolyMATH; WMT24++; MAXIFE; MathVision; We-Math; DynaMath; both ZEROBench rows; VlmsAreBlind; BabyVision; RealWorldQA; MMBench; SimpleVQA; HallusionBench; OmniDocBench1.5; CharXiv; MMLongBench-Doc; CC-OCR; AI2D_TEST; ERQA; CountBench; RefCOCO; EmbSpatialBench; RefSpatialBench; LingoQA; Hypersim; Nuscene; both VideoMME modes; VideoMMMU; MLVU; MVBench; LVBench; MMVU; ScreenSpot Pro; OSWorld-Verified; AndroidWorld; TIR-Bench; V*; SLAKE; PMC-VQA; MedXpertQA-MM |
| Nanbeige4.2-3B | HLE without search; SciCode; GPQA-Diamond; IF-Bench | DeepResearch Bench II (openbench has the original DeepResearch Bench, not Bench II) | GDPval/rubrics; Agent-IF-Oneday; Office-QA-Pro; Pinch-Bench-V2; Claw-Gym; Claw-Eval pass^3; MCP-Atlas; SWE-Bench Verified/Pro; Terminal-Bench 2.0; HMMT-Feb-2026; IMO-Answer-Bench; LiveCodeBench-V6; AA-LCR; Recruit-Bench; ResearchRubrics |
| Phi-4-mini-instruct scored table | MMLU; MMLU-Pro; ARC Challenge; BoolQ; GPQA; HellaSwag; OpenBookQA; PIQA; Social IQA; Winogrande; Multilingual MMLU/MMMLU; MGSM; GSM8K; MATH | BigBench Hard (openbench has 18 core tasks, not the complete 23-task aggregate); TruthfulQA MC2 (openbench currently implements MC1) | Arena Hard |
| Phi appendix additions | MedQA; ANLI; TriviaQA; HumanEval; MBPP; IFEval; AGI Eval; Toxigen | none | Berkeley Function Calling; GSM8K Hard; HumanEval+; MBPP+; LiveCodeBench (`LiveCodeBenh` typo in source); LiveBench; BigCode Bench; Spider; MEGA; DecodingTrust; XSTest; unnamed/internal evals |
| Gemma-4-E4B-it | MMLU Pro; GPQA Diamond; Tau2 average over retail/airline/telecom; HLE no tools; MMMLU; MMMU Pro | HLE with search (dataset exists, search-enabled harness does not); MRCR v2 8-needle 128k (8-needle/context controls exist, but no explicit v2 identity) | AIME 2026; LiveCodeBench v6; Codeforces ELO; BigBench Extra Hard; OmniDocBench 1.5; MATH-Vision; MedXPertQA MM; CoVoST; FLEURS |

## Existing implementations that cover the cards

| Card name/family | openbench2 registry ID(s) | Important caveat |
|---|---|---|
| MMLU / MMLU-Pro / MMLU-Redux | `mmlu`, `mmlu-pro`, `mmlu-redux` | Prompt/shot settings must be matched per card. |
| Multilingual MMLU / MMMLU | `mmmlu` and language subtasks | Alias-backed. |
| GPQA / GPQA Diamond | `gpqa`, `gpqa_diamond` | Diamond is separate and available. |
| SuperGPQA | `supergpqa` | Available. |
| IFEval / IFBench / MultiChallenge | `ifeval`, `ifbench`, `multichallenge` | Available. |
| HLE without tools | `hle`, `hle_text` | No search-enabled solver matching Gemma's HLE-with-search row. |
| AIME25 / HMMT Feb 25 | `aime_2025` or `gpt_oss_aime25`; `hmmt_feb_2025` | Newer 2026/November variants are absent. |
| Tau2 | `tau_bench_retail`, `tau_bench_airline`, `tau_bench_telecom` | Implementation downloads the official `sierra-research/tau2-bench`; reproducing Qwen3.5's score still requires confirming its stated airline fix. |
| MMMU / MMMU-Pro | `mmmu`, `mmmu_pro` | Available. |
| MathVista mini | `mathvista` | The implementation supports the testmini split. |
| MMStar | `mmstar` | Available. |
| MRCR 8 needle | `openai_mrcr_8n` | Related support only: `max_context_size` can constrain to 128k, but the task does not explicitly identify or pin Gemma's reported v2. |
| SciCode | `scicode` | Optional dependency group may be required. |
| Classic Phi tasks | `arc_challenge`, `boolq`, `hellaswag`, `openbookqa`, `piqa`, `social_iqa`, `winogrande`, `mgsm`, `gsm8k`, `math` | Dataset coverage exists; reproduce card shots/CoT separately. |
| Phi appendix tasks | `medqa`, `anli`, `triviaqa`, `humaneval`, `mbpp`, `ifeval`, `agieval`, `toxigen` | `HumanEval+` and `MBPP+` are distinct and absent. |

## What to implement, in order

### P0 — highest reuse across these cards

1. ~~**LiveCodeBench v6**~~ — implemented as `livecodebench_v6` with the
   official cumulative release composition and scoring protocol.
2. **SWE-bench + Terminal-Bench 2.0** — add Verified, Pro, Multilingual, and
   Terminal-Bench as versioned agent tasks with explicit scaffold selection.
   Do not bake Qwen's corrected SWE-bench Pro set into the public canonical ID.
3. **Current competition math** — AIME 2026, HMMT Nov 2025, HMMT Feb 2026,
   and IMOAnswerBench. Keep year/month in IDs and dataset metadata.
4. **BFCL v4** — shared by LFM and Qwen3.5 and strategically useful for tool
   calling. Preserve AST/function-call scoring and category aggregates.
5. **C-Eval** — a straightforward academic coverage gap appearing on both
   Qwen cards.
6. **TruthfulQA MC2** — extend the existing task rather than creating an
   unrelated implementation; current code explicitly scores MC1.
7. **Full BBH aggregate** — add the five missing tasks and a canonical
   23-task aggregate before claiming Phi's BigBench Hard row as reproducible.

### P1 — shared multimodal and agent coverage

1. Build shared multimodal primitives, then add the evals repeated across both
   Qwen cards: DynaMath, VlmsAreBlind, RealWorldQA, MMBench, SimpleVQA,
   CharXiv, OCRBench v1, ERQA, CountBench, RefCOCO, EmbSpatialBench,
   RefSpatialBench, VideoMME, VideoMMMU, MLVU, MVBench, V*, and AndroidWorld.
2. Add agent/tool suites as separately versioned packages: Claw-Eval,
   PinchBench/Pinch-Bench-V2, ToolSandbox, SkillsBench, and τ³-Bench. Do not
   alias τ³-Bench to the existing Tau2 implementation.
3. Add HLE-with-search as a solver/configuration over the existing HLE dataset,
   with search provider and cost captured in run metadata.
4. Evaluate whether DeepResearch Bench II can share the existing original
   DeepResearch Bench scorer; it must remain a separate registry ID unless the
   datasets and metric definitions are proven identical.

### P2 — breadth and card-specific tails

- Gemma: BigBench Extra Hard, OmniDocBench 1.5, MATH-Vision, MedXPertQA-MM,
  CoVoST, and FLEURS.
- Phi: Berkeley Function Calling, GSM8K Hard, HumanEval+, MBPP+, LiveBench,
  BigCode Bench, Spider, MEGA, DecodingTrust, and XSTest.
- Qwen-specific long tail: LongBench v2, OJBench, WMT24++, MAXIFE,
  MMLU-ProX, MathVision/We-Math/PolyMATH, document/UI/navigation, medical VQA,
  and the remaining proprietary or newly introduced suites.

## Findings that affect implementation design

- **Versioned IDs are mandatory.** The cards mix AIME25/AIME26, multiple HMMT
  dates, OCRBench/OCRBenchV2, and DeepResearch Bench/Bench II.
- **Harness is part of the eval.** SWE-bench, Terminal-Bench, Tau2, Claw, and
  OpenClaw scores depend on agent scaffold and tool environment.
- **Metric/configuration is part of identity.** TruthfulQA MC1 is not MC2;
  HLE no-tools is not HLE-with-search; VideoMME with subtitles is not without.
- **Store provenance.** Nanbeige's HMMT-Feb-2026 README and generated YAML
  disagree (82.8 vs 82.1), and several cards omit metrics entirely.
