# Model-card evaluation inventory

This inventory preserves the model-card spellings. Parenthetical/subscript
qualifiers are part of the reported evaluation configuration, not silently
collapsed aliases. A name in the Phi appendix means the card says the model was
evaluated on it, even where no score is published.

## Qwen/Qwen3.6-27B

Source: [revision 6a9e13b](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md#L56)

45 scored rows:

1. SWE-bench Verified
2. SWE-bench Pro
3. SWE-bench Multilingual
4. Terminal-Bench 2.0
5. SkillsBench (Avg5)
6. QwenWebBench
7. NL2Repo
8. Claw-Eval (Avg)
9. Claw-Eval (Pass^3)
10. QwenClawBench
11. MMLU-Pro
12. MMLU-Redux
13. SuperGPQA
14. C-Eval
15. GPQA Diamond
16. HLE
17. LiveCodeBench v6
18. HMMT Feb 25
19. HMMT Nov 25
20. HMMT Feb 26
21. IMOAnswerBench
22. AIME26
23. MMMU
24. MMMU-Pro
25. MathVista (mini)
26. DynaMath
27. VlmsAreBlind
28. RealWorldQA
29. MMStar
30. MMBench (EN-DEV-v1.1)
31. SimpleVQA
32. CharXiv (RQ)
33. CC-OCR
34. OCRBench
35. ERQA
36. CountBench
37. RefCOCO (avg)
38. EmbSpatialBench
39. RefSpatialBench
40. VideoMME (w sub.)
41. VideoMMMU
42. MLVU
43. MVBench
44. V*
45. AndroidWorld

Protocol notes: SWE-bench uses Qwen's internal agent scaffold; the card says
its SWE-bench Pro set contains corrections to problematic public tasks.
SkillsBench is a 78-task self-contained subset averaged across five runs.
NL2Repo uses Claude Code for comparator models. No uniform shot count or metric
name is supplied for the complete table.

## LiquidAI/LFM2.5-2.6B

Source: [revision a4e00e8](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/a4e00e83c0979ee9deb88d04b6360599fa956656/README.md#L203-L222)

12 scored rows:

1. AA Omniscience
2. AIME25
3. LiveCodeBenchv6
4. IFBench
5. Multi-IF
6. IFStruct
7. BFCLv4
8. ToolSandbox
9. τ³-Bench Banking
10. Claw-Eval average (EN)
11. PinchBench
12. BrowseComp+ (OpenClaw)

The card supplies scores but no metric names, shot counts, prompting protocol,
dataset revisions, or confidence intervals.

## Qwen/Qwen3.5-4B

Source: [revision 851bf6e](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md#L66)

70 scored rows (69 benchmark names; VideoMME has two configurations):

1. MMLU-Pro
2. MMLU-Redux
3. C-Eval
4. SuperGPQA
5. GPQA Diamond
6. IFEval
7. IFBench
8. MultiChallenge
9. AA-LCR
10. LongBench v2
11. HMMT Feb 25
12. HMMT Nov 25
13. LiveCodeBench v6
14. OJBench
15. BFCL-V4
16. TAU2-Bench
17. VITA-Bench
18. DeepPlanning
19. MMMLU
20. MMLU-ProX
21. NOVA-63
22. INCLUDE
23. Global PIQA
24. PolyMATH
25. WMT24++
26. MAXIFE
27. MMMU
28. MMMU-Pro
29. MathVision
30. Mathvista(mini)
31. We-Math
32. DynaMath
33. ZEROBench
34. ZEROBench_sub
35. VlmsAreBlind
36. BabyVision
37. RealWorldQA
38. MMStar
39. MMBench (EN-DEV-v1.1)
40. SimpleVQA
41. HallusionBench
42. OmniDocBench1.5
43. CharXiv(RQ)
44. MMLongBench-Doc
45. CC-OCR
46. AI2D_TEST
47. OCRBench
48. ERQA
49. CountBench
50. RefCOCO(avg)
51. EmbSpatialBench
52. RefSpatialBench
53. LingoQA
54. Hypersim
55. Nuscene
56. VideoMME (w sub.)
57. VideoMME (w/o sub.)
58. VideoMMMU
59. MLVU
60. MVBench
61. LVBench
62. MMVU
63. ScreenSpot Pro
64. OSWorld-Verified
65. AndroidWorld
66. TIR-Bench
67. V*
68. SLAKE
69. PMC-VQA
70. MedXpertQA-MM

The TAU2 result follows the official setup except for an airline-domain fix.
MathVision uses a fixed boxed-answer prompt for Qwen; comparator scores select
the better result with or without boxed formatting. No uniform shot-count or
metric specification is given for the full table.

## Nanbeige/Nanbeige4.2-3B

Sources: [general/agentic table](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/5d54321e9e01e0d026f8e371046678fc384dca39/README.md#L45-L109), [local-assistant table](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/5d54321e9e01e0d026f8e371046678fc384dca39/README.md#L114-L147)

25 scored row occurrences, 23 distinct labels/configurations:

1. GDPval rubrics
2. Agent-IF-Oneday (in-house scaffold)
3. Office-QA-Pro
4. Pinch-Bench-V2
5. Claw-Gym
6. Claw-Eval (pass^3)
7. MCP-Atlas
8. SWE-Bench Verified
9. SWE-Bench Pro
10. Terminal-Bench 2.0
11. HLE w/o Search
12. SciCode
13. GPQA-Diamond
14. HMMT-Feb-2026
15. IMO-Answer-Bench
16. LiveCodeBench-V6
17. AA-LCR
18. IF-Bench
19. Recruit-Bench
20. GDPval (OpenClaw)
21. Agent-IF-Oneday (OpenClaw)
22. DeepResearch Bench II
23. ResearchRubrics

Pinch-Bench-V2 and Claw-Gym are repeated unchanged in the second table;
Agent-IF-Oneday is repeated under a different scaffold and score. All README
evaluations are stated to use thinking mode with `preserve_thinking=true`.
Hugging Face's generated metadata duplicates seven rows and conflicts with the
README on HMMT-Feb-2026: 82.1 in YAML versus 82.8 in the README.

## microsoft/Phi-4-mini-instruct

Sources: [scored table](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/README.md#L83), [evaluated-datasets appendix](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/README.md#L328-L374)

17 scored rows:

1. Arena Hard
2. BigBench Hard (0-shot, CoT)
3. MMLU (5-shot)
4. MMLU-Pro (0-shot, CoT)
5. ARC Challenge (10-shot)
6. BoolQ (2-shot)
7. GPQA (0-shot, CoT)
8. HellaSwag (5-shot)
9. OpenBookQA (10-shot)
10. PIQA (5-shot)
11. Social IQA (5-shot)
12. TruthfulQA (MC2) (10-shot)
13. Winogrande (5-shot)
14. Multilingual MMLU (5-shot)
15. MGSM (0-shot, CoT)
16. GSM8K (8-shot, CoT)
17. MATH (0-shot, CoT)

Additional public named evals in the appendix (no result published there):
MedQA, ANLI, Berkeley function calling, TriviaQA, GSM8K Hard, HumanEval,
HumanEval+, MBPP, MBPP+, LiveCodeBenh (source typo), LiveBench, BigCode Bench,
Spider, IFEval, MEGA, AGI Eval, DecodingTrust, XSTest, and Toxigen. The appendix
also repeats several scored-table evals and mentions unnamed internal function
calling, coding, instruction-following, multilingual-safety, multi-turn, and red
team evaluations; those unnamed/internal sources are not implementable from the
card alone.

Important mismatch: the card reports TruthfulQA **MC2**, while openbench2's
current task is explicitly **MC1**.

## google/gemma-4-E4B-it

Source: [revision ee0ef60](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md#L84-L113)

17 scored rows:

1. MMLU Pro
2. AIME 2026 no tools
3. LiveCodeBench v6
4. Codeforces ELO
5. GPQA Diamond
6. Tau2 (average over 3)
7. HLE no tools
8. HLE with search
9. BigBench Extra Hard
10. MMMLU
11. MMMU Pro
12. OmniDocBench 1.5 (average edit distance, lower is better)
13. MATH-Vision
14. MedXPertQA MM
15. CoVoST
16. FLEURS (lower is better)
17. MRCR v2 8 needle 128k (average)

The card names no shots. It explicitly gives ELO for Codeforces and average edit
distance for OmniDocBench; other underlying metric names should not be inferred.
The later safety section names content categories but no safety benchmark.
