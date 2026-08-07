# TIR-Bench audit

## Decision

TIR-Bench now has a public canonical repository and dataset, so the earlier
"source unavailable" assessment is obsolete. It remains unsupported in this
OpenBench wave because the released code scores pre-generated response files;
it does not publish the agentic image-manipulation harness needed to reproduce
the reported with-CI/without-CI model runs. Its answer extraction also requires
a separately configured GPT-4.1 judge.

## Canonical sources

- Repository: `https://github.com/agents-x-project/TIR-Bench`
- Repository commit inspected:
  `f79c7562b59e4f8142b0437fc725eb3ee1aec76c`
- Dataset: `Agents-X/TIR-Bench`
- Dataset revision: `ae9976e81e86c4797fd0d28f9119393b37d9d006`
- Dataset license: Apache-2.0.
- Published scope: 1,215 examples across 13 image-reasoning tasks.

The repository itself does not include a license file. The Apache-2.0 label
above comes from the immutable Hugging Face dataset metadata and should not be
assumed to license the repository code.

## Released evaluation path

The repository provides two post-generation stages:

1. `extract_answer.py` reads a JSON result file, asserts that it contains 1,215
   entries, and calls an OpenAI-compatible GPT-4.1 endpoint with task-specific
   few-shot extraction prompts.
2. `calculate_score.py` applies deterministic task-specific comparisons to the
   extracted answers, including exact choice/integer/float checks, list IoU,
   jigsaw-position accuracy, OCR substring checks, Levenshtein normalization,
   and `math_verify` fallbacks.

The scripts expect model responses to exist already. They do not define how the
evaluated model receives images, creates or invokes image-processing tools,
iterates over intermediate images, limits tool calls, or converts that trajectory
into the final response. The Qwen3.5 model card reports TIR-Bench as "with CI /
without CI", but neither that card nor this repository defines a reproducible CI
runtime.

## OpenBench compatibility finding

A faithful integration needs two missing protocol components: the canonical
agent/tool generation harness and explicit authority/configuration for the
GPT-4.1 extraction judge. A static VQA prompt would measure a different task,
while substituting a hand-authored deterministic extractor would change the
published metric. OpenBench therefore records the public data and scorer but
does not expose a `tir_bench` task until those choices can be reproduced or made
explicit in the benchmark identity.
