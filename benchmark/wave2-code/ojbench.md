# OJBench audit

## Decision

OJBench is recorded as unsupported for this OpenBench wave. Unlike Spider and
MEGA, OJBench does publish a full prompt file for LLM generation, but faithful
scoring depends on a DMOJ-based online-judge runtime that has not been validated
inside OpenBench's hardened Docker policy.

## Canonical sources

- Repository: `https://github.com/He-Ren/OJBench.git`
- Repository commit: `5e94480b1e135b98855cf5bc81213c256aff5b17`
- Test data: `https://huggingface.co/datasets/He-Ren/OJBench_testdata`
- Test data HEAD: `61cf9986f22c25d08e1657b03742124099c74353`
- OJBench license: AGPL-3.0.

## Public prompt asset

The official test-data repository contains `prompts/full.jsonl`. It was read
without printing benchmark prompts.

- Rows: 464
- Problems: 232, each with Python and C++ variants
- SHA-256: `bcc8c94eb1fefb856355aa8b5a3e20cc0a2112f5436c5d83ab686edb417bce2c`
- Datasets: 318 NOI rows and 146 ICPC rows
- Languages: 232 Python rows and 232 C++ rows
- Difficulty labels: 72 easy, 158 medium, 234 hard
- Fields: `id`, `prompt`, `dataset`, `language`, `difficulty`

## Official judging requirements

The README requires:

- DMOJ judge-server, specifically checked out at
  `f098cd3a49a60186d1fadde5132329ec5f4f2213`;
- `dmoj==4.1.0`;
- C++17-compatible `g++`;
- `pypy3`;
- OJBench test data cloned through Git LFS;
- `ojbench.init()` pointed at both NOI and ICPC problem directories before
  calling `judge_jsonl`.

The judge reports full AC/WA/RE-style verdicts plus partial verdicts after 1/8,
1/4, and 1/2 of test cases. The main score is `is_passed`, equivalent to final
verdict `AC`.

## OpenCompass note

The OpenCompass adapter at
`opencompass/opencompass/datasets/ojbench.py` only loads `id` and `prompt` from a
JSONL file. It does not implement DMOJ setup, code extraction, test execution,
partial verdicts, or scoring, so it is not sufficient evidence for a faithful
OpenBench integration.

## OpenBench compatibility finding

OJBench should be integrated only once there is a validated Docker execution
boundary for DMOJ that preserves OpenBench's safety policy: network disabled,
capabilities dropped, no new privileges, bounded process/memory limits, and no
host compiler/runtime escape. That image also needs the large LFS problem zips
or a reproducible cache step. Until that exists, adding a registry ID would risk
either weakening the sandbox or reporting scores from an unvalidated judge. The
candidate is therefore blocked rather than approximated.
