# OJBench audit

## Decision

OJBench now has alpha `ojbench_python` and `ojbench_cpp` registry IDs backed by
the official DMOJ judge. Each language track contains 232 problems and uses
eight samples per problem, matching the paper's Pass@1/Pass@8 protocol. The
paper uses each model's recommended sampling parameters rather than one global
temperature/top-p configuration, so OpenBench deliberately leaves those model
settings configurable.

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

## OpenBench implementation

- Prompts are checksum-verified and loaded from the pinned Hugging Face
  revision.
- Test archives and custom validators are downloaded lazily per problem into a
  host cache mounted read-only at `/problems`. A limited run therefore does not
  require downloading the complete 7.85 GB source repository.
- The image uses digest-pinned Python 3.11 Bookworm (GCC 12) and installs
  OJBench commit
  `5e94480b1e135b98855cf5bc81213c256aff5b17` and DMOJ commit
  `f098cd3a49a60186d1fadde5132329ec5f4f2213`, with exact Python dependencies,
  C++17 `g++`, and PyPy3.
- The scorer returns only the final verdict, number of executed cases, and
  partial pass booleans. Per-case inputs, outputs, and feedback never enter
  Inspect logs.

The container runs without networking, with a read-only root filesystem,
`no-new-privileges`, bounded memory/PIDs, and `cap_drop: ALL`. DMOJ cptbox
requires `SYS_PTRACE` to supervise its own child process, so that single
capability is restored explicitly. Generated programs remain under DMOJ's
seccomp and filesystem policies; `/problems` is not in their readable policy.

## Validation and remaining gate

On local arm64 Docker, both CPP17 and PYPY3 executor self-tests passed. A
synthetic one-case problem exercised AC, WA, CE, RTE/IR, TLE, and memory-pressure
paths in both runtimes. A submission attempting to read the mounted hidden-test
`init.yml` returned DMOJ `IR`, confirming the anti-oracle filesystem boundary.

The complete pinned corpus was then downloaded: 232 problem directories and
232 archives totalling 7,505,819,661 bytes. The preflight script evaluated a
fast sentinel submission against every configuration with early stop after the
first failure:

- Python/PYPY3: 232/232 normal `WA` verdicts, zero infrastructure errors,
  137.26 seconds wall time;
- C++17: 232/232 normal `WA` verdicts, zero infrastructure errors, 247.54
  seconds wall time;
- all 17 problems using custom checkers, interactors, or output validators
  compiled and returned normal verdicts.

The first image used Debian Trixie/GCC 14 and exposed compilation failures in
two upstream `testlib.h` validators. Switching to digest-pinned Bookworm/GCC 12
restored all official validators without modifying benchmark artifacts.

A credential-free Pass@1/Pass@8 run with a local pinned 1.5B model is the final
active alpha gate. Leaderboard-quality multi-provider runs remain necessary
before treating the resulting model score as broadly comparable.
