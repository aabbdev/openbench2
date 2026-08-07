# VITA-Bench audit

## Decision

VITA-Bench is recorded as unsupported for this OpenBench wave. The public
benchmark is available, but the official protocol is not a single-model offline
task: it requires a model under test, a separate LLM user simulator, and a
separate LLM trajectory evaluator. OpenBench should not report a VITA score by
substituting a deterministic scorer or by silently choosing auxiliary LLMs.

## Canonical sources

- Repository: `https://github.com/meituan/vitabench.git`
- Repository commit: `973756f4754873474e2931a404f68093df9ef4e2`
- Dataset: `https://huggingface.co/datasets/meituan-longcat/VitaBench`
- Dataset HEAD: `5ca6848c215cdffd5ef9bc704ddcb62ed74696f0`
- License: MIT

## Public task assets

The official repository includes four task files under `data/vita/domains/`:

| Domain | Tasks | SHA-256 | Bytes |
| --- | ---: | --- | ---: |
| `cross_domain` | 100 | `3d662cd36efae511e256842e81d54b97f399152be9fdf26dc36ffe87cf0765bd` | 10,323,128 |
| `delivery` | 100 | `5a122f783d2b501f063c718b8dc9a637de573b90c9a87a7f9f4336b1ea9c9404` | 2,536,391 |
| `instore` | 100 | `f92b9313e5476499d51b73a929cc901b7d87cd7c3315df56942d62c83aa34407` | 5,290,647 |
| `ota` | 100 | `874e9117f94758a33ec565ab43ed53361bf09253e157b6f3174e0e5dbf32cbde` | 8,732,896 |

The audit intentionally does not inline task instructions, user scenarios, or
judge prompts. They are benchmark inputs in the canonical files above, and
printing them into OpenBench documentation is unnecessary for reproducibility.

## Official run settings

Observed from `src/vita/config.py`, `src/vita/cli.py`, and `src/vita/run.py`:

- Default domain: `delivery,instore,ota` for cross-domain evaluation.
- Default agent implementation: `llm_agent`.
- Default user implementation: `user_simulator`.
- Default agent LLM: `gpt-4.1`.
- Default user-simulator LLM: `gpt-4.1`.
- Default evaluator LLM: `anthropic.claude-3.7-sonnet`.
- Default evaluation type: `trajectory`.
- Default maximum steps: 300.
- Default maximum consecutive errors: 10.
- Default trials: 1.
- Default seed: 300.
- Default language: `chinese`.

`models.yaml` supplies provider base URLs, headers, token settings, and model
costs. The official runner reads that configuration and calls external model
APIs directly.

## Scoring protocol

All public evaluation modes route through `TrajectoryEvaluator` and call
`generate()` for the evaluator model:

- `trajectory`
- `trajectory_full_traj_rubric`
- `trajectory_sliding_wo_rubric`
- `trajectory_full_traj_wo_rubric`

The default `trajectory` evaluator uses a sliding window over the complete
conversation, keeps rubric state across windows, and returns a binary reward of
1 only when every natural-language rubric is met. That makes the judge model a
required part of the metric, not an implementation detail.

The simulator also calls `generate()` for the user role. Even the registered
`dummy_user` path is implemented as an LLM-backed user class, so it is not a
deterministic replacement for the official user simulator.

## OpenBench compatibility finding

OpenBench can support agentic tasks when the environment and scorer are
deterministic or explicitly parameterized, as with AgentDojo and tau-bench. A
faithful VITA-Bench integration would need new support for auxiliary LLM roles
and a configured judge model in addition to the evaluated model. Without that,
any OpenBench task would either omit the official user simulator, omit the
official judge, or hard-code extra paid model dependencies. The candidate is
therefore blocked rather than approximated.
