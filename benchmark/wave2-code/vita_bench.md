# VITA-Bench audit

## Decision

VitaBench 1.0 is actionable as an explicit live benchmark candidate. The
complete simulated environment, 400 tasks, user simulator, prompts, and
trajectory evaluator are public and pinnable. It is not a single-model offline
task: a faithful run requires the model under test, a fixed auxiliary user LLM,
and a fixed auxiliary evaluator LLM. OpenBench should therefore prototype
`vitabench_v1_live` and withhold the unqualified alias until credentialed
differential runs against the upstream CLI pass.

## Canonical sources

- Repository: `https://github.com/meituan-longcat/vitabench.git`
- Repository commit: `742e240855bf8686a0842360749d5ea970ea3987`
- Dataset: `https://huggingface.co/datasets/meituan-longcat/VitaBench`
- Dataset revision: `8be56c8ca02d3d15cd3e8d27cc9162bc58502f01`
- Paper: `https://arxiv.org/abs/2509.26490v2`
- License: MIT

## Public task assets

The official repository includes Chinese and English variants of four task
manifests under `data/vita/domains/`:

| Domain/language | Tasks | SHA-256 | Bytes |
| --- | ---: | --- | ---: |
| `cross_domain` Chinese | 100 | `a7e782785f40667511a54562d613e2122de645a36b8661a5658c1740d34a5815` | 10,318,151 |
| `cross_domain` English | 100 | `54a396cd2e5de2a26da29b954341ad190806ed7f2e870dd96511e81a947c2865` | 7,050,913 |
| `delivery` Chinese | 100 | `22746ae23eb6dbed2ee6a5e3a0f476af60e8a43bab0ea2d3ead54f0ef3829f94` | 2,572,829 |
| `delivery` English | 100 | `4906517e50ad578cdfd52426b76e4604bfd207db4254643bbc150891240f5500` | 2,081,689 |
| `instore` Chinese | 100 | `f92b9313e5476499d51b73a929cc901b7d87cd7c3315df56942d62c83aa34407` | 5,290,647 |
| `instore` English | 100 | `8e3ac7b53501d211a5261ef4e8ad9eb46bd304bd6a77794dea860f107aebaa95` | 3,828,442 |
| `ota` Chinese | 100 | `c9208c25accef24f4798bbd3f38403da081fdf48f3c7100dfc702aa7b379559e` | 8,732,896 |
| `ota` English | 100 | `f04ac51d84452557f4a68b6cc25f854edc88b0a501c85e48943859be7ab203b9` | 5,883,455 |

The audit intentionally does not inline task instructions, user scenarios, or
judge prompts. They are benchmark inputs in the canonical files above, and
printing them into OpenBench documentation is unnecessary for reproducibility.

## Official run settings

Observed from the paper, `src/vita/config.py`, `src/vita/cli.py`, and
`src/vita/run.py`:

- Default domain: `delivery,instore,ota` for cross-domain evaluation.
- Default agent implementation: `llm_agent`.
- Default user implementation: `user_simulator`.
- Default agent LLM: `gpt-4.1`.
- Main-results user-simulator LLM: `gpt-4.1-2025-04-14`; the repository uses
  the shorter configurable name `gpt-4.1`.
- Default evaluator LLM: `anthropic.claude-3.7-sonnet`.
- Default evaluation type: `trajectory`.
- Default maximum steps: 300.
- Default maximum consecutive errors: 10.
- CLI default trials: 1; paper main results: 4 trials per task.
- Main-results temperature: 0.0 for all model roles.
- Default seed: 300.
- Default language: `chinese`.

`models.yaml` supplies provider base URLs, headers, token settings, and model
costs. The official runner reads that configuration and calls external model
APIs directly.

The current canonical repository is newer than the frozen
`meituan/vitabench` mirror previously audited. Its January 2026 update rectifies
tasks and tools, adds English manifests, changes run ordering, and updates
reasoning-content handling. The two revisions must not be mixed in one score.

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

## OpenBench implementation path

The upstream environment is local Python state, not a browser or external
website snapshot. Its tools use OpenAI function schemas and do not execute
model-generated code, so the next implementation slice is bounded:

1. pin current source commit `742e240` and lock its transitive runtime;
2. adapt the target-agent calls to Inspect's evaluated model while keeping the
   upstream orchestrator, task state, prompts, and tool semantics unchanged;
3. require explicit auxiliary configurations for `gpt-4.1-2025-04-14` and
   `claude-3.7-sonnet`, failing closed when either is absent;
4. expose four epochs and preserve the upstream all-rubrics binary reward;
5. differential-test a small credentialed set against the pinned upstream CLI.

VitaBench 2.0 is a separate long-term personalization benchmark and must use a
separate identity; it is not a protocol update that can silently replace 1.0.

## Local source smoke

The pinned current source installed successfully into an isolated Python 3.13
environment through `uv`, resolving 110 packages. Without model credentials,
its official loader imported every Chinese and English manifest: 100 unique
tasks in each of delivery, in-store, OTA, and cross-domain for both languages.
The first cross-domain task in each language also constructed successfully with
all 66 unique tools. This closes the source/runtime discovery blocker;
credentialed auxiliary-model parity remains the implementation gate.
