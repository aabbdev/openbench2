# MEGA audit

## Decision

MEGA is recorded as unsupported for this OpenBench wave. The public repository is
available, but the released artifact is a collection of experiment scripts and
notebooks for multilingual LLM evaluation rather than a single stable benchmark
protocol that OpenBench can run provider-agnostically.

## Canonical source

- Repository: `https://github.com/microsoft/Multilingual-Evaluation-of-Generative-AI-MEGA.git`
- Repository commit: `3e96bab146151942ed6a7bbe59c0364a78ebf94f`
- License: MIT
- Paper scope stated in README: 16 NLP datasets across 70 languages.

## Official execution shape

The README describes MEGA as a framework and documents XNLI as the concrete
example. The repository then supplies task-specific shell scripts and notebooks.
The current script inventory contains 21 `python -m mega...` invocations across
these modules:

- `mega.XLSUM`
- `mega.analysis.contamination`
- `mega.answer_cls`
- `mega.eval_pawsx`
- `mega.eval_qa_gptindex`
- `mega.eval_qa_gptturbo`
- `mega.eval_tag`
- `mega.eval_xcopa`
- `mega.eval_xnli`
- `mega.eval_xstory_cloze`

Those scripts encode per-task/per-language choices such as prompt names,
few-shot counts, model names, validation-vs-test switches, translation modes,
and metric output paths. There is no single main-results manifest that freezes
all 16 datasets, languages, prompts, splits, few-shot selections, and model
settings in one runnable protocol.

## External service requirements

The official setup requires API credentials in `keys/` for OpenAI and Bing
Translator. The code also imports environment variables at module import time,
including OpenAI endpoint settings, Hugging Face endpoint/key settings, and Bing
Translator endpoint/key settings.

`mega/models/completion_models.py` calls the legacy OpenAI completion/chat APIs
directly, sleeps for rate limiting, and supports a fixed model list including
Azure-style names such as `gpt-35-turbo`, `gpt-35-turbo-16k`, `gpt-4`, and
`gpt-4-32k`, plus BLOOM/BLOOMZ through Hugging Face endpoints. That is not the
same execution path as OpenBench's Inspect model abstraction.

## Dependencies and artifacts

The repository declares Python 3.7 compatibility and includes older unpinned or
pinned dependencies such as `transformers==4.30.0`, `langchain==0.0.317`,
`networkx==1.11`, `word2word==1.0.0`, `openai`, `datasets`, `evaluate`,
`torch`, and a vendored PromptSource tree with 331 template files.

The repository includes prior GPT-4 XLSUM artifacts under `gpt-4-all-lang-eval/`:
35 prediction CSV files and `xlsum_gpt_4_metrics.csv` with SHA-256
`a235258dc91cd5d219f8fde96376db1ee07f5de4a6dc0bb8ed9011477ef0b429`. These are
historical result artifacts, not an evaluation dataset for arbitrary models.

## OpenBench compatibility finding

A faithful OpenBench integration would need a new MEGA manifest first: exact
dataset revisions, language list per dataset, prompt template IDs, split policy,
few-shot selection policy, translation-test policy, model sampling settings, and
metric aggregation rules. It would also need a provider-agnostic rewrite of the
OpenAI/Azure/HF-specific generation layer. Without those pieces, exposing a
single `mega` registry ID would either cover only a hand-picked subset or report
scores that are not comparable to the official MEGA experiments. The candidate
is therefore blocked rather than approximated.
