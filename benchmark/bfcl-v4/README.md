# BFCL v4 integration contract

OpenBench exposes four independently named BFCL v4 sections and one offline
aggregate:

- `bfcl_v4_single_turn`: all 13 official single-turn categories;
- `bfcl_v4_multi_turn`: all four official stateful categories;
- `bfcl_v4_agentic_offline`: three memory backends and two frozen web-search
  configurations;
- `bfcl_v4_agentic_live`: BFCL's model-generated memory prerequisites and live
  SERPAPI web backends;
- `bfcl_v4_offline`: all 5,106 samples with BFCL's 10/10/10/30/40 weights.

## Provenance

- Upstream: `ShishirPatil/gorilla`
- Revision: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- License: Apache-2.0
- Every downloaded question and answer file is protected by a pinned SHA-256.

## Included categories

- Non-live: Python, Java, JavaScript, multiple, parallel,
  parallel-multiple, and irrelevance.
- Live: simple, multiple, parallel, parallel-multiple, irrelevance, and
  relevance.

The integration preserves BFCL's category aggregation: simple non-live is a
macro-average across languages, non-live is a macro-average across task shapes,
live is sample-weighted, and irrelevance is averaged across live/non-live.

## Common function-calling layer

The implementation supplies reusable primitives for:

- JSON Schema to provider-safe Inspect tool definitions;
- reversible normalization of names such as `weather.get`;
- provider-native, JSON, and Python-style call parsing without `eval`;
- exact type/value validation, optional arguments, nested containers, and
  one-to-one parallel matching;
- per-category and official-style aggregate metrics.

## Explicit boundary

The multi-turn task runs BFCL's pinned official state and response checker in a
network-disabled, read-only Docker sandbox. It covers base, missing-function,
missing-parameter, and long-context categories.

The offline agentic task is intentionally an adaptation. Memory retrieval is
initialized from the public BFCL source record, while web search uses a frozen
corpus derived from BFCL's cited evidence. The no-snippet mode exposes URLs only
until the model calls `fetch_url_content`. This removes SERPAPI credentials and
web drift from CI, but it is not numerically interchangeable with the live
leaderboard environment.

The separate `bfcl_v4_agentic_live` task preserves the official agentic
protocol:

- each model generates all 37 prerequisite conversations once for each of the
  three memory backends;
- the resulting KV, vector, or recursive-summary state is reused for all 155
  questions without leaking one target conversation into another;
- `all-MiniLM-L6-v2` is pinned at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and all Python dependencies are
  locked;
- the 100 web questions run in both snippet and no-snippet modes against
  BFCL's DuckDuckGo SerpAPI backend.

The live task contains five workflow samples but reports 665 logical case
outcomes. It requires `SERPAPI_API_KEY` for web categories. A run can record
content-addressable, model-trace-specific web evidence with
`-T record_web_snapshot_dir=PATH`, then replay that exact trace without network
using `-T web_snapshot_dir=PATH`. Replay fails closed if a model changes any
tool call or step count; a recorded trace is therefore an audit artifact, not a
universal replacement for live search.

Consequently, `bfcl_v4_offline` applies the official section weights but does
not claim the official leaderboard score. The memory prerequisite pipeline and
live-search transport now exist, but the unqualified `bfcl_v4` alias remains
reserved until two credentialed provider runs and differential score comparison
against the upstream harness demonstrate parity.

The older “Berkeley Function Calling” label in the Phi model card is not aliased
to this task: without a harness/version citation it may refer to BFCL v1 rather
than the v4 composition.
