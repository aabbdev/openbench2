# BFCL v4 live parity decision

## Implemented

- Upstream code and datasets are pinned to
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
- All five memory prerequisite conversations are checksum-verified.
- Prerequisite calls are generated once per model, scenario, and backend.
- Target questions start from an isolated replay of the prerequisite state.
- The official KV, vector, recursive-summary, SerpAPI, URL-fetch, and answer
  checker implementations run inside a dedicated Docker boundary.
- The vector encoder and the complete Linux dependency graph are pinned.
- Live web traces can be recorded and replayed content-exactly with networking
  disabled.

## Evidence

- 155 logical cases load for each memory backend and 100 for each web mode.
- KV, vector, and recursive-summary backends were executed in the live image;
  each produced the official state-derived system prompt.
- A persistent KV prerequisite followed by a target retrieval scored 1.0 in an
  end-to-end Inspect run.
- A recorded web-search trace replayed in the network-disabled image and scored
  1.0; mismatched calls and unused steps fail closed.

## Alias decision

Do not expose `bfcl_v4` yet. No real provider credentials were available in the
execution environment, so the required two-provider transport run and
model-output differential against the upstream harness could not be performed.
In addition, live SerpAPI results and fetched pages are intrinsically mutable;
recorded snapshots prove a particular run but are not a provider-independent
official environment.

The supported live component is named `bfcl_v4_agentic_live`. The existing
`bfcl_v4_offline` remains the reproducible full-coverage diagnostic aggregate.
