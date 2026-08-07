# Codeforces ELO audit

## Decision

`Codeforces ELO` is recorded as unsupported because the cited Gemma 4 artifact
publishes final rating values rather than a fixed benchmark protocol. No task
window, contest/problem manifest, submission policy, judge environment, rating
calculation, or canonical evaluation repository is linked from the model card.

## Source evidence

- Model card: `google/gemma-4-E4B-it`
- Model-card revision inspected:
  `ee0ef6023621cff504d758262d4e04895a5af4a2`
- The benchmark table contains one `Codeforces ELO` row with final values for six
  Gemma variants.
- The card contains no other `Codeforces ELO` occurrence and supplies no
  methodology note or source link for that row.
- Gemma 4 Technical Report `arXiv:2607.02770v2` repeats the rating row but adds
  no Codeforces methods section, task manifest, or evaluator reference.

An exact-name GitHub search found downstream catalog/ranking references, but no
repository identified as the Gemma 4 evaluator. LiveOIBench and similarly named
competitive-programming evaluations are distinct protocols and cannot be used
as silent substitutes.

## Missing reproducibility contract

ELO is a derived rating, not a dataset-level metric. Reproducing it requires at
least:

- an immutable problem/contest/date window;
- language, compiler, time, memory, and submission limits;
- prompt and code-extraction rules;
- sample count and generation settings;
- the online/offline judge and hidden test assets;
- the opponent/reference population and exact rating update formula.

Without those pieces, an OpenBench `codeforces_elo` ID would assign the same name
to an independently invented evaluation whose score is not comparable to the
published row. The candidate remains blocked pending a canonical protocol from
the benchmark publisher.
