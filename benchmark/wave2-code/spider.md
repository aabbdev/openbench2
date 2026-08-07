# Spider audit

## Decision

Spider is recorded as unsupported for this OpenBench wave. The public data and
official scorer are available, but the canonical sources do not define a full
LLM evaluation protocol: there is no official prompt template, schema rendering,
or sampling configuration for modern generative models. Adding one in OpenBench
would create a new Spider-like evaluation, not a faithful Spider benchmark.

## Canonical sources

- Spider repository: `https://github.com/taoyds/spider.git`
- Spider repository commit: `b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c`
- Test-suite repository: `https://github.com/taoyds/test-suite-sql-eval.git`
- Test-suite repository commit: `e97acc546ecbee8fa27fa8dbf025ef61493a876c`
- Licenses: Apache-2.0 for both repositories.

## Public data and scorer facts

- Hugging Face `spider` and `xlangai/spider` expose the development split with
  1,034 examples and fields such as `db_id`, `question`, and `query`.
- The Spider repository states that since November 2020 Spider uses test-suite
  accuracy as the official metric and points to `test-suite-sql-eval`.
- The test-suite README instructs users to download the database/test-suite
  archive from Google Drive and place it under `database/`.
- Local archive inspected: `/tmp/spider-drive.body`
  - SHA-256: `9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96`
  - Compressed size: about 1.2 GB
  - Zip entries: 3,999
  - SQLite files: 3,889
  - Uncompressed bytes: 5,155,457,198
- The checked-out `test-suite-sql-eval/database/` directory contains only a
  README; it does not include the SQLite suites.

## Official metric

The official command shape is:

```text
python3 evaluation.py --gold [gold file] --pred [predicted file] --etype exec --db [database dir] --table [table file] --plug_value --keep_distinct --progress_bar_for_each_datapoint
```

The `exec` metric compares predicted SQL and gold SQL denotations over all
SQLite databases in each test-suite database directory. This is materially
different from ordinary single-database execution accuracy and cannot be
approximated by evaluating only the original Spider database.

## OpenBench compatibility finding

A faithful OpenBench task would need all of the following frozen by the
canonical benchmark:

- how to render each database schema to the model;
- the exact natural-language prompt around the user question;
- whether to include table contents or sampled rows;
- whether models predict SQL values directly or rely on `--plug_value`;
- generation temperature, stop sequences, and max tokens;
- how to package or require the 5.15 GB test-suite databases.

The official repositories define the data and scorer but not those generation
choices. OpenBench therefore records Spider as blocked rather than shipping a
hand-prompted variant whose scores would not be comparable to the official
leaderboard protocol.
