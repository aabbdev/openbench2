# BigCodeBench audit and integration note

## Decision

BigCodeBench now has an OpenBench adapter and an arm64 source-equivalent scorer
image validated against all 148 canonical Hard records on Apple Silicon. It
passes 145 records under the hardened no-network sandbox; the remaining three
require live external downloads and therefore fail closed instead of being
counted as model errors. The pinned official evaluator image is still
`linux/amd64` only, so official-image parity remains blocked until it is
smoke-tested on a usable native `linux/amd64` Docker host.

## Canonical sources

- Repository: `https://github.com/bigcode-project/bigcodebench.git`
- Repository commit inspected: `09dd993f46c3fbf3a799465bb96d524edcb0b199`
- Official package/data version used by upstream loader: `v0.1.4`
- Full dataset: `bigcode/bigcodebench`, revision
  `b74c0d0bf70d2c0bc459be537895cca163007f1a`, 1,140 tasks.
- Hard dataset: `bigcode/bigcodebench-hard`, revision
  `298d2cc7b96612e15e47313c3603ee124cee0c1f`, 148 tasks.
- Official evaluator image: `bigcodebench/bigcodebench-evaluate` manifest
  `sha256:1327bddf60be9bc241648c59e6060cac4ca50248a0588ab735cd0200b17cc8c2`
  for `linux/amd64`.
- License: Apache-2.0.

## Implemented OpenBench surface

- Registry ID: `bigcodebench`.
- Parameters: `split="complete" | "instruct"`, `subset="full" | "hard"`,
  `runtime="auto" | "official" | "arm64"`, optional `limit`, `epochs`, and
  `total_timeout`. The source-equivalent arm64 runtime currently accepts only
  `subset="hard"`; the unvalidated full subset fails at task construction.
- Dataset loader: immutable Hugging Face revisions, with hidden execution fields
  resolved only at score time.
- Prompting: follows BigCodeBench's OpenAI/API chat backend wrapper by applying
  the official instruction prefix and using the official complete-vs-instruct
  prompt field.
- Scoring: sandbox runner uses BigCodeBench's own `sanitize`, `trusted_check`,
  and `untrusted_check`. Canonical solution timing is computed per task and used
  to calibrate the generated solution timeout, matching the upstream evaluator.
  If canonical calibration fails, the scorer raises an evaluation error rather
  than silently assigning an incorrect model score.
- Docker: `runtime="official"` pins the official `linux/amd64` evaluator image;
  `runtime="arm64"` builds OpenBench's source-equivalent scorer image from the
  pinned upstream source commit for Apple Silicon; `runtime="auto"` selects
  arm64 on arm64/aarch64 hosts. Both compose files disable network, drop
  capabilities, set no-new-privileges, and use tmpfs work directories.

## Local validation

Passed locally:

```text
source .venv/bin/activate && ruff check src/openbench/datasets/bigcodebench.py src/openbench/scorers/bigcodebench.py src/openbench/scorers/bigcodebench_runner.py src/openbench/evals/bigcodebench tests/test_bigcodebench.py tests/test_registry.py
All checks passed!

source .venv/bin/activate && pytest tests/test_bigcodebench.py tests/test_registry.py
23 passed

source .venv/bin/activate && mypy src/openbench/datasets/bigcodebench.py src/openbench/scorers/bigcodebench.py src/openbench/scorers/bigcodebench_runner.py src/openbench/evals/bigcodebench tests/test_bigcodebench.py
Success: no issues found in 6 source files
```

Also passed dataset construction smoke for `bigcodebench(limit=1)` and the hard
subset loader without printing benchmark prompts.

Arm64 source-equivalent smoke passed locally:

```text
docker build --platform linux/arm64 -f src/openbench/evals/bigcodebench/Dockerfile.arm64 -t openbench-bigcodebench-arm64:dev src/openbench/evals/bigcodebench

docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,uid=1000,gid=1000,mode=0700,size=4294967296 --cap-drop ALL --security-opt no-new-privileges:true --pids-limit 256 --entrypoint python3 -v "$smoke_dir:/app:rw" openbench-bigcodebench-arm64:dev /app/runner.py /app/payload.json
BigCodeBench/13
{"passed": true, "status": "pass", ...}

docker compose -f src/openbench/evals/bigcodebench/compose.arm64.yaml -p openbench-bcb-arm64-smoke up -d --build
docker compose -f src/openbench/evals/bigcodebench/compose.arm64.yaml -p openbench-bcb-arm64-smoke exec -T default python3 -c "from bigcodebench.eval import untrusted_check; from bigcodebench.sanitize import sanitize; print('compose-imports-ok')"
compose-imports-ok
```

The arm64 image intentionally omits BigCodeBench's generation-only API clients
and vLLM dependency because OpenBench generates responses outside the scorer
container. Its evaluator dependencies are fully resolved in
`requirements-arm64.lock`; the Python base image, BigCodeBench source commit,
toolchain, direct/transitive Python packages, and NLTK asset hashes are pinned.

## Canonical Hard preflight

The source-equivalent image was iterated against all 148 canonical Hard records.
The machine-readable experiment log remains outside Git at
`/tmp/openbench-bcb-arm64-validation/results.tsv`; prompts, tests, and canonical
solutions are not included in the log.

| Image state | Canonical pass rate | Wall time |
| --- | ---: | ---: |
| Minimal scorer baseline | 79/148 (53.38%) | 170.22 s |
| First dependency layer | 133/148 (89.86%) | 607.98 s |
| Second dependency/NLTK layer | 143/148 (96.62%) | 422.87 s |
| Restored legacy imports | 144/148 (97.30%) | 437.43 s |
| Hardened tmpfs caches | 145/148 (97.97%) | 500.89 s |

The final three failures are `BigCodeBench/101`, `BigCodeBench/590`, and
`BigCodeBench/1012`. Their canonical solutions access live resources hosted by
CMU, Wikibooks, Google Drive, or learningcontainer. OpenBench keeps networking
disabled for untrusted generated programs and does not replace these resources
with invented fixtures. These samples therefore produce a canonical validation
error and no model score.

Blocked locally:

```text
docker run --platform linux/amd64 ... bigcodebench/bigcodebench-evaluate@sha256:1327bddf60be9bc241648c59e6060cac4ca50248a0588ab735cd0200b17cc8c2 ...
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
```

The host is `arm64` and Docker reports `linux/arm64`; the official evaluator is
only available as `linux/amd64`. Even a synthetic tiny runner payload segfaulted
under QEMU, so repeated local smoke attempts were stopped to avoid freezes.

An x86_64 GPU instance (`ghub5090`) was also tried after explicit user approval.
Docker was installed and started with bridge/iptables disabled, then retried with
the `vfs` storage driver because the instance appears containerized. The daemon
could start, but pulling/registering the official image failed with:

```text
failed to register layer: unshare: operation not permitted
```

Temporary Docker data and the smoke payload were removed from the GPU instance
after the failed attempt. The `docker.io` package remains installed there because
installation was explicitly approved for this validation attempt.

## Remaining requirement

For strict upstream-image parity, run the official Docker scorer smoke on a
native `linux/amd64` host with Docker privileges sufficient for image extraction
and container creation. A faithful full 148-task hardened run additionally needs
immutable, audited fixtures for the three live-network records; enabling network
for generated programs is not an acceptable fallback. Until both requirements
are satisfied, BigCodeBench should be described as arm64 Hard 145/148 validated,
not as full official-image parity.
