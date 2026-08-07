# Wave 2 benchmark integration contract

## Goal

Integrate OJBench, TIR-Bench, Codeforces ELO, LiveBench, BigCodeBench, Spider,
VITA-Bench, and MEGA into OpenBench without substituting similarly named data or
approximating metrics that require unavailable model/provider capabilities.

## Candidates

The candidate list is frozen to the eight benchmark identities above. Each is
treated as a separate integration candidate; no benchmark may borrow another's
score or dataset identity.

## Task matrix

For every candidate:

1. identify the canonical repository, release/dataset revision, license, prompt,
   sampling settings, and scorer;
2. record an immutable revision and checksums where practical;
3. implement the complete public evaluation protocol or mark the candidate
   unsupported with evidence;
4. add registry metadata, unit tests, and a real dataset/task construction smoke;
5. reuse a hardened Docker boundary whenever generated code, SQL, shell, or other
   untrusted actions execute;
6. run global lint, typing, unit tests, package build, and applicable Docker tests.

## Metrics and fairness

The primary integration metric is protocol completeness: `1` only when the
canonical public protocol is runnable and tested, otherwise `0`. Secondary
evidence records logical case count, task coverage, resource boundary, and known
historical-score limitations. Every candidate receives the same source audit and
validation gates; failures remain in `matrix.tsv` rather than being dropped.

## Environment

- Baseline commit: `7f11867` (`main`, after BFCL live merge)
- Python: project `.venv`, managed by UV
- Execution host: macOS/Docker Desktop; Linux execution images are digest-pinned
- Upstream network artifacts may drift unless an immutable revision and digest
  are recorded

## Baseline

At contract creation, none of the eight benchmark IDs exists in `src/` or the
registry. The repository baseline passes 446 unit tests with two documented
environment skips and four Docker sandbox integration tests.

## Stopping condition

Stop only when every candidate is either integrated and validated or explicitly
blocked with source-backed evidence, then commit, open a pull request, and follow
CI to completion. Model runs requiring unavailable paid credentials are logged as
blocked and never represented as benchmark scores.
