"""Preflight OJBench problem configurations inside the DMOJ scorer image.

The payload contains only problem identifiers and a language. This diagnostic
uses a deliberately wrong submission, stops execution after the first failed
case, and emits only identifiers, verdicts, counts, error types, and timing.
Prompts, test contents, per-case output, and judge feedback are never printed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import ojbench  # type: ignore[import-not-found]
from ojbench.judger import judge  # type: ignore[import-not-found]

RESULT_PREFIX = "OPENBENCH_RESULT\t"
SUMMARY_PREFIX = "OPENBENCH_SUMMARY\t"
SOURCES = {
    "python": ("PYPY3", "print(0)"),
    "cpp": ("CPP17", "#include <iostream>\nint main(){std::cout << 0;}"),
}


def validate_problem(problem_id: str, language: str) -> dict[str, Any]:
    """Load one problem and execute a fast sentinel submission."""

    runtime, source = SOURCES[language]
    started = time.monotonic()
    try:
        verdict, results = judge(
            problem_id=problem_id,
            time_limit=10,
            memory_limit=1024 * 1024,
            language=runtime,
            source=source,
            stop_when_fail=True,
            use_tqdm=False,
        )
        return {
            "problem_id": problem_id,
            "language": language,
            "verdict": verdict,
            "tests": len(results),
            "wall_seconds": round(time.monotonic() - started, 6),
            "error_type": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return {
            "problem_id": problem_id,
            "language": language,
            "verdict": None,
            "tests": 0,
            "wall_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
        }


def main() -> None:
    """Validate every requested problem after one DMOJ initialization."""

    payload = json.loads(Path(sys.argv[1]).read_text())
    language = str(payload["language"])
    if language not in SOURCES:
        raise ValueError("language must be 'python' or 'cpp'")
    problem_ids = [str(problem_id) for problem_id in payload["problem_ids"]]

    ojbench.init(
        problem_dirs=[Path("/problems/NOI"), Path("/problems/ICPC")],
        config_path=Path(os.environ["OJBENCH_CONFIG_PATH"]),
        runtime_path=Path(os.environ["OJBENCH_RUNTIME_PATH"]),
        compile_lock_path=Path("/workspace/compile.lock"),
    )

    started = time.monotonic()
    infrastructure_errors = 0
    for problem_id in problem_ids:
        result = validate_problem(problem_id, language)
        if result["error_type"] is not None or result["verdict"] in {"IE", "Skip"}:
            infrastructure_errors += 1
        print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)

    summary = {
        "language": language,
        "problems": len(problem_ids),
        "infrastructure_errors": infrastructure_errors,
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    print(f"{SUMMARY_PREFIX}{json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
