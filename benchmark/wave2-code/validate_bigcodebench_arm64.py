"""Preflight canonical BigCodeBench records inside the arm64 scorer image.

This script intentionally consumes a generated payload file rather than loading
the dataset in the sandbox. It emits only task identifiers, status, and timing;
prompts, tests, and canonical solutions remain in the local payload file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from bigcodebench.gen.util import trusted_check

RESULT_PREFIX = "OPENBENCH_RESULT\t"
SUMMARY_PREFIX = "OPENBENCH_SUMMARY\t"


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Execute one canonical solution with the pinned upstream checker."""

    started = time.monotonic()
    try:
        result = trusted_check(
            record["complete_prompt"] + "\n" + record["canonical_solution"],
            record["test"],
            record["task_id"],
            record["max_as_limit"],
            record["max_data_limit"],
            record["max_stack_limit"],
            record["min_time_limit"],
        )
        canonical_time = result["time"]
        return {
            "task_id": record["task_id"],
            "passed": canonical_time is not None,
            "canonical_time": canonical_time,
            "wall_seconds": round(time.monotonic() - started, 6),
            "error_type": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return {
            "task_id": record["task_id"],
            "passed": False,
            "canonical_time": None,
            "wall_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
        }


def main() -> None:
    """Validate every payload row and emit a machine-readable summary."""

    payload_path = Path(sys.argv[1])
    records = json.loads(payload_path.read_text())
    started = time.monotonic()
    passed = 0

    for record in records:
        result = validate_record(record)
        passed += int(result["passed"])
        print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)

    summary = {
        "passed": passed,
        "total": len(records),
        "pass_rate": passed / len(records) if records else 0.0,
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    print(f"{SUMMARY_PREFIX}{json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
