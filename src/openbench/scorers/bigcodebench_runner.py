"""Sandbox-side BigCodeBench runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bigcodebench.eval import PASS, untrusted_check  # type: ignore[import-not-found]
from bigcodebench.gen.util import trusted_check  # type: ignore[import-not-found]
from bigcodebench.sanitize import sanitize  # type: ignore[import-not-found]


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink(missing_ok=True)

    solution = sanitize(payload["completion"], payload["entry_point"])
    if payload["calibrated"]:
        solution = f"{payload['code_prompt']}\n    pass\n{solution}"

    canonical = trusted_check(
        payload["complete_prompt"] + "\n" + payload["canonical_solution"],
        payload["test"],
        payload["task_id"],
        payload["max_as_limit"],
        payload["max_data_limit"],
        payload["max_stack_limit"],
        payload["min_time_limit"],
    )
    canonical_time = canonical["time"]
    status, details = untrusted_check(
        solution,
        payload["test"],
        payload["entry_point"],
        payload["max_as_limit"],
        payload["max_data_limit"],
        payload["max_stack_limit"],
        payload["min_time_limit"],
        canonical_time if canonical_time is not None else 20,
    )
    print(
        json.dumps(
            {
                "passed": status == PASS,
                "status": status,
                "details": details,
                "canonical_time": canonical_time,
                "solution": solution,
            }
        )
    )


if __name__ == "__main__":
    main()
