"""Container-side adapter around the pinned OJBench/DMOJ judge."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _partial_verdict(
    full_verdict: str, results: list[dict[str, Any]], ratio: float
) -> str:
    if not results:
        return full_verdict
    first_not_ac = next(
        (i for i, result in enumerate(results) if result["readable_main_code"] != "AC"),
        len(results),
    )
    return full_verdict if first_not_ac < round(len(results) * ratio) else "AC"


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    import ojbench  # type: ignore[import-not-found]

    ojbench.init(
        problem_dirs=[Path("/problems/NOI"), Path("/problems/ICPC")],
        config_path=Path(os.environ["OJBENCH_CONFIG_PATH"]),
        runtime_path=Path(os.environ["OJBENCH_RUNTIME_PATH"]),
        compile_lock_path=Path("/workspace/compile.lock"),
    )
    verdict, results = ojbench.judge_entry(
        {
            "id": payload["problem_id"],
            "language": payload["language"],
            "content": payload["completion"],
        },
        use_tqdm=False,
    )
    if verdict in {"IE", "Skip"}:
        raise RuntimeError(f"OJBench infrastructure verdict: {verdict}")
    partial = {
        label: _partial_verdict(verdict, results, ratio) == "AC"
        for label, ratio in (("1/8", 1 / 8), ("1/4", 1 / 4), ("1/2", 1 / 2))
    }
    return {
        "passed": verdict == "AC",
        "verdict": verdict,
        "tests_run": len(results),
        "partial_passed": partial,
    }


def main() -> int:
    payload_path = Path(sys.argv[1])
    try:
        payload = json.loads(payload_path.read_text())
        payload_path.unlink(missing_ok=True)
        result = evaluate(payload)
    except Exception as exc:
        print(json.dumps({"error": "infrastructure_error", "type": type(exc).__name__}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
