"""Container-side adapter around BFCL's pinned official multi-turn checker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # type: ignore[import-not-found]
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # type: ignore[import-not-found]
    execute_multi_turn_func_call,
)


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    calls = [str(value) for value in payload["calls"]]
    previous_count = int(payload.get("previous_count", 0))
    outputs, _ = execute_multi_turn_func_call(
        func_call_list=calls,
        initial_config=payload["initial_config"],
        involved_classes=payload["involved_classes"],
        model_name="openbench_generation",
        test_entry_id=payload["id"],
        long_context="long_context" in payload["category"],
        is_evaL_run=False,
    )
    return {"outputs": outputs[previous_count:]}


def score(payload: dict[str, Any]) -> dict[str, Any]:
    result = multi_turn_checker(
        payload["model_turn_calls"],
        payload["ground_truth"],
        payload["test_entry"],
        payload["category"],
        "openbench",
    )
    if result.get("valid"):
        result = multi_turn_irrelevance_checker(
            payload["model_turn_calls"], payload["ground_truth"]
        )
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: bfcl_runner PAYLOAD.json")
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()
    operation = payload.pop("operation")
    result = execute(payload) if operation == "execute" else score(payload)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
