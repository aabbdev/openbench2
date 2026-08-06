"""EvalPlus scorer backed by a fail-closed Docker sandbox."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from openbench.datasets.evalplus import load_evalplus_record
from openbench.scorers.evalplus_sanitize import sanitize


def extract_python(completion: str) -> str:
    """Compatibility wrapper around the canonical EvalPlus sanitizer."""
    return sanitize(completion)


@scorer(
    metrics=[
        {
            "base": [accuracy(), stderr()],
            "plus": [accuracy(), stderr()],
        }
    ]
)
def evalplus_scorer(total_timeout: int = 900) -> Scorer:
    """Evaluate base and plus tests; both must pass for a correct sample."""
    if total_timeout <= 0:
        raise ValueError("total_timeout must be positive")

    async def score(state: TaskState, target: Target) -> Score:
        del target
        record = load_evalplus_record(state.metadata)
        completion = state.output.completion
        prompt = str(record["prompt"])
        entry_point = str(record["entry_point"])
        code = sanitize(completion, entry_point)
        payload = {
            "dataset": state.metadata["dataset"],
            "task_id": record["task_id"],
            "entry_point": entry_point,
            "prompt": prompt,
            "canonical_solution": record["canonical_solution"],
            "base_input": record["base_input"],
            "plus_input": record["plus_input"],
            "atol": record["atol"],
            "code": code,
        }
        environment = sandbox()
        payload_path = ".openbench_evalplus_payload.json"
        runner_path = ".openbench_evalplus_runner.py"
        await environment.write_file(payload_path, json.dumps(payload, allow_nan=True))
        await environment.write_file(
            runner_path,
            Path(__file__).with_name("evalplus_runner.py").read_text(),
        )
        try:
            result = await environment.exec(
                ["python", runner_path, payload_path],
                timeout=total_timeout,
                timeout_retry=False,
            )
        except TimeoutError:
            return Score(
                value={"base": INCORRECT, "plus": INCORRECT},
                answer=completion,
                explanation="EvalPlus runner timed out",
            )
        if not result.success:
            return Score(
                value={"base": INCORRECT, "plus": INCORRECT},
                answer=completion,
                explanation="EvalPlus sandbox failed",
            )
        try:
            evaluation = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return Score(
                value={"base": INCORRECT, "plus": INCORRECT},
                answer=completion,
                explanation="Invalid EvalPlus result",
            )
        base_passed = evaluation.get("base_passed") is True
        passed = evaluation.get("passed") is True
        return Score(
            value={
                "base": CORRECT if base_passed else INCORRECT,
                "plus": CORRECT if passed else INCORRECT,
            },
            answer=completion,
            explanation=(
                f"base={evaluation.get('base_passed')}, "
                f"plus={evaluation.get('plus_passed')}, "
                f"tests={evaluation.get('tests_run', 0)}, "
                f"error={evaluation.get('error')}"
            ),
            metadata={
                "base_passed": evaluation.get("base_passed", False),
                "plus_passed": evaluation.get("plus_passed", False),
            },
        )

    return score
