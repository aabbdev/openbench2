"""BigCodeBench scorer using the official evaluator inside Docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

from openbench.datasets.bigcodebench import load_bigcodebench_execution_fields


def build_bigcodebench_payload(
    completion: str,
    fields: tuple[str, str, str, str, str],
    *,
    task_id: str,
    calibrated: bool,
    min_time_limit: int,
    max_as_limit: int,
    max_data_limit: int,
    max_stack_limit: int,
) -> dict[str, Any]:
    """Build the JSON payload consumed by the sandbox runner."""

    code_prompt, test, entry_point, complete_prompt, canonical_solution = fields
    return {
        "completion": completion,
        "task_id": task_id,
        "code_prompt": code_prompt,
        "test": test,
        "entry_point": entry_point,
        "complete_prompt": complete_prompt,
        "canonical_solution": canonical_solution,
        "calibrated": calibrated,
        "min_time_limit": min_time_limit,
        "max_as_limit": max_as_limit,
        "max_data_limit": max_data_limit,
        "max_stack_limit": max_stack_limit,
    }


def score_bigcodebench_evaluation(evaluation: dict[str, Any], task_id: str) -> Score:
    """Convert a runner result while refusing uncalibrated canonical failures."""

    status = evaluation.get("status", "unknown")
    if status == "canonical_error":
        raise RuntimeError(
            f"BigCodeBench canonical validation failed for {task_id}; "
            "refusing to report a model score"
        )

    passed = evaluation.get("passed") is True
    explanation = (
        "Passed official BigCodeBench tests."
        if passed
        else f"Failed official BigCodeBench tests with status: {status}."
    )
    return Score(
        value=CORRECT if passed else INCORRECT,
        answer=evaluation.get("solution", ""),
        explanation=explanation,
    )


@scorer(metrics=[accuracy(), stderr()])
def bigcodebench_scorer(
    *,
    calibrated: bool = True,
    min_time_limit: int = 1,
    max_as_limit: int = 30 * 1024,
    max_data_limit: int = 30 * 1024,
    max_stack_limit: int = 10,
    total_timeout: int = 900,
) -> Scorer:
    """Score BigCodeBench completions with the official sanitizer/evaluator."""

    if min_time_limit <= 0:
        raise ValueError("min_time_limit must be positive")
    if max_as_limit <= 0:
        raise ValueError("max_as_limit must be positive")
    if max_data_limit <= 0:
        raise ValueError("max_data_limit must be positive")
    if max_stack_limit <= 0:
        raise ValueError("max_stack_limit must be positive")
    if total_timeout <= 0:
        raise ValueError("total_timeout must be positive")

    async def score(state: TaskState, target: Target) -> Score:
        del target
        fields = load_bigcodebench_execution_fields(state.metadata)
        payload = build_bigcodebench_payload(
            state.output.completion,
            fields,
            task_id=str(state.metadata["task_id"]),
            calibrated=calibrated,
            min_time_limit=min_time_limit,
            max_as_limit=max_as_limit,
            max_data_limit=max_data_limit,
            max_stack_limit=max_stack_limit,
        )
        environment = sandbox()
        payload_path = ".openbench_bigcodebench_payload.json"
        runner_path = ".openbench_bigcodebench_runner.py"
        runner_source = Path(__file__).with_name("bigcodebench_runner.py").read_text()
        await environment.write_file(payload_path, json.dumps(payload))
        await environment.write_file(runner_path, runner_source)

        try:
            result = await environment.exec(
                ["python3", runner_path, payload_path],
                timeout=total_timeout,
                timeout_retry=False,
            )
        except TimeoutError:
            return Score(
                value=INCORRECT,
                explanation="BigCodeBench evaluation exceeded its total timeout.",
            )

        if not result.success:
            return Score(
                value=INCORRECT,
                explanation="BigCodeBench runner failed inside the sandbox.",
            )

        try:
            evaluation = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return Score(
                value=INCORRECT,
                explanation="BigCodeBench runner returned an invalid result.",
            )

        return score_bigcodebench_evaluation(
            evaluation,
            task_id=str(state.metadata["task_id"]),
        )

    return score
