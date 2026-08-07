"""OJBench scorer backed by the pinned DMOJ sandbox."""

from __future__ import annotations

import asyncio
import json
import threading
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

from openbench.datasets.ojbench import ensure_problem_data

_DOWNLOAD_LOCK = threading.Lock()


def _prepare_problem(dataset: str, problem_id: str) -> None:
    with _DOWNLOAD_LOCK:
        ensure_problem_data(dataset, problem_id)


@scorer(metrics=[accuracy(), stderr()])
def ojbench_scorer(total_timeout: int = 3600) -> Scorer:
    """Judge one completion without exposing problem archives to Inspect logs."""

    if total_timeout <= 0:
        raise ValueError("total_timeout must be positive")

    async def score(state: TaskState, target: Target) -> Score:
        del target
        dataset = str(state.metadata["dataset"])
        problem_id = str(state.metadata["problem_id"])
        await asyncio.to_thread(_prepare_problem, dataset, problem_id)

        payload = {
            "record_id": state.metadata["record_id"],
            "problem_id": problem_id,
            "language": state.metadata["language"],
            "completion": state.output.completion,
        }
        environment = sandbox()
        payload_path = ".openbench_ojbench_payload.json"
        runner_path = ".openbench_ojbench_runner.py"
        await environment.write_file(payload_path, json.dumps(payload))
        await environment.write_file(
            runner_path,
            Path(__file__).with_name("ojbench_runner.py").read_text(),
        )
        try:
            result = await environment.exec(
                ["python", runner_path, payload_path],
                timeout=total_timeout,
                timeout_retry=False,
            )
        except TimeoutError as exc:
            raise RuntimeError("OJBench judge exceeded its total timeout") from exc
        if not result.success:
            raise RuntimeError("OJBench judge failed inside the DMOJ sandbox")
        try:
            evaluation = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("OJBench judge returned an invalid result") from exc
        if not isinstance(evaluation.get("passed"), bool):
            raise RuntimeError("OJBench judge omitted its pass verdict")
        return Score(
            value=CORRECT if evaluation["passed"] else INCORRECT,
            answer=state.output.completion,
            explanation=(
                f"OJBench verdict={evaluation['verdict']}; "
                f"tests={evaluation['tests_run']}"
            ),
            metadata={"partial_passed": evaluation["partial_passed"]},
        )

    return score
