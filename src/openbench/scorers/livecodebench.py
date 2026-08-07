"""LiveCodeBench v6 scorer using isolated, sandboxed program execution."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from io import BytesIO
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

from openbench.datasets.livecodebench import load_livecodebench_test_fields


class _DataOnlyUnpickler(pickle.Unpickler):
    """Decode upstream's compressed JSON string without loading classes."""

    def find_class(self, module: str, name: str) -> None:
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def extract_code(completion: str) -> str:
    """Extract the last fenced code block, matching LiveCodeBench parsing."""

    lines = completion.splitlines()
    fence_lines = [index for index, line in enumerate(lines) if "```" in line]
    if len(fence_lines) < 2:
        return ""
    return "\n".join(lines[fence_lines[-2] + 1 : fence_lines[-1]])


def extract_livebench_code(completion: str) -> str:
    """Extract code using LiveBench's generic-model fallback behavior."""

    stripped = completion.rstrip()
    lines = stripped.splitlines()
    fence_lines = [index for index, line in enumerate(lines) if "```" in line]
    if len(fence_lines) >= 2:
        return "\n".join(lines[fence_lines[-2] + 1 : fence_lines[-1]])
    if len(completion) > 1 and completion[0] == "`" and completion[-1] == "`":
        return completion[1:-1]
    if len(fence_lines) == 1 and fence_lines[0] == len(lines) - 1:
        return "\n".join(lines[:-1])
    return stripped


def decode_test_cases(value: str | list[dict[str, object]]) -> list[dict[str, object]]:
    """Decode plain JSON or the pinned dataset's compressed hidden tests."""

    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        packed = zlib.decompress(base64.b64decode(value.encode("utf-8")))
        serialized_json = _DataOnlyUnpickler(BytesIO(packed)).load()
        decoded = json.loads(serialized_json)
    if not isinstance(decoded, list):
        raise TypeError("LiveCodeBench test cases must decode to a list")
    return decoded


def decode_metadata(value: str | dict[str, object]) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError("LiveCodeBench metadata must decode to an object")
    return decoded


@scorer(metrics=[accuracy(), stderr()])
def livecodebench_scorer(
    test_timeout: int = 6,
    total_timeout: int = 600,
) -> Scorer:
    """Run a completion against every public and private test in the sandbox."""

    if test_timeout <= 0:
        raise ValueError("test_timeout must be positive")
    if total_timeout <= 0:
        raise ValueError("total_timeout must be positive")

    async def score(state: TaskState, target: Target) -> Score:
        del target
        code = extract_code(state.output.completion)
        if "source_file" in state.metadata:
            public_tests, private_tests, metadata = load_livecodebench_test_fields(
                state.metadata
            )
        else:
            public_tests = state.metadata["public_test_cases"]
            private_tests = state.metadata["private_test_cases"]
            metadata = state.metadata["test_metadata"]
        tests = decode_test_cases(public_tests)
        tests += decode_test_cases(private_tests)
        test_metadata = decode_metadata(metadata)

        payload = {
            "code": code,
            "tests": tests,
            "function_name": test_metadata.get("func_name"),
            "timeout": test_timeout,
        }
        environment = sandbox()
        payload_path = ".openbench_livecodebench_payload.json"
        runner_path = ".openbench_livecodebench_runner.py"
        runner_source = Path(__file__).with_name("livecodebench_runner.py").read_text()
        await environment.write_file(payload_path, json.dumps(payload))
        await environment.write_file(runner_path, runner_source)

        try:
            result = await environment.exec(
                ["python", runner_path, payload_path],
                timeout=total_timeout,
                timeout_retry=False,
            )
        except TimeoutError:
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveCodeBench evaluation exceeded its total timeout.",
            )

        if not result.success:
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveCodeBench runner failed inside the sandbox.",
            )

        try:
            evaluation = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveCodeBench runner returned an invalid result.",
            )

        passed = evaluation.get("passed") is True
        explanation = (
            f"Passed all {evaluation['tests_run']} tests."
            if passed
            else (
                f"Failed after {evaluation.get('tests_run', 0)} test(s): "
                f"{evaluation.get('error', 'unknown error')}."
            )
        )
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=code,
            explanation=explanation,
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def livebench_coding_scorer(
    test_timeout: int = 6,
    total_timeout: int = 600,
) -> Scorer:
    """Score public LiveBench coding tasks in the hardened LCB sandbox."""

    if test_timeout <= 0:
        raise ValueError("test_timeout must be positive")
    if total_timeout <= 0:
        raise ValueError("total_timeout must be positive")

    async def score(state: TaskState, target: Target) -> Score:
        del target
        from openbench.datasets.livebench import load_livebench_test_fields

        code = extract_livebench_code(state.output.completion)
        public_tests, private_tests, metadata, partial_solution = (
            load_livebench_test_fields(state.metadata)
        )
        if partial_solution and not code.startswith(partial_solution):
            code = f"{partial_solution}\n{code}"
        tests = decode_test_cases(public_tests) + decode_test_cases(private_tests)
        test_metadata = decode_metadata(metadata)
        payload = {
            "code": code,
            "tests": tests,
            "function_name": test_metadata.get("func_name"),
            "timeout": test_timeout,
        }
        environment = sandbox()
        payload_path = ".openbench_livebench_payload.json"
        runner_path = ".openbench_livebench_runner.py"
        runner_source = Path(__file__).with_name("livecodebench_runner.py").read_text()
        await environment.write_file(payload_path, json.dumps(payload))
        await environment.write_file(runner_path, runner_source)
        try:
            result = await environment.exec(
                ["python", runner_path, payload_path],
                timeout=total_timeout,
                timeout_retry=False,
            )
        except TimeoutError:
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveBench evaluation exceeded its total timeout.",
            )
        if not result.success:
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveBench runner failed inside the sandbox.",
            )
        try:
            evaluation = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return Score(
                value=INCORRECT,
                answer=code,
                explanation="LiveBench runner returned an invalid result.",
            )
        passed = evaluation.get("passed") is True
        explanation = (
            f"Passed all {evaluation['tests_run']} tests."
            if passed
            else (
                f"Failed after {evaluation.get('tests_run', 0)} test(s): "
                f"{evaluation.get('error', 'unknown error')}."
            )
        )
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=code,
            explanation=explanation,
        )

    return score
