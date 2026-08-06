"""Container-side differential runner for EvalPlus.

The payload containing canonical code and hidden outputs is unlinked before any
candidate process starts. Candidates receive only invocation inputs and time limits.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

MBPP_OUTPUT_NOT_NONE_TASKS = {"check_str", "text_match_three", "text_starta_endb"}
MBPP_OUTPUT_SET_EQ_TASKS = {
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
}

CHILD_RUNNER = r"""
import json as _json
import os as _os
import signal as _signal
import sys as _sys
import time as _time

def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "set":
        return set(_decode(item) for item in value["items"])
    if kind == "complex":
        return complex(value["real"], value["imag"])
    if kind == "dict":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    raise ValueError("Unknown wire type")

def _encode(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value, "pytype": type(value).__module__ + "." + type(value).__qualname__}
    if isinstance(value, complex):
        return {"type": "complex", "real": value.real, "imag": value.imag, "pytype": type(value).__module__ + "." + type(value).__qualname__}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_encode(item) for item in value]
        return {"type": "set", "items": sorted(items, key=repr)}
    if isinstance(value, dict):
        items = [(_encode(key), _encode(item)) for key, item in value.items()]
        return {"type": "dict", "items": sorted(items, key=lambda item: repr(item[0]))}
    if hasattr(value, "tolist"):
        return {"type": "array", "value": _encode(value.tolist()), "pytype": type(value).__module__ + "." + type(value).__qualname__}
    return {"type": "non_none"}

def _timeout(signum, frame):
    raise TimeoutError()

request = _json.loads(_sys.stdin.read())
inputs = _decode(request["inputs"])
namespace = {}
try:
    exec(request["code"], namespace)
    function = namespace[request["entry_point"]]
except BaseException:
    result = {"error": "compile_error", "outputs": [], "times": []}
    _os.write(1, (_json.dumps(result) + "\n").encode())
    raise SystemExit(0)

outputs = []
times = []
for args, limit in zip(inputs, request["time_limits"]):
    _signal.signal(_signal.SIGALRM, _timeout)
    _signal.setitimer(_signal.ITIMER_REAL, limit)
    started = _time.perf_counter()
    try:
        outputs.append({"ok": True, "value": _encode(function(*args))})
    except TimeoutError:
        outputs.append({"ok": False, "error": "timeout"})
    except BaseException:
        outputs.append({"ok": False, "error": "runtime_error"})
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        times.append(_time.perf_counter() - started)

result = {"error": None, "outputs": outputs, "times": times}
_os.write(1, (_json.dumps(result, allow_nan=True) + "\n").encode())
"""


def _deserialize_mbpp_inputs(task_id: str, inputs: list) -> list:
    number = int(task_id.split("/")[-1])
    tuple_lists = {
        2,
        116,
        132,
        143,
        222,
        261,
        273,
        394,
        399,
        421,
        424,
        429,
        470,
        560,
        579,
        596,
        616,
        630,
        726,
        740,
        744,
        809,
    }
    nested_tuple_lists = {
        63,
        64,
        70,
        94,
        120,
        237,
        272,
        299,
        400,
        409,
        417,
        438,
        473,
        614,
        780,
    }
    if number in tuple_lists:
        return [[tuple(item) for item in args] for args in inputs]
    if number in nested_tuple_lists:
        return [[[tuple(item) for item in group] for group in args] for args in inputs]
    if number in {75, 413, 444, 753}:
        return [[[tuple(item) for item in args[0]], args[1]] for args in inputs]
    if number in {106, 750}:
        return [[args[0], tuple(args[1])] for args in inputs]
    if number == 115:
        return [[[set(item) if item else {} for item in args[0]]] for args in inputs]
    if number == 124:
        return [[float(args[0]), complex(args[1])] for args in inputs]
    if number in {250, 405, 446, 617, 720, 763, 808}:
        return [[tuple(args[0]), args[1]] for args in inputs]
    if number in {259, 401, 445}:
        converted = [
            [[tuple(item) for item in group] for group in args] for args in inputs
        ]
        return [[tuple(group) for group in args] for args in converted]
    if number == 278:
        return [
            [tuple(tuple(item) if isinstance(item, list) else item for item in args[0])]
            for args in inputs
        ]
    if number == 307:
        return [[tuple(args[0]), args[1], args[2]] for args in inputs]
    if number == 722:
        return [
            [{key: tuple(value) for key, value in args[0].items()}, *args[1:]]
            for args in inputs
        ]
    if number == 252:
        return [[complex(args[0])] for args in inputs]
    if number in {580, 615, 791}:

        def tuples(value: Any) -> Any:
            return (
                tuple(tuples(item) for item in value)
                if isinstance(value, list)
                else value
            )

        return [tuples(args) for args in inputs]
    return inputs


def _wire(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_wire(item) for item in value]}
    if isinstance(value, set):
        return {"__type__": "set", "items": [_wire(item) for item in value]}
    if isinstance(value, complex):
        return {"__type__": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "items": [[_wire(key), _wire(item)] for key, item in value.items()],
        }
    return value


def _run_code(
    code: str,
    entry_point: str,
    inputs: list,
    limits: list[float],
    timeout: float,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="openbench-evalplus-") as workdir:
        process = subprocess.Popen(
            [sys.executable, "-c", CHILD_RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            start_new_session=True,
        )
        request = json.dumps(
            {
                "code": code,
                "entry_point": entry_point,
                "inputs": _wire(inputs),
                "time_limits": limits,
            },
            allow_nan=True,
        )
        try:
            stdout, _ = process.communicate(request, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return {"error": "task_timeout", "outputs": [], "times": []}
    if process.returncode != 0:
        return {"error": "runner_error", "outputs": [], "times": []}
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": "invalid_result", "outputs": [], "times": []}


def _value(encoded: dict) -> Any:
    if encoded.get("type") == "scalar":
        return encoded.get("value")
    if encoded.get("type") == "complex":
        return complex(encoded["real"], encoded["imag"])
    if encoded.get("type") == "tuple":
        return tuple(_value(item) for item in encoded.get("items", []))
    if encoded.get("type") == "list":
        return [_value(item) for item in encoded.get("items", [])]
    if encoded.get("type") == "set":
        return set(_value(item) for item in encoded.get("items", []))
    if encoded.get("type") == "dict":
        return {_value(key): _value(item) for key, item in encoded.get("items", [])}
    if encoded.get("type") == "array":
        return np.asarray(_value(encoded["value"]))
    return object()


def _scalar(encoded: dict) -> Any:
    value = _value(encoded)
    return value if isinstance(value, (bool, int, float, complex, str)) else None


def _allclose(actual: dict, expected: dict, atol: float) -> bool:
    left, right = _value(actual), _value(expected)
    try:
        if bool(left == right):
            return True
    except (TypeError, ValueError):
        return False
    expected_is_floats = (
        isinstance(right, float)
        or (
            isinstance(right, (list, tuple))
            and bool(right)
            and all(isinstance(item, float) for item in right)
        )
        or (isinstance(right, np.ndarray) and right.dtype in {np.float32, np.float64})
    )
    if atol == 0 and expected_is_floats:
        atol = 1e-6
    if atol == 0 or type(left) is not type(right):
        return False
    if isinstance(right, (list, tuple)) and len(left) != len(right):
        return False
    try:
        return bool(np.allclose(left, right, rtol=1e-7, atol=atol))
    except (TypeError, ValueError):
        return False


def _poly(coefficients: list, x: float) -> float:
    return sum(
        coefficient * math.pow(x, index)
        for index, coefficient in enumerate(coefficients)
    )


def _matches(
    dataset: str,
    entry_point: str,
    args: list,
    actual: dict,
    expected: dict,
    atol: float,
) -> bool:
    try:
        if actual == expected:
            return True
        if dataset == "mbpp":
            if entry_point == "are_equivalent":
                return True
            if entry_point == "sum_div" and _scalar(actual) == 0:
                return True
            if entry_point == "surface_Area":
                base_edge, height = args
                slant = math.sqrt((base_edge / 2) ** 2 + height**2)
                reference = round(base_edge**2 + 2 * base_edge * slant)
                return abs(_scalar(actual) - reference) <= atol
            if entry_point == "digit_distance_nums":
                one, two = str(args[0]), str(args[1])
                width = max(len(one), len(two))
                reference = sum(
                    abs(int(a) - int(b))
                    for a, b in zip(one.zfill(width), two.zfill(width))
                )
                return _scalar(actual) == reference
            if entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                return set(_value(actual)) == set(_value(expected))
            if entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                value, expected_value = _scalar(actual), _scalar(expected)
                if isinstance(value, bool):
                    return value == expected_value
                return expected_value == (
                    actual.get("type") != "scalar" or value is not None
                )
        if dataset == "humaneval" and entry_point == "find_zero":
            root = _scalar(actual)
            return isinstance(root, (int, float)) and abs(_poly(args[0], root)) <= atol
        return _allclose(actual, expected, atol)
    except (ArithmeticError, TypeError, ValueError):
        return False


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = str(payload["dataset"])
    task_id = str(payload["task_id"])
    entry_point = str(payload["entry_point"])
    base_inputs, plus_inputs = payload["base_input"], payload["plus_input"]
    if dataset == "mbpp":
        base_inputs = _deserialize_mbpp_inputs(task_id, base_inputs)
        plus_inputs = _deserialize_mbpp_inputs(task_id, plus_inputs)
    canonical = str(payload["prompt"]) + str(payload["canonical_solution"])
    atol = float(payload.get("atol", 0.0))

    def run_suite(inputs: list) -> tuple[bool, int, str | None]:
        if not inputs:
            return True, 0, None
        oracle = _run_code(
            canonical,
            entry_point,
            inputs,
            [30.0] * len(inputs),
            max(60.0, len(inputs) * 30.0),
        )
        if oracle.get("error") or len(oracle.get("outputs", [])) != len(inputs):
            return False, 0, "oracle_failure"
        if dataset == "mbpp" and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
            for output in oracle["outputs"]:
                if output.get("ok"):
                    is_not_none = not (
                        output["value"].get("type") == "scalar"
                        and output["value"].get("value") is None
                    )
                    output["value"] = {
                        "type": "scalar",
                        "value": is_not_none,
                        "pytype": "builtins.bool",
                    }
        limits = [max(4.0, 4.0 * elapsed) for elapsed in oracle["times"]]
        candidate = _run_code(
            str(payload["code"]),
            entry_point,
            inputs,
            limits,
            min(60.0, sum(limits)) + 2.0,
        )
        if candidate.get("error") or len(candidate.get("outputs", [])) != len(inputs):
            return False, 0, candidate.get("error", "candidate_failure")
        for args, actual, expected in zip(
            inputs, candidate["outputs"], oracle["outputs"]
        ):
            if not (
                actual.get("ok") is True
                and expected.get("ok") is True
                and _matches(
                    dataset,
                    entry_point,
                    args,
                    actual["value"],
                    expected["value"],
                    atol,
                )
            ):
                return False, len(inputs), "wrong_answer"
        return True, len(inputs), None

    base_passed, base_count, base_error = run_suite(base_inputs)
    plus_passed, plus_count, plus_error = run_suite(plus_inputs)
    return {
        "passed": base_passed and plus_passed,
        "base_passed": base_passed,
        "plus_passed": plus_passed,
        "tests_run": base_count + plus_count,
        "error": base_error or plus_error,
    }


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()
    print(json.dumps(evaluate(payload)))


if __name__ == "__main__":
    main()
