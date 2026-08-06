"""Child-process runner for LiveCodeBench Python submissions.

The module deliberately has no Inspect imports so it can run inside the task's
sandbox. Each individual test executes in another bounded child process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

RESULT_MARKER = "__OPENBENCH_LCB_RESULT__="
_IS_CHILD_SUBREAPER = False
IMPORT_PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import json
import signal
import sys
sys.setrecursionlimit(50000)
sys.set_int_max_str_digits(50000)
"""


def _execute_script(
    script: str,
    input_text: str,
    timeout: int,
) -> tuple[int, str] | None:
    """Run a submission in an empty directory and kill its process group on timeout."""

    with tempfile.TemporaryDirectory(prefix="openbench-lcb-") as workdir:
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return None
    return process.returncode, stdout


def _enable_child_subreaper() -> None:
    """Adopt daemonized submission descendants inside the Linux sandbox."""

    global _IS_CHILD_SUBREAPER
    if not sys.platform.startswith("linux"):
        return

    import ctypes

    pr_set_child_subreaper = 36
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    _IS_CHILD_SUBREAPER = True


def _linux_descendants(root_pid: int) -> set[int]:
    """Return descendants from procfs, including children in new sessions."""

    proc = Path("/proc")
    if not proc.is_dir():
        return set()

    parents: dict[int, int] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue

    descendants: set[int] = set()
    frontier = {root_pid}
    while frontier:
        children = {pid for pid, parent in parents.items() if parent in frontier}
        children -= descendants
        descendants.update(children)
        frontier = children
    return descendants


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop and kill the submission tree, then reap the direct child."""

    root_pid = os.getpid() if _IS_CHILD_SUBREAPER else process.pid
    descendants: set[int] = set()
    for _ in range(3):
        newly_found = _linux_descendants(root_pid) - descendants
        if not newly_found:
            break
        descendants.update(newly_found)
        for pid in newly_found:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass

    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)
    if _IS_CHILD_SUBREAPER:
        while True:
            try:
                reaped_pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if reaped_pid == 0:
                break
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _decimal_tokens(value: str) -> list[Decimal] | None:
    try:
        return [Decimal(token) for token in value.split()]
    except InvalidOperation:
        return None


def outputs_match(actual: str, expected: str) -> bool:
    """Compare stdout using LiveCodeBench's stripped-line semantics."""

    actual_lines = [line.strip() for line in actual.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue
        actual_numbers = _decimal_tokens(actual_line)
        expected_numbers = _decimal_tokens(expected_line)
        if actual_numbers is None or expected_numbers is None:
            return False
        if actual_numbers != expected_numbers:
            return False
    return True


def _run_stdio(code: str, test: dict[str, Any], timeout: int) -> tuple[bool, str]:
    result = _execute_script(
        IMPORT_PRELUDE + "\n" + code,
        str(test["input"]),
        timeout,
    )
    if result is None:
        return False, "timeout"

    returncode, stdout = result
    if returncode != 0:
        return False, "runtime_error"
    if not outputs_match(stdout, str(test["output"])):
        return False, "wrong_answer"
    return True, ""


def _functional_harness(code: str) -> str:
    return (
        IMPORT_PRELUDE
        + "\n"
        + code
        + "\n"
        + """
def __openbench_lcb_main():
    import builtins as __ob_builtins
    import json as __ob_json
    import signal as __ob_signal
    import sys as __ob_sys
    namespace = __ob_builtins.globals()
    request = __ob_json.loads(__ob_sys.stdin.read())
    solution_class = namespace.get("Solution")
    target = (
        solution_class()
        if __ob_builtins.isinstance(solution_class, __ob_builtins.type)
        else namespace
    )
    function = (
        __ob_builtins.getattr(target, request["fn_name"])
        if __ob_builtins.isinstance(solution_class, __ob_builtins.type)
        else target[request["fn_name"]]
    )
    results = []
    error = None
    for args in request["args_list"]:
        def timeout_handler(signum, frame):
            raise __ob_builtins.TimeoutError()
        __ob_signal.signal(__ob_signal.SIGALRM, timeout_handler)
        __ob_signal.alarm(request["timeout"])
        try:
            result = function(*args)
            if __ob_builtins.isinstance(result, __ob_builtins.tuple):
                result = __ob_builtins.list(result)
            results.append(result)
        except __ob_builtins.TimeoutError:
            error = "timeout"
            break
        except __ob_builtins.Exception:
            error = "runtime_error"
            break
        finally:
            __ob_signal.alarm(0)
    __ob_sys.stdout.write("""
        + repr(RESULT_MARKER)
        + ' + __ob_json.dumps({"results": results, "error": error}, ensure_ascii=False) + "\\n")\n'
        + "__openbench_lcb_main()\n"
    )


def _run_functional_tests(
    code: str,
    tests: list[dict[str, Any]],
    function_name: str,
    timeout: int,
) -> tuple[bool, str, int]:
    try:
        arguments = [
            [json.loads(line) for line in str(test["input"]).splitlines()]
            for test in tests
        ]
        expected = [json.loads(str(test["output"])) for test in tests]
    except (json.JSONDecodeError, TypeError):
        return False, "invalid_test", 0

    result = _execute_script(
        _functional_harness(code),
        json.dumps(
            {
                "fn_name": function_name,
                "args_list": arguments,
                "timeout": timeout,
            }
        ),
        timeout * max(len(tests), 1) + 5,
    )
    if result is None:
        return False, "timeout", 0

    returncode, stdout = result
    if returncode != 0:
        return False, "runtime_error", 0
    marker_position = stdout.rfind(RESULT_MARKER)
    if marker_position < 0:
        return False, "missing_result", 0
    encoded_result = stdout[marker_position + len(RESULT_MARKER) :].strip()
    try:
        evaluation = json.loads(encoded_result)
    except json.JSONDecodeError:
        return False, "invalid_result", 0

    actual = evaluation.get("results", [])
    for index, expected_result in enumerate(expected):
        if index >= len(actual):
            return False, evaluation.get("error") or "missing_result", index + 1
        if actual[index] != expected_result:
            return False, "wrong_answer", index + 1
    return True, "", len(tests)


def evaluate_submission(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one generated program against all public and hidden tests."""

    code = payload["code"]
    tests = payload["tests"]
    function_name = payload.get("function_name")
    timeout = int(payload.get("timeout", 6))

    if function_name:
        passed, error, tests_run = _run_functional_tests(
            code,
            tests,
            function_name,
            timeout,
        )
        return {
            "passed": passed,
            "tests_run": tests_run,
            "failed_test_index": None if passed else max(tests_run - 1, 0),
            "error": None if passed else error,
        }

    for index, test in enumerate(tests):
        passed, error = _run_stdio(code, test, timeout)
        if not passed:
            return {
                "passed": False,
                "tests_run": index + 1,
                "failed_test_index": index,
                "error": error,
            }
    return {"passed": True, "tests_run": len(tests), "error": None}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: livecodebench_runner PAYLOAD.json")
    _enable_child_subreaper()
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()
    print(json.dumps(evaluate_submission(payload)))


if __name__ == "__main__":
    main()
