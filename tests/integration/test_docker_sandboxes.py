"""End-to-end checks for hardened code-execution sandboxes."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.docker

ROOT = Path(__file__).parents[2]


def _docker(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_container_file(compose: list[str], destination: str, source: Path) -> None:
    subprocess.run(
        [
            "docker",
            *compose,
            "exec",
            "-T",
            "default",
            "sh",
            "-c",
            f"cat > {destination}",
        ],
        cwd=ROOT,
        input=source.read_text(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _exercise_sandbox(
    tmp_path: Path,
    *,
    compose_path: Path,
    runner_path: Path | None,
    payload: dict[str, object],
    payload_name: str,
    image_runner_path: str | None = None,
) -> dict[str, object]:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is unavailable")
    try:
        _docker("info", timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon is unavailable")

    project = f"openbench-sandbox-{uuid.uuid4().hex[:12]}"
    compose = ["compose", "-p", project, "-f", str(compose_path)]
    local_payload = tmp_path / payload_name
    local_payload.write_text(json.dumps(payload))

    try:
        _docker(*compose, "build", timeout=600)
        _docker(*compose, "up", "-d", timeout=120)
        container_id = _docker(*compose, "ps", "-q", "default").stdout.strip()
        assert container_id

        inspection = json.loads(_docker("inspect", container_id).stdout)[0]
        host = inspection["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["ReadonlyRootfs"] is True
        assert host["PidsLimit"] == 64
        assert "ALL" in host["CapDrop"]
        assert "no-new-privileges:true" in host["SecurityOpt"]

        container_runner = image_runner_path or (
            f"/workspace/{runner_path.name}" if runner_path is not None else ""
        )
        container_payload = f"/workspace/{payload_name}"
        if runner_path is not None:
            _write_container_file(compose, container_runner, runner_path)
        _write_container_file(compose, container_payload, local_payload)
        completed = _docker(
            *compose,
            "exec",
            "-T",
            "default",
            "python",
            container_runner,
            container_payload,
            timeout=180,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        subprocess.run(
            ["docker", *compose, "down", "--volumes", "--remove-orphans"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )


def test_livecodebench_sandbox_executes_without_payload_leak(tmp_path: Path) -> None:
    result = _exercise_sandbox(
        tmp_path,
        compose_path=ROOT / "src/openbench/evals/livecodebench/compose.yaml",
        runner_path=ROOT / "src/openbench/scorers/livecodebench_runner.py",
        payload={
            "code": (
                "from pathlib import Path\n"
                "print('leaked' if list(Path('/workspace').glob('*payload*')) "
                "else 'safe')"
            ),
            "tests": [{"input": "", "output": "safe\n"}],
            "function_name": None,
            "timeout": 2,
        },
        payload_name=".openbench_livecodebench_payload.json",
    )
    assert result["passed"] is True


def test_evalplus_sandbox_executes_without_payload_leak(tmp_path: Path) -> None:
    result = _exercise_sandbox(
        tmp_path,
        compose_path=ROOT / "src/openbench/evals/evalplus/compose.yaml",
        runner_path=ROOT / "src/openbench/scorers/evalplus_runner.py",
        payload={
            "dataset": "humaneval",
            "task_id": "HumanEval/0",
            "entry_point": "add",
            "prompt": "def add(a, b):\n",
            "canonical_solution": "    return a + b\n",
            "base_input": [[1, 2]],
            "plus_input": [[-4, 9]],
            "atol": 0,
            "code": (
                "def add(a, b):\n"
                "    from pathlib import Path\n"
                "    leaked = list(Path('/workspace').glob('*payload*'))\n"
                "    return -1 if leaked else a + b\n"
            ),
        },
        payload_name=".openbench_evalplus_payload.json",
    )
    assert result["passed"] is True


def test_bfcl_official_multi_turn_checker_runs_in_sandbox(tmp_path: Path) -> None:
    ground_truth = [["add(a=1.0, b=2.0)"]]
    result = _exercise_sandbox(
        tmp_path,
        compose_path=ROOT / "src/openbench/evals/bfcl/compose.yaml",
        runner_path=None,
        image_runner_path="/opt/openbench/bfcl_runner.py",
        payload={
            "operation": "score",
            "model_turn_calls": [[ground_truth[0]]],
            "ground_truth": ground_truth,
            "category": "multi_turn_base",
            "test_entry": {
                "id": "multi_turn_base_smoke",
                "initial_config": {},
                "involved_classes": ["MathAPI"],
            },
        },
        payload_name=".openbench_bfcl_payload.json",
    )
    assert result["valid"] is True
