"""Tests for the pinned OJBench adapter."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai.dataset import MemoryDataset, Sample
import yaml

from openbench.config import BENCHMARKS
from openbench.datasets.ojbench import (
    PROMPT_SHA256,
    canonical_problem_id,
    ensure_problem_data,
    get_ojbench_dataset,
    record_to_sample,
)
from openbench.evals.ojbench import ojbench_cpp, ojbench_python
from openbench.scorers.ojbench_runner import _partial_verdict, evaluate


def _record(language: str = "python") -> dict[str, object]:
    return {
        "id": 2083,
        "prompt": "private prompt",
        "dataset": "NOI",
        "language": language,
        "difficulty": "hard",
    }


def test_record_to_sample_preserves_official_identity():
    sample = record_to_sample(_record())
    assert sample.id == "loj-2083:python"
    assert sample.input == "private prompt"
    assert sample.metadata["problem_id"] == "loj-2083"
    assert sample.metadata["dataset"] == "NOI"


def test_canonical_problem_id_rejects_path_traversal():
    assert canonical_problem_id("icpc", "nwerc2022_A") == ("ICPC", "nwerc2022_A")
    try:
        canonical_problem_id("icpc", "../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")


def test_dataset_filters_language_and_checks_release(tmp_path: Path):
    rows = [_record("python"), _record("cpp")]
    prompt_file = tmp_path / "full.jsonl"
    prompt_file.write_text("\n".join(json.dumps(row) for row in rows))
    digest = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    with (
        patch("openbench.datasets.ojbench._prompt_path", return_value=prompt_file),
        patch("openbench.datasets.ojbench.PROMPT_COUNT", 2),
        patch("openbench.datasets.ojbench.PROMPT_SHA256", digest),
        patch.dict(os.environ, {"OPENBENCH_OJBENCH_DATA_DIR": str(tmp_path / "data")}),
    ):
        dataset = get_ojbench_dataset("python")
    assert len(dataset) == 1


def test_problem_download_uses_pinned_allow_pattern(tmp_path: Path):
    problem = tmp_path / "NOI" / "loj-2083"

    def fake_download(**kwargs):
        assert kwargs["allow_patterns"] == ["NOI/loj-2083/**"]
        problem.mkdir(parents=True)
        (problem / "init.yml").write_text("archive: tests.zip")
        (problem / "tests.zip").write_bytes(b"zip")

    with (
        patch.dict(os.environ, {"OPENBENCH_OJBENCH_DATA_DIR": str(tmp_path)}),
        patch(
            "openbench.datasets.ojbench.snapshot_download", side_effect=fake_download
        ),
    ):
        assert ensure_problem_data("NOI", "loj-2083") == problem


def test_partial_verdict_matches_official_rounding():
    results = [{"readable_main_code": "AC"}] * 7 + [{"readable_main_code": "WA"}]
    assert _partial_verdict("WA", results, 1 / 8) == "AC"
    assert _partial_verdict("WA", results, 1) == "WA"


def test_runner_uses_canonical_problem_id_without_details():
    captured: dict[str, object] = {}

    def fake_judge_entry(entry, use_tqdm):
        captured.update(entry)
        assert use_tqdm is False
        return "WA", [{"readable_main_code": "WA"}]

    fake_ojbench = SimpleNamespace(
        init=lambda **kwargs: None, judge_entry=fake_judge_entry
    )
    with (
        patch.dict(sys.modules, {"ojbench": fake_ojbench}),
        patch.dict(
            os.environ,
            {
                "OJBENCH_CONFIG_PATH": "/config.yaml",
                "OJBENCH_RUNTIME_PATH": "/runtime.yaml",
            },
        ),
    ):
        result = evaluate(
            {
                "record_id": 2083,
                "problem_id": "loj-2083",
                "language": "python",
                "completion": "print(0)",
            }
        )
    assert captured["id"] == "loj-2083"
    assert result["verdict"] == "WA"
    assert "results" not in result


def test_tasks_use_eight_samples_and_docker():
    dataset = MemoryDataset([Sample(input="x", target="")])
    with patch(
        "openbench.evals.ojbench.ojbench.get_ojbench_dataset", return_value=dataset
    ):
        python_task = ojbench_python.__wrapped__()
        cpp_task = ojbench_cpp.__wrapped__()
    assert python_task.epochs == 8
    assert cpp_task.epochs == 8
    assert python_task.sandbox.type == "docker"


def test_compose_declares_required_dmoj_boundary():
    compose = Path(__file__).parents[1] / "src/openbench/evals/ojbench/compose.yaml"
    service = yaml.safe_load(compose.read_text())["services"]["default"]
    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["cap_add"] == ["SYS_PTRACE"]
    assert service["security_opt"] == ["no-new-privileges:true"]


def test_dockerfile_pins_bookworm_for_validator_compatibility():
    dockerfile = (
        Path(__file__).parents[1] / "src/openbench/evals/ojbench/Dockerfile"
    ).read_text()
    assert dockerfile.count("python:3.11-slim-bookworm@sha256:") == 2
    assert "python:3.11-slim@sha256:" not in dockerfile


def test_registry_entries():
    assert BENCHMARKS["ojbench_python"].function_name == "ojbench_python"
    assert BENCHMARKS["ojbench_cpp"].function_name == "ojbench_cpp"
    assert BENCHMARKS["ojbench_python"].is_alpha is True
    assert BENCHMARKS["ojbench_cpp"].is_alpha is True


def test_prompt_checksum_literal_is_sha256():
    assert len(PROMPT_SHA256) == 64
