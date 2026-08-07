"""Tests for the BigCodeBench adapter."""

from unittest.mock import patch

import pytest
from inspect_ai.dataset import MemoryDataset, Sample

from openbench.datasets.bigcodebench import (
    DATASET_REVISIONS,
    DATASET_VERSION,
    INSTRUCTION_PREFIX,
    format_bigcodebench_prompt,
    get_bigcodebench_dataset,
    load_bigcodebench_execution_fields,
    record_to_sample,
)
from openbench.evals.bigcodebench import bigcodebench
from openbench.evals.bigcodebench.bigcodebench import compose_path_for_runtime
from openbench.scorers.bigcodebench import build_bigcodebench_payload


def _record(**overrides):
    record = {
        "task_id": "BigCodeBench/0",
        "complete_prompt": "def task_func():\n    pass",
        "instruct_prompt": "Create task_func.",
        "canonical_solution": "def task_func():\n    return 1",
        "code_prompt": "def task_func():",
        "test": "class TestCases: pass",
        "entry_point": "task_func",
    }
    record.update(overrides)
    return record


def test_format_bigcodebench_prompt_matches_api_backend_shape():
    assert format_bigcodebench_prompt("Do it", "instruct") == (
        f"{INSTRUCTION_PREFIX}\nDo it\n"
    )
    assert format_bigcodebench_prompt("def f(): pass", "complete") == (
        f"{INSTRUCTION_PREFIX}\n```\ndef f(): pass\n```\n"
    )


def test_record_to_sample_hides_execution_fields_from_metadata():
    sample = record_to_sample(_record(), split="instruct", subset="full")
    assert sample.id == "BigCodeBench/0"
    assert sample.input[0].content == f"{INSTRUCTION_PREFIX}\nCreate task_func.\n"
    assert sample.metadata == {
        "task_id": "BigCodeBench/0",
        "split": "instruct",
        "subset": "full",
        "dataset_version": DATASET_VERSION,
        "dataset_revision": DATASET_REVISIONS["full"],
    }


def test_get_bigcodebench_dataset_validates_options_and_limit():
    with patch(
        "openbench.datasets.bigcodebench._bigcodebench_records",
        return_value={"BigCodeBench/0": _record()},
    ):
        dataset = get_bigcodebench_dataset(split="complete", subset="hard", limit=1)

    assert isinstance(dataset, MemoryDataset)
    assert dataset.name == "bigcodebench_complete_hard"
    assert len(dataset) == 1

    with pytest.raises(ValueError, match="split"):
        get_bigcodebench_dataset(split="bad", subset="hard")
    with pytest.raises(ValueError, match="limit"):
        get_bigcodebench_dataset(limit=0)


def test_load_bigcodebench_execution_fields_resolves_from_revisioned_record():
    metadata = {
        "task_id": "BigCodeBench/0",
        "subset": "full",
        "dataset_version": DATASET_VERSION,
        "dataset_revision": DATASET_REVISIONS["full"],
    }
    with patch(
        "openbench.datasets.bigcodebench._bigcodebench_records",
        return_value={"BigCodeBench/0": _record()},
    ):
        fields = load_bigcodebench_execution_fields(metadata)

    assert fields == (
        "def task_func():",
        "class TestCases: pass",
        "task_func",
        "def task_func():\n    pass",
        "def task_func():\n    return 1",
    )


def test_task_uses_official_greedy_generation_settings():
    dataset = MemoryDataset([Sample(input="prompt", target="")])
    with patch(
        "openbench.evals.bigcodebench.bigcodebench.get_bigcodebench_dataset",
        return_value=dataset,
    ):
        task = bigcodebench(epochs=10, limit=1)

    assert task.name == "bigcodebench_instruct_full"
    assert task.epochs == 10
    assert task.config.temperature == 0
    assert task.config.top_p == 0.95
    assert task.config.max_tokens == 1280


def test_bigcodebench_runtime_selects_expected_compose_files():
    assert compose_path_for_runtime("official").name == "compose.yaml"
    assert compose_path_for_runtime("arm64").name == "compose.arm64.yaml"

    with patch(
        "openbench.evals.bigcodebench.bigcodebench.platform_machine",
        return_value="arm64",
    ):
        assert compose_path_for_runtime("auto").name == "compose.arm64.yaml"

    with patch(
        "openbench.evals.bigcodebench.bigcodebench.platform_machine",
        return_value="x86_64",
    ):
        assert compose_path_for_runtime("auto").name == "compose.yaml"

    with pytest.raises(ValueError, match="runtime"):
        compose_path_for_runtime("bad")  # type: ignore[arg-type]


def test_build_bigcodebench_payload_preserves_official_execution_fields():
    payload = build_bigcodebench_payload(
        "```python\ndef task_func(): return 1\n```",
        (
            "def task_func():",
            "class TestCases: pass",
            "task_func",
            "def task_func():\n    pass",
            "def task_func():\n    return 1",
        ),
        task_id="BigCodeBench/0",
        calibrated=True,
        min_time_limit=1,
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
    )
    assert payload["task_id"] == "BigCodeBench/0"
    assert payload["entry_point"] == "task_func"
    assert payload["calibrated"] is True
