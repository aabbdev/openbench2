"""Tests for the pinned public LiveBench coding adapter."""

from datetime import datetime
from unittest.mock import patch

import pyarrow as pa
from inspect_ai.dataset import MemoryDataset, Sample

from openbench.datasets.livebench import (
    DATASET_REVISION,
    RELEASE_DATE,
    _cached_test_rows,
    get_livebench_coding_dataset,
    load_livebench_test_fields,
    record_to_sample,
)
from openbench.evals.livebench import livebench_coding_2024_11_25
from openbench.scorers.livecodebench import extract_livebench_code


def _record(**overrides):
    record = {
        "question_id": "live-add",
        "turns": ["Write a Python program that adds two integers."],
        "question_title": "Add",
        "task": "LCB_generation",
        "livebench_release_date": datetime(2024, 6, 24),
        "livebench_removal_date": datetime(2025, 4, 2),
        "partial_solution": "",
    }
    record.update(overrides)
    return record


def test_record_to_sample_applies_official_release_membership():
    sample = record_to_sample(_record(), 7)
    assert isinstance(sample, Sample)
    assert sample.id == "live-add"
    assert sample.metadata["source_row"] == 7
    assert sample.metadata["release_date"] == RELEASE_DATE
    assert sample.metadata["dataset_revision"] == DATASET_REVISION

    assert (
        record_to_sample(_record(livebench_release_date=datetime(2025, 4, 2)), 0)
        is None
    )
    assert (
        record_to_sample(_record(livebench_removal_date=datetime(2024, 11, 25)), 0)
        is None
    )
    assert record_to_sample(_record(task="agentic_coding"), 0) is None


def test_dataset_keeps_hidden_tests_out_of_sample_metadata(tmp_path):
    path = tmp_path / "coding.parquet"
    index_table = pa.Table.from_pylist([_record()])
    with (
        patch("openbench.datasets.livebench._dataset_path", return_value=path),
        patch("openbench.datasets.livebench._read_table", return_value=index_table),
    ):
        dataset = get_livebench_coding_dataset()

    assert len(dataset) == 1
    assert dataset[0].input == "Write a Python program that adds two integers."
    assert "private_test_cases" not in dataset[0].metadata


def test_hidden_tests_resolve_from_pinned_source(tmp_path):
    path = tmp_path / "coding.parquet"
    test_table = pa.Table.from_pylist(
        [
            {
                "question_id": "live-add",
                "public_test_cases": "[]",
                "private_test_cases": "[]",
                "original_json": {"metadata": '{"func_name": "add"}'},
                "partial_solution": "def add(a, b):",
            }
        ]
    )
    metadata = {
        "source_row": 0,
        "source_question_id": "live-add",
    }
    _cached_test_rows.cache_clear()
    with (
        patch("openbench.datasets.livebench._dataset_path", return_value=path),
        patch("openbench.datasets.livebench._read_table", return_value=test_table),
    ):
        fields = load_livebench_test_fields(metadata)
    _cached_test_rows.cache_clear()

    assert fields == ("[]", "[]", '{"func_name": "add"}', "def add(a, b):")


def test_extract_livebench_code_preserves_official_unfenced_fallback():
    assert extract_livebench_code("print(1)") == "print(1)"
    assert extract_livebench_code("```python\nprint(2)\n```") == "print(2)"
    assert extract_livebench_code("print(3)\n```") == "print(3)"


def test_task_uses_public_release_generation_defaults():
    dataset = MemoryDataset([Sample(input="question", target="")])
    with patch(
        "openbench.evals.livebench.livebench.get_livebench_coding_dataset",
        return_value=dataset,
    ):
        task = livebench_coding_2024_11_25()

    assert task.config.temperature == 0
    assert task.config.max_tokens == 4096
