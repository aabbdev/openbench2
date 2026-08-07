"""Dataset adapter for the public LiveBench coding release 2024-11-25."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

DATASET_REPOSITORY = "livebench/coding"
DATASET_REVISION = "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4"
DATASET_FILE = "data/test-00000-of-00001.parquet"
RELEASE_DATE = "2024-11-25"
SUPPORTED_TASKS = frozenset({"LCB_generation", "coding_completion"})

_INDEX_COLUMNS = (
    "question_id",
    "turns",
    "question_title",
    "task",
    "livebench_release_date",
    "livebench_removal_date",
    "partial_solution",
)
_TEST_COLUMNS = (
    "question_id",
    "public_test_cases",
    "private_test_cases",
    "original_json",
    "partial_solution",
)


def _dataset_path(*, local_files_only: bool = False) -> Path:
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_files_only=local_files_only,
        )
    )


def _read_table(path: Path, columns: tuple[str, ...]) -> pa.Table:
    return pq.read_table(path, columns=list(columns))


@cache
def _cached_test_rows(path: str) -> list[dict[str, Any]]:
    """Keep hidden tests in process memory, outside Inspect sample metadata."""

    return _read_table(Path(path), _TEST_COLUMNS).to_pylist()


def _iso_date(value: Any) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10]


def _is_release_member(record: dict[str, Any]) -> bool:
    released = _iso_date(record["livebench_release_date"])
    removal_value = record.get("livebench_removal_date")
    removed = _iso_date(removal_value) if removal_value else ""
    return released <= RELEASE_DATE and (not removed or removed > RELEASE_DATE)


def record_to_sample(record: dict[str, Any], row_index: int) -> Sample | None:
    """Convert one official row while applying LiveBench release semantics."""

    task_name = record["task"]
    if task_name not in SUPPORTED_TASKS or not _is_release_member(record):
        return None
    turns = record["turns"]
    if not isinstance(turns, list) or len(turns) != 1:
        raise ValueError("LiveBench coding 2024-11-25 expects one user turn")
    question_id = record["question_id"]
    return Sample(
        id=question_id,
        input=turns[0],
        target="",
        metadata={
            "question_title": record["question_title"],
            "task": task_name,
            "release_date": RELEASE_DATE,
            "dataset_revision": DATASET_REVISION,
            "source_file": DATASET_FILE,
            "source_row": row_index,
            "source_question_id": question_id,
        },
    )


def get_livebench_coding_dataset() -> Dataset:
    """Load the 128 public coding questions active in release 2024-11-25."""

    path = _dataset_path()
    records = _read_table(path, _INDEX_COLUMNS).to_pylist()
    samples = [
        sample
        for index, record in enumerate(records)
        if (sample := record_to_sample(record, index)) is not None
    ]
    samples.sort(key=lambda sample: str(sample.id))
    return MemoryDataset(
        samples=samples,
        name="livebench_coding_2024_11_25",
        location=DATASET_REPOSITORY,
    )


def load_livebench_test_fields(
    metadata: dict[str, Any],
) -> tuple[str, str, str, str]:
    """Resolve hidden tests from the immutable parquet source at score time."""

    row_index = metadata.get("source_row")
    if not isinstance(row_index, int):
        raise TypeError("LiveBench source_row must be an integer")
    path = _dataset_path(local_files_only=True)
    rows = _cached_test_rows(str(path))
    try:
        record = rows[row_index]
    except IndexError as error:
        raise ValueError(
            "LiveBench source_row is outside the pinned dataset"
        ) from error
    if record["question_id"] != metadata.get("source_question_id"):
        raise ValueError("LiveBench source reference resolved to the wrong record")
    original = record["original_json"]
    test_metadata = original.get("metadata") if isinstance(original, dict) else None
    if not isinstance(test_metadata, str):
        test_metadata = json.dumps(test_metadata or {})
    return (
        record["public_test_cases"],
        record["private_test_cases"],
        test_metadata,
        record.get("partial_solution") or "",
    )
