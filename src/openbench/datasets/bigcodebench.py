"""Dataset loader for BigCodeBench v0.1.4."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser

Split = Literal["complete", "instruct"]
Subset = Literal["full", "hard"]

DATASET_VERSION = "v0.1.4"
DATASET_REPOSITORIES: dict[Subset, str] = {
    "full": "bigcode/bigcodebench",
    "hard": "bigcode/bigcodebench-hard",
}
DATASET_REVISIONS: dict[Subset, str] = {
    "full": "b74c0d0bf70d2c0bc459be537895cca163007f1a",
    "hard": "298d2cc7b96612e15e47313c3603ee124cee0c1f",
}

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)


def validate_bigcodebench_options(split: str, subset: str) -> tuple[Split, Subset]:
    """Validate BigCodeBench split/subset names."""

    if split not in {"complete", "instruct"}:
        raise ValueError("split must be one of: complete, instruct")
    if subset not in {"full", "hard"}:
        raise ValueError("subset must be one of: full, hard")
    return split, subset  # type: ignore[return-value]


def format_bigcodebench_prompt(prompt: str, split: Split) -> str:
    """Format prompts like BigCodeBench's API/chat backend."""

    prompt = prompt.strip()
    if split == "complete":
        return f"{INSTRUCTION_PREFIX}\n```\n{prompt}\n```\n"
    return f"{INSTRUCTION_PREFIX}\n{prompt}\n"


@lru_cache(maxsize=2)
def _bigcodebench_records(subset: Subset) -> dict[str, dict[str, Any]]:
    repository = DATASET_REPOSITORIES[subset]
    dataset = load_dataset(
        repository,
        split=DATASET_VERSION,
        revision=DATASET_REVISIONS[subset],
    )
    return {str(record["task_id"]): dict(record) for record in dataset}


def load_bigcodebench_execution_fields(
    metadata: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    """Resolve test and calibration fields from the immutable HF dataset."""

    subset = metadata["subset"]
    if subset not in DATASET_REPOSITORIES:
        raise ValueError(f"Unknown BigCodeBench subset: {subset}")
    if metadata.get("dataset_version") != DATASET_VERSION:
        raise ValueError("BigCodeBench dataset_version mismatch")
    if metadata.get("dataset_revision") != DATASET_REVISIONS[subset]:
        raise ValueError("BigCodeBench dataset_revision mismatch")

    task_id = metadata["task_id"]
    record = _bigcodebench_records(subset)[task_id]
    return (
        record["code_prompt"],
        record["test"],
        record["entry_point"],
        record["complete_prompt"],
        record["canonical_solution"],
    )


def record_to_sample(record: dict[str, Any], split: Split, subset: Subset) -> Sample:
    """Convert one official BigCodeBench record into an Inspect sample."""

    prompt_key = f"{split}_prompt"
    task_id = str(record["task_id"])
    return Sample(
        id=task_id,
        input=[
            ChatMessageUser(
                content=format_bigcodebench_prompt(record[prompt_key], split)
            )
        ],
        target="",
        metadata={
            "task_id": task_id,
            "split": split,
            "subset": subset,
            "dataset_version": DATASET_VERSION,
            "dataset_revision": DATASET_REVISIONS[subset],
        },
    )


def get_bigcodebench_dataset(
    split: str = "instruct",
    subset: str = "full",
    limit: int | None = None,
) -> Dataset:
    """Load the official BigCodeBench dataset as Inspect samples."""

    resolved_split, resolved_subset = validate_bigcodebench_options(split, subset)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    records = list(_bigcodebench_records(resolved_subset).values())
    samples = [
        record_to_sample(record, split=resolved_split, subset=resolved_subset)
        for record in records[:limit]
    ]
    return MemoryDataset(
        samples=samples,
        name=f"bigcodebench_{resolved_split}_{resolved_subset}",
        location=DATASET_REPOSITORIES[resolved_subset],
    )
