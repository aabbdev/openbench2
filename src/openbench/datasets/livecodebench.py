"""Dataset loader for LiveCodeBench code generation release v6."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

DATASET_REPOSITORY = "livecodebench/code_generation_lite"
DATASET_REVISION = "819a13a5c0347c1bd2f5600a35bc9ac9695d461b"
RELEASE_VERSION = "release_v6"
RELEASE_FILES = (
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

STDIN_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)

STARTER_FORMAT = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)


def _release_path(filename: str, *, local_files_only: bool = False) -> Path:
    if filename not in RELEASE_FILES:
        raise ValueError(f"Unknown LiveCodeBench release file: {filename}")
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_files_only=local_files_only,
        )
    )


def _release_paths() -> list[tuple[str, Path]]:
    """Cache and return the immutable release-v6 source shards."""

    return [(filename, _release_path(filename)) for filename in RELEASE_FILES]


def load_livecodebench_test_fields(
    metadata: dict[str, Any],
) -> tuple[str, str, str]:
    """Load one sample's test payloads from its immutable shard reference."""

    filename = metadata["source_file"]
    offset = metadata["source_offset"]
    length = metadata["source_length"]
    if not isinstance(filename, str):
        raise TypeError("LiveCodeBench source_file must be a string")
    if not isinstance(offset, int) or not isinstance(length, int):
        raise TypeError("LiveCodeBench source offsets must be integers")

    path = _release_path(filename, local_files_only=True)
    with path.open("rb") as shard:
        shard.seek(offset)
        record = json.loads(shard.read(length))
    if record["question_id"] != metadata["source_question_id"]:
        raise ValueError("LiveCodeBench source reference resolved to the wrong record")
    return (
        record["public_test_cases"],
        record["private_test_cases"],
        record["metadata"],
    )


def _parse_date(value: str | datetime) -> datetime:
    """Parse the ISO date format used by LiveCodeBench records."""

    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _format_prompt(question: str, starter_code: str) -> str:
    """Format the generic official LiveCodeBench code-generation prompt."""

    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {STARTER_FORMAT}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {STDIN_FORMAT}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def record_to_sample(
    start_date: str | None = None,
    end_date: str | None = None,
) -> Callable[[dict[str, Any]], Sample | list[Sample]]:
    """Create a converter for official LiveCodeBench release-v6 records.

    Date boundaries are inclusive, matching the official runner. Returning an
    empty list allows Inspect to filter records before constructing the dataset.
    """

    parsed_start = _parse_date(start_date) if start_date else None
    parsed_end = _parse_date(end_date) if end_date else None
    if (
        parsed_start is not None
        and parsed_end is not None
        and parsed_start > parsed_end
    ):
        raise ValueError("start_date must not be after end_date")

    def _record_to_sample(record: dict[str, Any]) -> Sample | list[Sample]:
        contest_date = _parse_date(record["contest_date"])
        if parsed_start is not None and contest_date < parsed_start:
            return []
        if parsed_end is not None and contest_date > parsed_end:
            return []

        starter_code = record.get("starter_code") or ""
        return Sample(
            id=record["question_id"],
            input=[
                ChatMessageSystem(content=SYSTEM_PROMPT),
                ChatMessageUser(
                    content=_format_prompt(record["question_content"], starter_code)
                ),
            ],
            target="",
            metadata={
                "question_title": record["question_title"],
                "platform": record["platform"],
                "contest_id": record["contest_id"],
                "contest_date": contest_date.isoformat(),
                "difficulty": record["difficulty"],
                "starter_code": starter_code,
                "public_test_cases": record["public_test_cases"],
                "private_test_cases": record["private_test_cases"],
                "test_metadata": record["metadata"],
                "release_version": RELEASE_VERSION,
                "dataset_revision": DATASET_REVISION,
            },
        )

    return _record_to_sample


def get_livecodebench_v6_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dataset:
    """Load the cumulative 1,055-problem LiveCodeBench release v6.

    The official Hugging Face repository uses an executable loading script.
    Loading the six immutable JSONL shards through the generic JSON builder
    avoids remote-code execution while preserving release-v6 composition.
    """

    converter = record_to_sample(start_date=start_date, end_date=end_date)
    samples: list[Sample] = []
    for filename, path in _release_paths():
        with path.open("rb") as shard:
            while line := shard.readline():
                offset = shard.tell() - len(line)
                converted = converter(json.loads(line))
                if isinstance(converted, Sample):
                    metadata = converted.metadata
                    if metadata is None:
                        raise ValueError("LiveCodeBench sample metadata is required")
                    metadata.pop("public_test_cases")
                    metadata.pop("private_test_cases")
                    metadata.pop("test_metadata")
                    metadata.update(
                        {
                            "source_file": filename,
                            "source_offset": offset,
                            "source_length": len(line),
                            "source_question_id": converted.id,
                        }
                    )
                    samples.append(converted)
                else:
                    samples.extend(converted)
    samples.sort(key=lambda sample: str(sample.id))
    return MemoryDataset(
        samples=samples,
        name="livecodebench_v6",
        location=DATASET_REPOSITORY,
    )
