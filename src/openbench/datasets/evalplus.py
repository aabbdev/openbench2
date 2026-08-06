"""Pinned loaders for the official EvalPlus release artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample
from platformdirs import user_cache_dir

RELEASES = {
    "humaneval": {
        "version": "v0.1.10",
        "url": "https://raw.githubusercontent.com/evalplus/humanevalplus_release/200defce9e3429d28ca215b6dd061c0f7f31c18b/HumanEvalPlus.jsonl.gz",
        "sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "expanded_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        "count": 164,
    },
    "mbpp": {
        "version": "v0.2.0",
        "url": "https://raw.githubusercontent.com/evalplus/mbppplus_release/64fc4195b858a17cdfdb3324f0baf37939144e14/MbppPlus.jsonl.gz",
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        "expanded_sha256": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
        "count": 378,
    },
}


def _cache_dir() -> Path:
    return Path(user_cache_dir("openbench")) / "evalplus"


def _ensure_release(dataset: str) -> Path:
    release = RELEASES[dataset]
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    compressed = cache_dir / f"{dataset}-{release['version']}.jsonl.gz"
    expanded = cache_dir / f"{dataset}-{release['version']}.jsonl"

    valid_cache = (
        compressed.exists()
        and hashlib.sha256(compressed.read_bytes()).hexdigest() == release["sha256"]
    )
    if not valid_cache:
        with urllib.request.urlopen(str(release["url"]), timeout=120) as response:
            content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if digest != release["sha256"]:
            raise ValueError(
                f"EvalPlus {dataset} checksum mismatch: "
                f"expected {release['sha256']}, got {digest}"
            )
        compressed.write_bytes(content)

    expanded_valid = (
        expanded.exists()
        and hashlib.sha256(expanded.read_bytes()).hexdigest()
        == release["expanded_sha256"]
    )
    if not expanded_valid:
        content = gzip.decompress(compressed.read_bytes())
        digest = hashlib.sha256(content).hexdigest()
        if digest != release["expanded_sha256"]:
            raise ValueError(
                f"Expanded EvalPlus {dataset} checksum mismatch: "
                f"expected {release['expanded_sha256']}, got {digest}"
            )
        expanded.write_bytes(content)
    return expanded


def load_evalplus_record(metadata: dict) -> dict:
    """Reload one record without putting hidden tests in Inspect's sample log."""
    path = Path(str(metadata["source_file"]))
    with path.open("rb") as source:
        source.seek(int(metadata["source_offset"]))
        line = source.read(int(metadata["source_length"]))
    return json.loads(line)


def _instruction(prompt: str) -> str:
    return (
        "Please provide a self-contained Python script that solves the following "
        f"problem in a markdown code block:\n```python\n{prompt.strip()}\n```"
    )


def get_evalplus_dataset(dataset: str) -> MemoryDataset:
    """Create samples whose metadata references, but never embeds, hidden tests."""
    if dataset not in RELEASES:
        raise ValueError(f"Unknown EvalPlus dataset: {dataset}")
    path = _ensure_release(dataset)
    samples: list[Sample] = []
    with path.open("rb") as source:
        while line := source.readline():
            offset = source.tell() - len(line)
            record = json.loads(line)
            samples.append(
                Sample(
                    id=str(record["task_id"]),
                    input=_instruction(str(record["prompt"])),
                    target=str(record["entry_point"]),
                    metadata={
                        "dataset": dataset,
                        "entry_point": str(record["entry_point"]),
                        "prompt": str(record["prompt"]),
                        "source_file": str(path),
                        "source_offset": offset,
                        "source_length": len(line),
                        "release_version": RELEASES[dataset]["version"],
                        "release_sha256": RELEASES[dataset]["sha256"],
                    },
                )
            )
    if len(samples) != RELEASES[dataset]["count"]:
        raise ValueError(
            f"EvalPlus {dataset} expected {RELEASES[dataset]['count']} records, "
            f"got {len(samples)}"
        )
    return MemoryDataset(samples=samples, name=f"{dataset}plus")
