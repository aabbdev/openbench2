"""Pinned loader for the official OJBench prompt release."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download, snapshot_download
from inspect_ai.dataset import MemoryDataset, Sample
from platformdirs import user_cache_dir

DATASET_REPOSITORY = "He-Ren/OJBench_testdata"
DATASET_REVISION = "61cf9986f22c25d08e1657b03742124099c74353"
PROMPT_FILE = "prompts/full.jsonl"
PROMPT_SHA256 = "bcc8c94eb1fefb856355aa8b5a3e20cc0a2112f5436c5d83ab686edb417bce2c"
PROMPT_COUNT = 464


def problem_cache_dir() -> Path:
    """Return the host directory mounted read-only into the judge."""

    configured = os.environ.get("OPENBENCH_OJBENCH_DATA_DIR")
    path = (
        Path(configured).expanduser()
        if configured
        else Path(user_cache_dir("openbench")) / "ojbench" / "problems"
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "NOI").mkdir(exist_ok=True)
    (path / "ICPC").mkdir(exist_ok=True)
    os.environ["OPENBENCH_OJBENCH_DATA_DIR"] = str(path.resolve())
    return path.resolve()


def _prompt_path() -> Path:
    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            filename=PROMPT_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PROMPT_SHA256:
        raise ValueError(
            f"OJBench prompt checksum mismatch: expected {PROMPT_SHA256}, got {digest}"
        )
    return path


def canonical_problem_id(dataset: str, record_id: int | str) -> tuple[str, str]:
    """Map prompt identifiers to the canonical problem directory."""

    if dataset == "NOI" and isinstance(record_id, int):
        return "NOI", f"loj-{record_id}"
    if dataset == "icpc" and isinstance(record_id, str):
        if re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
            return "ICPC", record_id
    raise ValueError(f"Invalid OJBench problem identity: {dataset}/{record_id!r}")


def ensure_problem_data(dataset: str, problem_id: str) -> Path:
    """Download one immutable problem package into the mounted host cache."""

    if dataset not in {"NOI", "ICPC"}:
        raise ValueError(f"Invalid OJBench dataset: {dataset}")
    pattern = r"loj-[0-9]+" if dataset == "NOI" else r"[A-Za-z0-9_-]+"
    if re.fullmatch(pattern, problem_id) is None:
        raise ValueError(f"Invalid OJBench problem id: {problem_id}")

    root = problem_cache_dir()
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[f"{dataset}/{problem_id}/**"],
        local_dir=root,
    )
    problem = root / dataset / problem_id
    if not (problem / "init.yml").is_file():
        raise FileNotFoundError(f"Missing OJBench init.yml for {dataset}/{problem_id}")
    if not any(problem.glob("*.zip")):
        raise FileNotFoundError(
            f"Missing OJBench test archive for {dataset}/{problem_id}"
        )
    return problem


def record_to_sample(record: dict[str, object]) -> Sample:
    dataset = str(record["dataset"])
    language = str(record["language"])
    if language not in {"python", "cpp"}:
        raise ValueError(f"Unsupported OJBench language: {language}")
    raw_id = record["id"]
    if not isinstance(raw_id, (int, str)):
        raise TypeError("OJBench id must be an integer or string")
    canonical_dataset, problem_id = canonical_problem_id(dataset, raw_id)
    return Sample(
        id=f"{problem_id}:{language}",
        input=str(record["prompt"]),
        target="",
        metadata={
            "record_id": raw_id,
            "problem_id": problem_id,
            "dataset": canonical_dataset,
            "language": language,
            "difficulty": str(record["difficulty"]),
            "dataset_revision": DATASET_REVISION,
        },
    )


def get_ojbench_dataset(language: Literal["python", "cpp"]) -> MemoryDataset:
    """Load one official 232-problem language track."""

    if language not in {"python", "cpp"}:
        raise ValueError("OJBench language must be 'python' or 'cpp'")
    problem_cache_dir()
    rows = [json.loads(line) for line in _prompt_path().read_text().splitlines()]
    if len(rows) != PROMPT_COUNT:
        raise ValueError(f"OJBench expected {PROMPT_COUNT} prompts, got {len(rows)}")
    samples = [record_to_sample(row) for row in rows if row["language"] == language]
    if len(samples) != PROMPT_COUNT // 2:
        raise ValueError(f"OJBench {language} expected 232 prompts, got {len(samples)}")
    return MemoryDataset(
        samples=samples,
        name=f"ojbench_{language}",
        location=DATASET_REPOSITORY,
    )
