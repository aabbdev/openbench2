import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from inspect_ai.dataset import MemoryDataset, Sample

from openbench.config import BENCHMARKS
from openbench.datasets.evalplus import (
    RELEASES,
    _ensure_release,
    get_evalplus_dataset,
    load_evalplus_record,
)
from openbench.evals.evalplus.evalplus import humaneval_plus, mbpp_plus
from openbench.scorers.evalplus import extract_python
from openbench.scorers.evalplus_runner import evaluate


def _record(task_id: str = "HumanEval/0") -> dict:
    return {
        "task_id": task_id,
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "canonical_solution": "    return a + b\n",
        "base_input": [[1, 2]],
        "plus_input": [[-4, 9]],
        "atol": 0,
    }


def test_evalplus_dataset_keeps_tests_out_of_metadata(tmp_path: Path):
    path = tmp_path / "human.jsonl"
    path.write_text(json.dumps(_record()) + "\n")
    with (
        patch("openbench.datasets.evalplus._ensure_release", return_value=path),
        patch.dict(RELEASES["humaneval"], {"count": 1}),
    ):
        sample = list(get_evalplus_dataset("humaneval"))[0]

    assert sample.metadata is not None
    assert "base_input" not in sample.metadata
    assert "plus_input" not in sample.metadata
    assert load_evalplus_record(sample.metadata)["plus_input"] == [[-4, 9]]


def test_evalplus_runner_accepts_correct_and_rejects_wrong():
    payload = {
        "dataset": "humaneval",
        **_record(),
        "code": "def add(a, b):\n    return a + b\n",
    }
    assert evaluate(payload)["passed"] is True

    payload["code"] = "def add(a, b):\n    return a - b\n"
    assert evaluate(payload)["passed"] is False


def test_evalplus_runs_base_and_plus_in_fresh_processes():
    payload = {
        "dataset": "humaneval",
        **_record(),
        "code": (
            "calls = 0\n"
            "def add(a, b):\n"
            "    global calls\n"
            "    calls += 1\n"
            "    return a + b if calls == 1 else None\n"
        ),
    }
    assert evaluate(payload)["passed"] is True


def test_evalplus_mbpp_special_oracles():
    not_none = {
        "dataset": "mbpp",
        "task_id": "Mbpp/1",
        "prompt": "",
        "entry_point": "check_str",
        "canonical_solution": (
            "import re\ndef check_str(value):\n    return re.match(r'a', value)\n"
        ),
        "base_input": [["abc"]],
        "plus_input": [],
        "atol": 0,
        "code": "def check_str(value):\n    return 'accepted'\n",
    }
    assert evaluate(not_none)["passed"] is True

    set_equivalent = {
        "dataset": "mbpp",
        "task_id": "Mbpp/2",
        "prompt": "",
        "entry_point": "similar_elements",
        "canonical_solution": "def similar_elements(value):\n    return [1]\n",
        "base_input": [[[0]]],
        "plus_input": [],
        "atol": 0,
        "code": "def similar_elements(value):\n    return [1.0]\n",
    }
    assert evaluate(set_equivalent)["passed"] is True


def test_evalplus_extracts_last_python_fence():
    assert (
        extract_python("text\n```python\ndef f():\n    pass\n```")
        == "def f():\n    pass"
    )


def test_evalplus_repairs_corrupt_expanded_cache(tmp_path: Path):
    content = (json.dumps(_record()) + "\n").encode()
    compressed = gzip.compress(content)
    release = {
        "version": "test",
        "url": "https://invalid.example/test.jsonl.gz",
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "expanded_sha256": hashlib.sha256(content).hexdigest(),
        "count": 1,
    }
    compressed_path = tmp_path / "humaneval-test.jsonl.gz"
    expanded_path = tmp_path / "humaneval-test.jsonl"
    compressed_path.write_bytes(compressed)
    expanded_path.write_text("corrupt")
    with (
        patch("openbench.datasets.evalplus._cache_dir", return_value=tmp_path),
        patch.dict(RELEASES, {"humaneval": release}),
    ):
        assert _ensure_release("humaneval").read_bytes() == content


def test_evalplus_tasks_use_docker_sandbox():
    dataset = MemoryDataset([Sample(input="x", target="x")])
    with patch(
        "openbench.evals.evalplus.evalplus.get_evalplus_dataset",
        return_value=dataset,
    ):
        human = humaneval_plus.__wrapped__(epochs=1)
        mbpp = mbpp_plus.__wrapped__(epochs=1)

    assert human.sandbox.type == "docker"
    assert mbpp.sandbox.type == "docker"


def test_evalplus_registry_entries():
    assert BENCHMARKS["humaneval_plus"].function_name == "humaneval_plus"
    assert BENCHMARKS["mbpp_plus"].function_name == "mbpp_plus"
