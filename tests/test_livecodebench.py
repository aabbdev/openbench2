"""Tests for the LiveCodeBench v6 adapter and execution runner."""

import base64
import json
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
from inspect_ai.dataset import MemoryDataset, Sample

from openbench.datasets.livecodebench import (
    DATASET_REVISION,
    RELEASE_VERSION,
    SYSTEM_PROMPT,
    get_livecodebench_v6_dataset,
    load_livecodebench_test_fields,
    record_to_sample,
)
from openbench.evals.livecodebench import livecodebench_v6
from openbench.scorers.livecodebench import decode_test_cases, extract_code
from openbench.scorers.livecodebench_runner import evaluate_submission, outputs_match


def _record(**overrides):
    record = {
        "question_title": "Add",
        "question_content": "Read two integers and print their sum.",
        "platform": "codeforces",
        "question_id": "cf-add",
        "contest_id": "1",
        "contest_date": "2025-04-01T00:00:00",
        "starter_code": "",
        "difficulty": "easy",
        "public_test_cases": json.dumps(
            [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
        ),
        "private_test_cases": json.dumps([]),
        "metadata": json.dumps({}),
    }
    record.update(overrides)
    return record


def test_record_to_sample_preserves_release_and_hidden_tests():
    sample = record_to_sample()(_record())
    assert isinstance(sample, Sample)
    assert sample.id == "cf-add"
    assert "Read two integers" in str(sample.input)
    assert sample.metadata["release_version"] == RELEASE_VERSION
    assert sample.metadata["dataset_revision"] == DATASET_REVISION
    assert sample.metadata["private_test_cases"] == "[]"
    assert sample.input[0].content == SYSTEM_PROMPT
    assert sample.input[1].content == (
        "### Question:\nRead two integers and print their sum.\n\n"
        "### Format: Read the inputs from stdin solve the problem and write the "
        "answer to stdout (do not directly test on the sample inputs). Enclose "
        "your code within delimiters as follows. Ensure that when the python "
        "program runs, it reads the inputs, runs the algorithm and writes output "
        "to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )


def test_record_to_sample_applies_inclusive_date_filter():
    converter = record_to_sample(
        start_date="2025-04-01T00:00:00",
        end_date="2025-04-01T00:00:00",
    )
    assert converter(_record()) != []
    assert converter(_record(contest_date="2025-03-31T23:59:59")) == []


def test_record_to_sample_includes_starter_code_contract():
    sample = record_to_sample()(_record(starter_code="class Solution:\n    pass"))
    assert isinstance(sample, Sample)
    assert "starter code" in str(sample.input)
    assert "class Solution" in str(sample.input)


def test_dataset_parses_cached_shards_and_sorts_without_arrow_cache(tmp_path):
    later = _record(question_id="z", contest_date="2025-04-02T00:00:00")
    earlier = _record(question_id="a")
    shard = tmp_path / "test.jsonl"
    shard.write_text("\n".join([json.dumps(later), json.dumps(earlier)]))
    with patch(
        "openbench.datasets.livecodebench._release_paths",
        return_value=[("test.jsonl", shard)],
    ) as release_paths:
        dataset = get_livecodebench_v6_dataset()

    assert [sample.id for sample in dataset] == ["a", "z"]
    assert dataset[1].metadata["contest_date"] == "2025-04-02T00:00:00"
    assert "private_test_cases" not in dataset[1].metadata
    release_paths.assert_called_once_with()

    with patch(
        "openbench.datasets.livecodebench._release_path",
        return_value=shard,
    ):
        public, private, metadata = load_livecodebench_test_fields(dataset[1].metadata)
    assert json.loads(public)[0]["output"] == "3\n"
    assert private == "[]"
    assert metadata == "{}"


def test_task_uses_official_generic_sampling_configuration():
    dataset = MemoryDataset([Sample(input="question", target="")])
    with patch(
        "openbench.evals.livecodebench.livecodebench.get_livecodebench_v6_dataset",
        return_value=dataset,
    ):
        task = livecodebench_v6()

    assert task.epochs == 10
    assert task.config.temperature == 0.2
    assert task.config.top_p == 0.95
    assert task.config.max_tokens == 2000
    assert task.config.stop_seqs == ["###"]


def test_decode_test_cases_supports_plain_and_compressed_payloads():
    tests = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
    assert decode_test_cases(json.dumps(tests)) == tests

    compressed = base64.b64encode(
        zlib.compress(pickle.dumps(json.dumps(tests)))
    ).decode("utf-8")
    assert decode_test_cases(compressed) == tests


def test_extract_code_uses_final_fence_pair_for_any_label():
    completion = "```python\nprint(1)\n```\nreasoning\n```Python\nprint(2)\n```"
    assert extract_code(completion) == "print(2)"
    assert extract_code("```py\nprint(3)\n```") == "print(3)"
    assert extract_code("print(4)") == ""


def test_outputs_match_uses_exact_decimal_tokens():
    assert outputs_match("1.0 2\n", "1.00 2.0\n")
    assert not outputs_match("1.0000000000000001\n", "1.0\n")


def test_runner_scores_standard_input_submission():
    result = evaluate_submission(
        {
            "code": "a, b = map(int, input().split())\nprint(a + b)",
            "tests": [
                {"input": "1 2\n", "output": "3\n"},
                {"input": "-2 5\n", "output": "3\n"},
            ],
            "function_name": None,
            "timeout": 2,
        }
    )
    assert result == {"passed": True, "tests_run": 2, "error": None}


def test_runner_provides_official_module_import_prelude():
    result = evaluate_submission(
        {
            "code": "print(math.isqrt(int(input())))",
            "tests": [{"input": "81\n", "output": "9\n"}],
            "function_name": None,
            "timeout": 2,
        }
    )
    assert result["passed"] is True


def test_runner_scores_functional_submission_and_failure():
    passing = evaluate_submission(
        {
            "code": "class Solution:\n    def add(self, a, b):\n        return a + b",
            "tests": [{"input": "1\n2", "output": "3"}],
            "function_name": "add",
            "timeout": 2,
        }
    )
    assert passing["passed"] is True

    failing = evaluate_submission(
        {
            "code": "class Solution:\n    def add(self, a, b):\n        return a - b",
            "tests": [{"input": "1\n2", "output": "3"}],
            "function_name": "add",
            "timeout": 2,
        }
    )
    assert failing["passed"] is False
    assert failing["error"] == "wrong_answer"


def test_runner_preserves_functional_solution_state_across_tests():
    result = evaluate_submission(
        {
            "code": (
                "class Solution:\n"
                "    def __init__(self): self.total = 0\n"
                "    def add(self, value):\n"
                "        self.total += value\n"
                "        return self.total"
            ),
            "tests": [
                {"input": "1", "output": "1"},
                {"input": "2", "output": "3"},
            ],
            "function_name": "add",
            "timeout": 2,
        }
    )
    assert result["passed"] is True


def test_runner_kills_submission_process_group_on_timeout():
    started = time.monotonic()
    result = evaluate_submission(
        {
            "code": (
                "import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "time.sleep(30)"
            ),
            "tests": [{"input": "", "output": ""}],
            "function_name": None,
            "timeout": 1,
        }
    )
    assert result["error"] == "timeout"
    assert time.monotonic() - started < 5


def test_runner_deletes_payload_before_submission_starts(tmp_path):
    payload_path = tmp_path / ".openbench_livecodebench_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "code": (
                    "from pathlib import Path\n"
                    "print('leaked' if list(Path('.').glob('*payload*')) else 'safe')"
                ),
                "tests": [{"input": "", "output": "safe\n"}],
                "function_name": None,
                "timeout": 2,
            }
        )
    )
    runner = Path(__file__).parents[1] / "src/openbench/scorers/livecodebench_runner.py"
    sandbox_runner = tmp_path / ".openbench_livecodebench_runner.py"
    sandbox_runner.write_text(runner.read_text())
    completed = subprocess.run(
        [sys.executable, str(sandbox_runner), str(payload_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["passed"] is True
    assert not payload_path.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires procfs")
def test_runner_reaps_double_forked_descendants(tmp_path):
    payload_path = tmp_path / ".openbench_livecodebench_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "code": (
                    "import os, time\n"
                    "child = os.fork()\n"
                    "if child == 0:\n"
                    "    os.setsid()\n"
                    "    if os.fork() > 0: os._exit(0)\n"
                    "    time.sleep(30)\n"
                    "    os._exit(0)\n"
                    "os.waitpid(child, 0)\n"
                    "time.sleep(30)"
                ),
                "tests": [{"input": "", "output": ""}],
                "function_name": None,
                "timeout": 1,
            }
        )
    )
    runner = Path(__file__).parents[1] / "src/openbench/scorers/livecodebench_runner.py"
    sandbox_runner = tmp_path / ".openbench_livecodebench_runner.py"
    sandbox_runner.write_text(runner.read_text())

    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(sandbox_runner), str(payload_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["error"] == "timeout"
    assert time.monotonic() - started < 5
