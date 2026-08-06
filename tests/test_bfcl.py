"""Tests for the BFCL v4 single-turn integration."""

from typing import Any, cast
from unittest.mock import patch

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from openbench.config import BENCHMARKS
from openbench.datasets.bfcl import (
    BFCL_REVISION,
    get_bfcl_v4_agentic_dataset,
    get_bfcl_v4_agentic_live_dataset,
    get_bfcl_v4_multi_turn_dataset,
    get_bfcl_v4_single_turn_dataset,
)
from openbench.evals.bfcl import bfcl_v4_agentic_live, bfcl_v4_single_turn
from openbench.scorers.bfcl import bfcl_v4_scorer


QUESTION = {
    "id": "simple_python_0",
    "question": [[{"role": "user", "content": "What is 1 + 2?"}]],
    "function": [
        {
            "name": "math.add",
            "description": "Add two integers.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        }
    ],
}
ANSWER = {
    "id": "simple_python_0",
    "ground_truth": [{"math.add": {"a": [1], "b": [2]}}],
}


def test_bfcl_dataset_preserves_tools_answers_and_revision() -> None:
    def fake_load(path: str):
        return [ANSWER] if path.startswith("possible_answer/") else [QUESTION]

    with (
        patch("openbench.datasets.bfcl._load_jsonl", side_effect=fake_load),
        patch.dict("openbench.datasets.bfcl._COUNTS", {"simple_python": 1}),
    ):
        sample = list(get_bfcl_v4_single_turn_dataset(["simple_python"]))[0]

    assert sample.metadata is not None
    assert sample.id == "simple_python_0"
    assert sample.metadata["category"] == "simple_python"
    assert sample.metadata["bfcl_revision"] == BFCL_REVISION
    assert sample.metadata["functions"][0]["name"] == "math.add"
    assert sample.metadata["expected_calls"] == ANSWER["ground_truth"]


def test_bfcl_task_uses_deterministic_native_tool_generation() -> None:
    dataset = MemoryDataset([Sample(input="question", target="")])
    with patch(
        "openbench.evals.bfcl.get_bfcl_v4_single_turn_dataset",
        return_value=dataset,
    ):
        task_factory = cast(Any, bfcl_v4_single_turn)
        task = task_factory.__wrapped__(["simple_python"])

    assert task.config.temperature == 0
    assert task.config.parallel_tool_calls is True
    assert task.metadata["official_overall_score"] is False


def test_bfcl_live_task_selects_network_disabled_snapshot_replay() -> None:
    dataset = MemoryDataset(
        [Sample(input="workflow", target="", metadata={"category": "memory_kv"})]
    )
    with patch(
        "openbench.evals.bfcl.get_bfcl_v4_agentic_live_dataset",
        return_value=dataset,
    ):
        task_factory = cast(Any, bfcl_v4_agentic_live)
        task = task_factory.__wrapped__(["memory_kv"], web_snapshot_dir="snapshots")

    assert str(task.sandbox.config).endswith("compose.replay.yaml")
    assert dataset[0].metadata is not None
    assert dataset[0].metadata["web_snapshot_dir"] == "snapshots"


def test_bfcl_registry_entry_is_explicitly_single_turn() -> None:
    metadata = BENCHMARKS["bfcl_v4_single_turn"]
    assert metadata.function_name == "bfcl_v4_single_turn"
    assert "Single-Turn" in metadata.name
    assert BENCHMARKS["bfcl_v4_multi_turn"].function_name == "bfcl_v4_multi_turn"
    assert (
        BENCHMARKS["bfcl_v4_agentic_offline"].function_name == "bfcl_v4_agentic_offline"
    )
    assert BENCHMARKS["bfcl_v4_agentic_live"].function_name == "bfcl_v4_agentic_live"
    assert BENCHMARKS["bfcl_v4_offline"].function_name == "bfcl_v4_offline"


def test_bfcl_multi_turn_dataset_holds_out_functions() -> None:
    question = {
        "id": "multi_turn_miss_func_0",
        "question": [[{"role": "user", "content": "Add numbers"}], []],
        "initial_config": {},
        "involved_classes": ["MathAPI"],
        "missed_function": {"1": ["add"]},
    }
    answer = {"id": question["id"], "ground_truth": [[], ["add(a=1,b=2)"]]}

    def fake_load(path: str):
        if path.startswith("possible_answer/"):
            return [answer]
        if path.startswith("multi_turn_func_doc/"):
            function = cast(dict[str, Any], QUESTION["function"][0])
            return [{**function, "name": "add"}]
        return [question]

    with (
        patch("openbench.datasets.bfcl._load_jsonl", side_effect=fake_load),
        patch("openbench.datasets.bfcl._MULTI_TURN_COUNT", 1),
    ):
        sample = list(get_bfcl_v4_multi_turn_dataset(["multi_turn_miss_func"]))[0]

    assert sample.metadata is not None
    assert sample.metadata["functions"] == []
    assert sample.metadata["missed_functions"]["1"][0]["name"] == "add"


def test_bfcl_agentic_dataset_expands_backends_and_web_modes() -> None:
    memory_question = {
        "id": "memory_0-customer-0",
        "question": [[{"role": "user", "content": "My name?"}]],
    }
    memory_answer = {
        "id": memory_question["id"],
        "ground_truth": ["Michael"],
        "source": "My name is Michael.",
    }
    web_question = {
        "id": "web_search_0",
        "question": [[{"role": "user", "content": "Who?"}]],
    }
    web_answer = {
        "id": web_question["id"],
        "ground_truth": ["Ada"],
        "source": [{"subquestion": "Who?", "answer": "Ada", "source": "u"}],
    }

    def fake_load(path: str):
        if "multi_turn_func_doc" in path:
            return QUESTION["function"]
        if "memory" in path:
            return (
                [memory_answer]
                if path.startswith("possible_answer/")
                else [memory_question]
            )
        return [web_answer] if path.startswith("possible_answer/") else [web_question]

    with patch("openbench.datasets.bfcl._load_jsonl", side_effect=fake_load):
        dataset = get_bfcl_v4_agentic_dataset(
            ["memory_kv", "web_search_base", "web_search_no_snippet"]
        )

    assert len(dataset) == 3
    metadata = [cast(dict[str, Any], sample.metadata) for sample in dataset]
    assert [item["category"] for item in metadata] == [
        "memory_kv",
        "web_search_base",
        "web_search_no_snippet",
    ]


def test_bfcl_live_agentic_dataset_groups_model_dependent_workflows() -> None:
    memory_question = {
        "id": "memory_0-customer-0",
        "scenario": "customer",
        "question": [[{"role": "user", "content": "My name?"}]],
    }
    memory_answer = {
        "id": memory_question["id"],
        "ground_truth": ["Michael"],
    }
    prerequisite = {
        "id": "memory_prereq_0-customer-0",
        "scenario": "customer",
        "question": [[{"role": "user", "content": "My name is Michael."}]],
    }
    web_question = {
        "id": "web_search_0",
        "question": [[{"role": "user", "content": "Who?"}]],
    }
    web_answer = {"id": web_question["id"], "ground_truth": ["Ada"]}

    def fake_load(path: str):
        if "multi_turn_func_doc" in path:
            return QUESTION["function"]
        if "memory_prereq_conversation" in path:
            return [prerequisite]
        if path == "BFCL_v4_memory.json":
            return [memory_question]
        if path == "possible_answer/BFCL_v4_memory.json":
            return [memory_answer]
        if path == "BFCL_v4_web_search.json":
            return [web_question]
        return [web_answer]

    with (
        patch("openbench.datasets.bfcl._load_jsonl", side_effect=fake_load),
        patch("openbench.datasets.bfcl.MEMORY_SCENARIOS", ("customer",)),
    ):
        dataset = get_bfcl_v4_agentic_live_dataset(["memory_kv", "web_search_base"])

    assert len(dataset) == 2
    memory_metadata = cast(dict[str, Any], dataset[0].metadata)
    web_metadata = cast(dict[str, Any], dataset[1].metadata)
    assert memory_metadata["case_count"] == 1
    assert memory_metadata["workflow"]["scenarios"][0]["prerequisites"] == [
        prerequisite
    ]
    assert web_metadata["case_count"] == 1


@pytest.mark.asyncio
async def test_bfcl_scorer_reads_native_tool_calls() -> None:
    output = ModelOutput.from_message(
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function="math_add",
                    arguments={"a": 1, "b": 2},
                )
            ],
        ),
        stop_reason="tool_calls",
    )
    state = TaskState(
        model=ModelName("mock/test"),
        sample_id="simple_python_0",
        epoch=1,
        input="What is 1 + 2?",
        messages=[],
        output=output,
        metadata={
            "category": "simple_python",
            "functions": QUESTION["function"],
            "expected_calls": ANSWER["ground_truth"],
            "tool_name_mapping": {"math_add": "math.add"},
        },
    )
    result = await bfcl_v4_scorer()(state, Target(""))
    assert result is not None
    assert result.value == CORRECT
