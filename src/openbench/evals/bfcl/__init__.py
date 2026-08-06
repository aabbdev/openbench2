"""Berkeley Function Calling Leaderboard v4 tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from openbench.datasets.bfcl import (
    AGENTIC_CATEGORIES,
    MULTI_TURN_CATEGORIES,
    SINGLE_TURN_CATEGORIES,
    get_bfcl_v4_agentic_dataset,
    get_bfcl_v4_multi_turn_dataset,
    get_bfcl_v4_single_turn_dataset,
)
from openbench.function_calling import parse_function_calls
from openbench.function_calling.schema import build_tool_definitions
from openbench.function_calling.stateful import (
    format_python_call,
    frozen_agentic_result,
)
from openbench.scorers.bfcl import (
    bfcl_v4_agentic_scorer,
    bfcl_v4_multi_turn_scorer,
    bfcl_v4_offline_scorer,
    bfcl_v4_scorer,
)

COMPOSE_PATH = (Path(__file__).parent / "compose.yaml").resolve()
MAX_STEPS = 20


@solver
def bfcl_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        functions: list[dict[str, Any]] = list(state.metadata["functions"])
        tools, mapping = build_tool_definitions(functions)
        state.tools = [definition.as_tool() for definition in tools]
        state.tool_choice = "auto"
        state.metadata["tool_name_mapping"] = mapping
        return await generate(state, tool_calls="none")

    return solve


def _append_messages(state: TaskState, messages: list[dict[str, str]]) -> None:
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "system":
            state.messages.append(ChatMessageSystem(content=content))
        else:
            state.messages.append(ChatMessageUser(content=content))


async def _sandbox_request(state: TaskState, payload: dict[str, Any]) -> dict[str, Any]:
    environment = sandbox()
    payload_path = f".openbench_bfcl_{state.uuid}.json"
    await environment.write_file(payload_path, json.dumps(payload))
    result = await environment.exec(
        ["python", "/opt/openbench/bfcl_runner.py", payload_path],
        timeout=120,
        timeout_retry=False,
    )
    if not result.success:
        raise RuntimeError("BFCL sandbox runner failed")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _append_tool_results(
    state: TaskState,
    outputs: list[str],
    native_calls: list[Any] | None,
) -> None:
    if native_calls:
        for index, call in enumerate(native_calls):
            state.messages.append(
                ChatMessageTool(
                    content=outputs[index] if index < len(outputs) else "No result",
                    tool_call_id=call.id,
                    function=call.function,
                )
            )
    else:
        state.messages.append(
            ChatMessageUser(content=f"Tool execution results: {json.dumps(outputs)}")
        )


@solver
def bfcl_multi_turn_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        functions = list(state.metadata["functions"])
        missed = dict(state.metadata.get("missed_functions", {}))
        turns = list(state.metadata["turns"])
        model_turn_calls: list[list[list[str]]] = []
        all_calls: list[str] = []

        for turn_index, turn_messages in enumerate(turns):
            if turn_index > 0:
                if str(turn_index) in missed:
                    functions.extend(missed[str(turn_index)])
                    _append_messages(
                        state,
                        [
                            {
                                "role": "user",
                                "content": "You now have additional tools available. Continue the previous request.",
                            }
                        ],
                    )
                else:
                    _append_messages(state, turn_messages)

            tools, mapping = build_tool_definitions(functions)
            state.tools = [definition.as_tool() for definition in tools]
            state.tool_choice = "auto"
            state.metadata["tool_name_mapping"] = mapping
            turn_steps: list[list[str]] = []

            for _ in range(MAX_STEPS):
                state = await generate(state, tool_calls="none")
                message = state.output.message
                calls = parse_function_calls(
                    message.tool_calls, state.output.completion, mapping
                )
                if not calls:
                    break
                encoded = [format_python_call(call) for call in calls]
                previous_count = len(all_calls)
                all_calls.extend(encoded)
                turn_steps.append(encoded)
                execution = await _sandbox_request(
                    state,
                    {
                        "operation": "execute",
                        "calls": all_calls,
                        "previous_count": previous_count,
                        "initial_config": state.metadata["initial_config"],
                        "involved_classes": state.metadata["involved_classes"],
                        "id": state.sample_id,
                        "category": state.metadata["category"],
                    },
                )
                _append_tool_results(
                    state, list(execution["outputs"]), message.tool_calls
                )
            model_turn_calls.append(turn_steps)

        state.metadata["model_turn_calls"] = model_turn_calls
        return state

    return solve


@solver
def bfcl_agentic_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools, mapping = build_tool_definitions(list(state.metadata["functions"]))
        state.tools = [definition.as_tool() for definition in tools]
        state.tool_choice = "auto"
        state.metadata["tool_name_mapping"] = mapping
        trace: list[dict[str, Any]] = []
        for _ in range(MAX_STEPS):
            state = await generate(state, tool_calls="none")
            message = state.output.message
            calls = parse_function_calls(
                message.tool_calls, state.output.completion, mapping
            )
            if not calls:
                break
            outputs = [
                frozen_agentic_result(
                    call,
                    source=state.metadata["frozen_source"],
                    show_snippet=bool(state.metadata["show_snippet"]),
                )
                for call in calls
            ]
            trace.extend(
                {"name": call.name, "arguments": call.arguments, "result": result}
                for call, result in zip(calls, outputs)
            )
            _append_tool_results(state, outputs, message.tool_calls)
        state.metadata["agentic_trace"] = trace
        return state

    return solve


@task
def bfcl_v4_single_turn(
    categories: list[str] | None = None,
) -> Task:
    """Evaluate BFCL v4 single-turn native or prompted function calls."""

    return Task(
        dataset=get_bfcl_v4_single_turn_dataset(
            categories or list(SINGLE_TURN_CATEGORIES)
        ),
        solver=bfcl_generate(),
        scorer=bfcl_v4_scorer(),
        config=GenerateConfig(
            temperature=0,
            max_tokens=1024,
            parallel_tool_calls=True,
        ),
        metadata={
            "benchmark": "BFCL v4",
            "scope": "single-turn",
            "official_overall_score": False,
        },
    )


@task
def bfcl_v4_multi_turn(categories: list[str] | None = None) -> Task:
    """Evaluate BFCL v4 multi-turn with the pinned official state checker."""

    return Task(
        dataset=get_bfcl_v4_multi_turn_dataset(
            categories or list(MULTI_TURN_CATEGORIES)
        ),
        solver=bfcl_multi_turn_generate(),
        scorer=bfcl_v4_multi_turn_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(temperature=0, max_tokens=2048, parallel_tool_calls=True),
        message_limit=200,
        metadata={"benchmark": "BFCL v4", "scope": "multi-turn"},
    )


@task
def bfcl_v4_agentic_offline(categories: list[str] | None = None) -> Task:
    """Evaluate BFCL v4 agentic tasks against immutable offline evidence."""

    return Task(
        dataset=get_bfcl_v4_agentic_dataset(categories or list(AGENTIC_CATEGORIES)),
        solver=bfcl_agentic_generate(),
        scorer=bfcl_v4_agentic_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=2048, parallel_tool_calls=True),
        message_limit=100,
        metadata={
            "benchmark": "BFCL v4",
            "scope": "agentic-offline",
            "official_overall_score": False,
        },
    )


@solver
def bfcl_offline_generate() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        category = str(state.metadata["category"])
        if category in MULTI_TURN_CATEGORIES:
            return await bfcl_multi_turn_generate()(state, generate)
        if category in AGENTIC_CATEGORIES:
            return await bfcl_agentic_generate()(state, generate)
        return await bfcl_generate()(state, generate)

    return solve


@task
def bfcl_v4_offline() -> Task:
    """Run all BFCL v4 sections with frozen offline agentic evidence."""

    samples = [
        *list(get_bfcl_v4_single_turn_dataset()),
        *list(get_bfcl_v4_multi_turn_dataset()),
        *list(get_bfcl_v4_agentic_dataset()),
    ]
    return Task(
        dataset=samples,
        solver=bfcl_offline_generate(),
        scorer=bfcl_v4_offline_scorer(),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(temperature=0, max_tokens=2048, parallel_tool_calls=True),
        message_limit=200,
        metadata={
            "benchmark": "BFCL v4",
            "scope": "offline-complete",
            "official_overall_score": False,
        },
    )
