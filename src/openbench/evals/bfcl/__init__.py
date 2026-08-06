"""Berkeley Function Calling Leaderboard v4 tasks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

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
    get_bfcl_v4_agentic_live_dataset,
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
    agentic_answer_matches,
    bfcl_v4_agentic_scorer,
    bfcl_v4_agentic_live_scorer,
    bfcl_v4_multi_turn_scorer,
    bfcl_v4_offline_scorer,
    bfcl_v4_scorer,
)

COMPOSE_PATH = (Path(__file__).parent / "compose.yaml").resolve()
LIVE_COMPOSE_PATH = (Path(__file__).parent / "compose.live.yaml").resolve()
REPLAY_COMPOSE_PATH = (Path(__file__).parent / "compose.replay.yaml").resolve()
MAX_STEPS = 20

AGENTIC_RESPONSE_PROMPT = """For your final answer to the user, you must respond in this format: {'answer': A short and precise answer to the question, 'context': A brief explanation of how you arrived at this answer or why it is correct}. If you do not know the answer, respond with {'answer': 'I do not know', 'context': 'I do not know'}. If you think the question cannot be properly answered, response with {'answer': 'I cannot answer this question', 'context': A short reason explaining why this question cannot be answered}.
"""


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


def _configure_tools(state: TaskState) -> dict[str, str]:
    tools, mapping = build_tool_definitions(list(state.metadata["functions"]))
    state.tools = [definition.as_tool() for definition in tools]
    state.tool_choice = "auto"
    state.metadata["tool_name_mapping"] = mapping
    return mapping


async def _run_agentic_conversation(
    state: TaskState,
    generate: Generate,
    *,
    turns: list[list[dict[str, str]]],
    system_prompt: str,
    execute_calls: Callable[[list[str], list[Any] | None], Awaitable[list[str]]],
) -> str:
    """Run one isolated BFCL conversation through its complete tool loop."""

    state.messages = [ChatMessageSystem(content=system_prompt)]
    mapping = _configure_tools(state)
    for turn in turns:
        _append_messages(state, turn)
        for _ in range(MAX_STEPS):
            state = await generate(state, tool_calls="none")
            message = state.output.message
            calls = parse_function_calls(
                message.tool_calls, state.output.completion, mapping
            )
            if not calls:
                break
            outputs = await execute_calls(
                [format_python_call(call) for call in calls], message.tool_calls
            )
            _append_tool_results(state, outputs, message.tool_calls)
        else:
            raise RuntimeError(f"BFCL agentic workflow exceeded {MAX_STEPS} tool steps")
    return state.output.completion


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


async def _start_live_runner(state: TaskState) -> str:
    environment = sandbox()
    socket_path = f"/workspace/.openbench_bfcl_{state.uuid}.sock"
    log_path = f"/workspace/.openbench_bfcl_{state.uuid}.log"
    command = (
        f"rm -f {socket_path}; "
        f"nohup python /opt/openbench/bfcl_runner.py --serve {socket_path} "
        f">{log_path} 2>&1 & "
        f"for i in $(seq 1 300); do [ -S {socket_path} ] && exit 0; sleep 0.1; done; "
        f"cat {log_path}; exit 1"
    )
    result = await environment.exec(["sh", "-c", command], timeout=60)
    if not result.success:
        raise RuntimeError(f"BFCL live runner failed to start: {result.stderr}")
    return socket_path


async def _live_runner_request(
    state: TaskState, socket_path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    environment = sandbox()
    payload_path = f".openbench_bfcl_live_{state.uuid}.json"
    await environment.write_file(payload_path, json.dumps(payload))
    result = await environment.exec(
        [
            "python",
            "/opt/openbench/bfcl_runner.py",
            "--client",
            socket_path,
            payload_path,
        ],
        timeout=180,
        timeout_retry=False,
    )
    if not result.success:
        raise RuntimeError("BFCL live runner request failed")
    response = json.loads(result.stdout.strip().splitlines()[-1])
    if "runner_error" in response:
        raise RuntimeError(f"BFCL live backend failed: {response['runner_error']}")
    return response


async def _stop_live_runner(state: TaskState, socket_path: str) -> None:
    try:
        await _live_runner_request(state, socket_path, {"operation": "shutdown"})
    except Exception:
        pass


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


def _memory_request(
    state: TaskState,
    *,
    operation: str,
    scenario: str,
    session_id: str,
    calls: list[str],
) -> dict[str, Any]:
    class_name = str(state.metadata["involved_classes"][0])
    category = str(state.metadata["category"])
    return {
        "operation": operation,
        "calls": calls,
        "previous_count": 0,
        "initial_config": {
            class_name: {
                "model_result_dir": "/workspace/memory_snapshots",
                "scenario": scenario,
                "test_id": session_id,
                "test_category": category,
            }
        },
        "involved_classes": [class_name],
        "id": session_id,
        "category": category,
        "scenario": scenario,
    }


@solver
def bfcl_agentic_live_generate() -> Solver:
    """Run BFCL's model-specific memory prerequisites and live web tools."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        category = str(state.metadata["category"])
        snapshot_dir_value = state.metadata.get("web_snapshot_dir")
        record_dir_value = state.metadata.get("record_web_snapshot_dir")
        if snapshot_dir_value and record_dir_value:
            raise ValueError(
                "BFCL web snapshot replay and recording are mutually exclusive"
            )
        if (
            category.startswith("web_search_")
            and not snapshot_dir_value
            and not os.getenv("SERPAPI_API_KEY")
        ):
            raise RuntimeError(
                "bfcl_v4_agentic_live requires SERPAPI_API_KEY for web search"
            )

        socket_path = await _start_live_runner(state)
        results: list[dict[str, Any]] = []
        try:
            if category.startswith("memory_"):
                for scenario_data in state.metadata["workflow"]["scenarios"]:
                    scenario = str(scenario_data["name"])
                    prerequisite_calls: list[str] = []
                    prerequisite_session = f"{category}_prereq_{scenario}_session-0"

                    for prerequisite in scenario_data["prerequisites"]:
                        context = await _live_runner_request(
                            state,
                            socket_path,
                            _memory_request(
                                state,
                                operation="memory_context",
                                scenario=scenario,
                                session_id=prerequisite_session,
                                calls=[],
                            ),
                        )

                        async def execute_prerequisite(
                            encoded: list[str], native_calls: list[Any] | None
                        ) -> list[str]:
                            del native_calls
                            prerequisite_calls.extend(encoded)
                            response = await _live_runner_request(
                                state,
                                socket_path,
                                _memory_request(
                                    state,
                                    operation="execute",
                                    scenario=scenario,
                                    session_id=prerequisite_session,
                                    calls=encoded,
                                ),
                            )
                            return list(response["outputs"])

                        await _run_agentic_conversation(
                            state,
                            generate,
                            turns=list(prerequisite["question"]),
                            system_prompt=str(context["system_prompt"]),
                            execute_calls=execute_prerequisite,
                        )

                    for index, case in enumerate(scenario_data["cases"]):
                        target_session = (
                            f"{category}_prereq_{scenario}_target_{index}-0"
                        )
                        context = await _live_runner_request(
                            state,
                            socket_path,
                            _memory_request(
                                state,
                                operation="memory_context",
                                scenario=scenario,
                                session_id=target_session,
                                calls=prerequisite_calls,
                            ),
                        )

                        async def execute_target(
                            encoded: list[str], native_calls: list[Any] | None
                        ) -> list[str]:
                            del native_calls
                            response = await _live_runner_request(
                                state,
                                socket_path,
                                _memory_request(
                                    state,
                                    operation="execute",
                                    scenario=scenario,
                                    session_id=target_session,
                                    calls=encoded,
                                ),
                            )
                            return list(response["outputs"])

                        completion = await _run_agentic_conversation(
                            state,
                            generate,
                            turns=list(case["question"]),
                            system_prompt=(
                                f"{context['system_prompt']}\n\n"
                                f"{AGENTIC_RESPONSE_PROMPT}"
                            ),
                            execute_calls=execute_target,
                        )
                        expected = [str(value) for value in case["expected_answers"]]
                        results.append(
                            {
                                "id": case["id"],
                                "correct": agentic_answer_matches(completion, expected),
                                "completion": completion,
                            }
                        )
            else:
                class_name = str(state.metadata["involved_classes"][0])
                snapshot: dict[str, Any] | None = None
                if snapshot_dir_value:
                    snapshot_path = Path(str(snapshot_dir_value)) / f"{category}.json"
                    loaded_snapshot = cast(
                        dict[str, Any], json.loads(snapshot_path.read_text())
                    )
                    if (
                        loaded_snapshot.get("bfcl_revision")
                        != state.metadata["bfcl_revision"]
                    ):
                        raise ValueError(
                            "BFCL web snapshot revision does not match dataset"
                        )
                    if loaded_snapshot.get("category") != category:
                        raise ValueError(
                            "BFCL web snapshot category does not match sample"
                        )
                    snapshot = loaded_snapshot
                recorded_cases: dict[str, list[dict[str, Any]]] = {}
                initial_config = {
                    class_name: {"show_snippet": bool(state.metadata["show_snippet"])}
                }
                for case in state.metadata["workflow"]["cases"]:
                    session_id = str(case["id"])
                    step_index = 0
                    case_records: list[dict[str, Any]] = []

                    async def execute_web(
                        encoded: list[str], native_calls: list[Any] | None
                    ) -> list[str]:
                        nonlocal step_index
                        del native_calls
                        if snapshot is not None:
                            expected_steps = snapshot["cases"].get(session_id, [])
                            if step_index >= len(expected_steps):
                                raise ValueError(
                                    f"BFCL snapshot has no step {step_index} for {session_id}"
                                )
                            expected_step = expected_steps[step_index]
                            if expected_step["calls"] != encoded:
                                raise ValueError(
                                    f"BFCL snapshot call mismatch for {session_id} step {step_index}"
                                )
                            outputs = [str(value) for value in expected_step["outputs"]]
                        else:
                            response = await _live_runner_request(
                                state,
                                socket_path,
                                {
                                    "operation": "execute",
                                    "calls": encoded,
                                    "previous_count": 0,
                                    "initial_config": initial_config,
                                    "involved_classes": [class_name],
                                    "id": session_id,
                                    "category": category,
                                },
                            )
                            outputs = list(response["outputs"])
                        case_records.append({"calls": encoded, "outputs": outputs})
                        step_index += 1
                        return outputs

                    completion = await _run_agentic_conversation(
                        state,
                        generate,
                        turns=list(case["question"]),
                        system_prompt=AGENTIC_RESPONSE_PROMPT,
                        execute_calls=execute_web,
                    )
                    if snapshot is not None and step_index != len(
                        snapshot["cases"].get(session_id, [])
                    ):
                        raise ValueError(
                            f"BFCL snapshot has unused steps for {session_id}"
                        )
                    expected = [str(value) for value in case["expected_answers"]]
                    results.append(
                        {
                            "id": case["id"],
                            "correct": agentic_answer_matches(completion, expected),
                            "completion": completion,
                        }
                    )
                    recorded_cases[session_id] = case_records

                if record_dir_value:
                    record_dir = Path(str(record_dir_value))
                    record_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_payload = {
                        "schema_version": 1,
                        "bfcl_revision": state.metadata["bfcl_revision"],
                        "category": category,
                        "cases": recorded_cases,
                    }
                    serialized = json.dumps(
                        snapshot_payload, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    destination = record_dir / f"{category}.json"
                    temporary = destination.with_suffix(".json.tmp")
                    temporary.write_text(serialized)
                    temporary.replace(destination)
                    state.metadata["web_snapshot_sha256"] = hashlib.sha256(
                        serialized.encode()
                    ).hexdigest()
        finally:
            await _stop_live_runner(state, socket_path)

        correct_count = sum(bool(result["correct"]) for result in results)
        state.metadata["agentic_results"] = results
        state.metadata["agentic_correct_count"] = correct_count
        state.metadata["agentic_accuracy"] = (
            correct_count / len(results) if results else 0.0
        )
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


@task
def bfcl_v4_agentic_live(
    categories: list[str] | None = None,
    web_snapshot_dir: str | None = None,
    record_web_snapshot_dir: str | None = None,
) -> Task:
    """Run BFCL v4 agentic workflows with official model-specific state and web."""

    dataset = get_bfcl_v4_agentic_live_dataset(categories or list(AGENTIC_CATEGORIES))
    for sample in dataset:
        if sample.metadata is not None:
            sample.metadata["web_snapshot_dir"] = web_snapshot_dir
            sample.metadata["record_web_snapshot_dir"] = record_web_snapshot_dir

    return Task(
        dataset=dataset,
        solver=bfcl_agentic_live_generate(),
        scorer=bfcl_v4_agentic_live_scorer(),
        sandbox=(
            "docker",
            str(REPLAY_COMPOSE_PATH if web_snapshot_dir else LIVE_COMPOSE_PATH),
        ),
        config=GenerateConfig(
            temperature=0.001,
            max_tokens=2048,
            parallel_tool_calls=True,
        ),
        message_limit=200,
        metadata={
            "benchmark": "BFCL v4",
            "scope": "agentic-live",
            "logical_case_count": 665,
            "official_overall_score": False,
            "web_snapshot_dir": web_snapshot_dir,
            "record_web_snapshot_dir": record_web_snapshot_dir,
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
