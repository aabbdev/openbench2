"""Scoring for BFCL v4 function calls."""

from __future__ import annotations

import json
import re

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from openbench.function_calling import match_function_calls, parse_function_calls
from openbench.metrics.bfcl import (
    bfcl_v4_agentic_metrics,
    bfcl_v4_multi_turn_metrics,
    bfcl_v4_offline_metrics,
    bfcl_v4_single_turn_metrics,
)


@scorer(metrics=[bfcl_v4_single_turn_metrics()])
def bfcl_v4_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        del target
        message = state.output.message
        calls = parse_function_calls(
            message.tool_calls,
            state.output.completion,
            state.metadata.get("tool_name_mapping", {}),
        )
        category = str(state.metadata["category"])

        if category in {"irrelevance", "live_irrelevance"}:
            matched = not calls
            error = None if matched else "A tool was called for an irrelevant request"
        elif category == "live_relevance":
            matched = bool(calls)
            error = None if matched else "No tool was called for a relevant request"
        else:
            result = match_function_calls(
                calls,
                list(state.metadata["expected_calls"]),
                list(state.metadata["functions"]),
                order_sensitive=False,
            )
            matched, error = result.matched, result.error

        answer = [{"name": call.name, "arguments": call.arguments} for call in calls]
        return Score(
            value=CORRECT if matched else INCORRECT,
            answer=str(answer),
            explanation=error,
            metadata={"category": category, "call_count": len(calls)},
        )

    return score


async def _multi_turn_score(state: TaskState) -> Score:
    payload_path = f".openbench_bfcl_score_{state.uuid}.json"
    payload = {
        "operation": "score",
        "model_turn_calls": state.metadata.get("model_turn_calls", []),
        "ground_truth": state.metadata["ground_truth"],
        "category": state.metadata["category"],
        "test_entry": {
            "id": state.sample_id,
            "initial_config": state.metadata["initial_config"],
            "involved_classes": state.metadata["involved_classes"],
        },
    }
    environment = sandbox()
    await environment.write_file(payload_path, json.dumps(payload))
    result = await environment.exec(
        ["python", "/opt/openbench/bfcl_runner.py", payload_path],
        timeout=180,
        timeout_retry=False,
    )
    if not result.success:
        return Score(value=INCORRECT, explanation="BFCL official checker failed")
    evaluation = json.loads(result.stdout.strip().splitlines()[-1])
    matched = evaluation.get("valid") is True
    return Score(
        value=CORRECT if matched else INCORRECT,
        answer=str(state.metadata.get("model_turn_calls", [])),
        explanation=None
        if matched
        else str(evaluation.get("error_message", evaluation)),
        metadata={"category": state.metadata["category"]},
    )


@scorer(metrics=[bfcl_v4_multi_turn_metrics()])
def bfcl_v4_multi_turn_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        del target
        return await _multi_turn_score(state)

    return score


def _standardize_answer(value: str) -> str:
    return re.sub(r"[,./\-_*^()]", "", value).lower().replace("'", '"')


def _agentic_score(state: TaskState) -> Score:
    completion = state.output.completion
    standardized = _standardize_answer(completion)
    expected = [str(value) for value in state.metadata["expected_answers"]]
    matched = any(
        re.search(rf"\b{re.escape(_standardize_answer(answer))}\b", standardized)
        for answer in expected
    )
    return Score(
        value=CORRECT if matched else INCORRECT,
        answer=completion,
        explanation=None if matched else f"Expected one of {expected}",
        metadata={
            "category": state.metadata["category"],
            "offline_adaptation": True,
        },
    )


@scorer(metrics=[bfcl_v4_agentic_metrics()])
def bfcl_v4_agentic_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        del target
        return _agentic_score(state)

    return score


@scorer(metrics=[bfcl_v4_offline_metrics()])
def bfcl_v4_offline_scorer() -> Scorer:
    single = bfcl_v4_scorer()

    async def score(state: TaskState, target: Target) -> Score:
        category = str(state.metadata["category"])
        if category.startswith("multi_turn_"):
            return await _multi_turn_score(state)
        if category.startswith("memory_") or category.startswith("web_search_"):
            return _agentic_score(state)
        result = await single(state, target)
        if result is None:
            return Score(value=INCORRECT, explanation="BFCL scorer returned no score")
        return result

    return score
