"""Safe parsing for native, JSON, and Python-style function calls."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from inspect_ai.tool import ToolCall

from openbench.function_calling.matching import FunctionCall


def _from_json(value: Any) -> list[FunctionCall]:
    if isinstance(value, dict):
        if "function" in value and isinstance(value["function"], dict):
            value = value["function"]
        name = value.get("name") or value.get("function")
        arguments = value.get("arguments", value.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if isinstance(name, str) and isinstance(arguments, dict):
            return [FunctionCall(name=name, arguments=arguments)]
        if len(value) == 1:
            name, arguments = next(iter(value.items()))
            if isinstance(name, str) and isinstance(arguments, dict):
                return [FunctionCall(name=name, arguments=arguments)]
        return []
    if isinstance(value, list):
        calls: list[FunctionCall] = []
        for item in value:
            calls.extend(_from_json(item))
        return calls
    return []


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _from_python(text: str) -> list[FunctionCall]:
    parsed = ast.parse(text.strip(), mode="eval").body
    nodes = parsed.elts if isinstance(parsed, (ast.List, ast.Tuple)) else [parsed]
    calls: list[FunctionCall] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            return []
        name = _call_name(node.func)
        if name is None or any(keyword.arg is None for keyword in node.keywords):
            return []
        arguments = {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in node.keywords
        }
        calls.append(FunctionCall(name=name, arguments=arguments))
    return calls


def parse_function_calls(
    native_calls: list[ToolCall] | None,
    completion: str,
    name_mapping: dict[str, str] | None = None,
) -> list[FunctionCall]:
    """Normalize provider-native calls, then fall back to safe text parsing."""

    mapping = name_mapping or {}
    if native_calls:
        return [
            FunctionCall(
                name=mapping.get(call.function, call.function),
                arguments=dict(call.arguments),
                parse_error=call.parse_error,
            )
            for call in native_calls
        ]

    text = completion.strip()
    fenced = re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced[-1].strip()
    for parser in (lambda value: _from_json(json.loads(value)), _from_python):
        try:
            calls = parser(text)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if calls:
            return [
                FunctionCall(
                    name=mapping.get(call.name, call.name),
                    arguments=call.arguments,
                    parse_error=call.parse_error,
                )
                for call in calls
            ]
    return []
