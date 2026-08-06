"""Deterministic, provider-neutral function-call matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass(frozen=True)
class FunctionMatch:
    matched: bool
    error: str | None = None


_PYTHON_TYPES: dict[str, type[Any] | tuple[type[Any], ...]] = {
    "string": str,
    "String": str,
    "char": str,
    "integer": int,
    "byte": int,
    "short": int,
    "long": int,
    "Bigint": int,
    "number": float,
    "float": float,
    "double": float,
    "boolean": bool,
    "bool": bool,
    "Boolean": bool,
    "array": list,
    "Array": list,
    "ArrayList": list,
    "list": list,
    "Queue": list,
    "Stack": list,
    "tuple": (list, tuple),
    "dict": dict,
    "HashMap": dict,
    "Hashtable": dict,
    "object": dict,
    "Any": object,
    "any": object,
    "": object,
}


def _standardize_string(value: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')


def _value_matches(actual: Any, allowed: list[Any]) -> bool:
    for expected in allowed:
        if expected == "":
            continue
        if isinstance(actual, str) and isinstance(expected, str):
            if _standardize_string(actual) == _standardize_string(expected):
                return True
        elif isinstance(actual, (list, tuple)) and isinstance(expected, list):
            if len(actual) == len(expected) and all(
                _value_matches(item, [target]) for item, target in zip(actual, expected)
            ):
                return True
        elif isinstance(actual, dict) and isinstance(expected, dict):
            if any(key not in expected for key in actual):
                continue
            if not all(
                _value_matches(
                    actual[key],
                    expected[key]
                    if isinstance(expected[key], list)
                    else [expected[key]],
                )
                for key in actual
            ):
                continue
            if all(
                key in actual or (isinstance(options, list) and "" in options)
                for key, options in expected.items()
            ):
                return True
        elif type(actual) is type(expected) and actual == expected:
            return True
        elif (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and float(actual) == float(expected)
        ):
            return True
    return False


def _type_matches(value: Any, schema: dict[str, Any], allowed: list[Any]) -> bool:
    expected = _PYTHON_TYPES.get(str(schema.get("type", "")), object)
    if expected is object:
        return True
    if expected is int and isinstance(value, bool):
        return False
    if expected is float:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)) or any(
            item != "" and type(value) is type(item) for item in allowed
        )
    if not isinstance(value, expected):
        # BFCL uses symbolic variable names in the Java/JavaScript sets and a
        # few Python records. The official checker accepts the public answer's
        # concrete type when it intentionally differs from the schema type.
        return any(item != "" and type(value) is type(item) for item in allowed)
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), dict):
        allowed_items = [
            item
            for candidate in allowed
            if isinstance(candidate, (list, tuple))
            for item in candidate
        ]
        return all(
            _type_matches(item, schema["items"], allowed_items) for item in value
        )
    return True


def _single_call_matches(
    actual: FunctionCall,
    expected: dict[str, dict[str, list[Any]]],
    function: dict[str, Any],
) -> FunctionMatch:
    if actual.parse_error:
        return FunctionMatch(False, f"parse error: {actual.parse_error}")
    expected_name, expected_arguments = next(iter(expected.items()))
    if actual.name != expected_name:
        return FunctionMatch(False, f"expected {expected_name}, got {actual.name}")

    properties = function.get("parameters", {}).get("properties", {})
    required = set(function.get("parameters", {}).get("required", []))
    unexpected = set(actual.arguments) - set(properties)
    unexpected.update(set(actual.arguments) - set(expected_arguments))
    if unexpected:
        return FunctionMatch(False, f"unexpected arguments: {sorted(unexpected)}")

    for name, allowed in expected_arguments.items():
        if name not in actual.arguments:
            if name in required:
                return FunctionMatch(False, f"missing required argument: {name}")
            if "" not in allowed:
                return FunctionMatch(False, f"missing expected argument: {name}")
            continue
        value = actual.arguments[name]
        schema = properties.get(name, {})
        if not _type_matches(value, schema, allowed):
            return FunctionMatch(False, f"invalid type for argument: {name}")
        if not _value_matches(value, allowed):
            return FunctionMatch(False, f"invalid value for argument: {name}")

    missing_required = required - set(actual.arguments)
    for name in missing_required:
        if "" not in expected_arguments.get(name, []):
            return FunctionMatch(False, f"missing required argument: {name}")
    return FunctionMatch(True)


def match_function_calls(
    actual: list[FunctionCall],
    expected: list[dict[str, dict[str, list[Any]]]],
    functions: list[dict[str, Any]],
    *,
    order_sensitive: bool = False,
) -> FunctionMatch:
    """Match calls one-to-one, allowing parallel calls in any order."""

    if len(actual) != len(expected):
        return FunctionMatch(
            False, f"expected {len(expected)} call(s), got {len(actual)}"
        )
    functions_by_name = {str(function["name"]): function for function in functions}

    if order_sensitive:
        pairs = zip(actual, expected)
        for actual_call, expected_call in pairs:
            name = next(iter(expected_call))
            result = _single_call_matches(
                actual_call, expected_call, functions_by_name[name]
            )
            if not result.matched:
                return result
        return FunctionMatch(True)

    unmatched = list(range(len(actual)))
    for expected_call in expected:
        name = next(iter(expected_call))
        for index in unmatched:
            actual_call = actual[index]
            result = _single_call_matches(
                actual_call, expected_call, functions_by_name[name]
            )
            if result.matched:
                unmatched.remove(index)
                break
        else:
            return FunctionMatch(False, f"no actual call matched {name}")
    return FunctionMatch(True)
