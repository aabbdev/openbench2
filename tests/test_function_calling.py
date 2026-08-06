"""Tests for provider-neutral function-calling primitives."""

from typing import Any

from inspect_ai.tool import ToolCall

from openbench.function_calling import (
    FunctionCall,
    build_tool_definitions,
    match_function_calls,
    parse_function_calls,
)
from openbench.function_calling.stateful import frozen_agentic_result


FUNCTIONS = [
    {
        "name": "weather.get",
        "description": "Get weather.",
        "parameters": {
            "type": "dict",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "string"},
            },
            "required": ["city", "days"],
        },
    }
]


def test_build_tool_definitions_normalizes_names_and_schema() -> None:
    definitions, mapping = build_tool_definitions(FUNCTIONS)
    info = definitions[0]

    assert info.name == "weather_get"
    assert mapping == {"weather_get": "weather.get"}
    assert info.parameters.required == ["city", "days"]
    assert info.parameters.properties["days"].type == "integer"


def test_parse_native_json_and_python_calls() -> None:
    native = [ToolCall(id="1", function="weather_get", arguments={"city": "Paris"})]
    assert parse_function_calls(native, "", {"weather_get": "weather.get"}) == [
        FunctionCall("weather.get", {"city": "Paris"})
    ]

    assert parse_function_calls(
        None, '[{"name":"weather.get","arguments":{"city":"Paris"}}]'
    ) == [FunctionCall("weather.get", {"city": "Paris"})]
    assert parse_function_calls(None, "[weather.get(city='Paris', days=2)]") == [
        FunctionCall("weather.get", {"city": "Paris", "days": 2})
    ]


def test_match_calls_handles_optional_values_and_parallel_order() -> None:
    expected: list[dict[str, dict[str, list[Any]]]] = [
        {
            "weather.get": {
                "city": ["Paris"],
                "days": [2],
                "units": ["", "metric"],
            }
        },
        {
            "weather.get": {
                "city": ["New York"],
                "days": [1],
                "units": ["", "metric"],
            }
        },
    ]
    actual = [
        FunctionCall("weather.get", {"city": "New-York", "days": 1}),
        FunctionCall("weather.get", {"city": "paris", "days": 2, "units": "metric"}),
    ]
    result = match_function_calls(actual, expected, FUNCTIONS)
    assert result.matched is True


def test_match_calls_rejects_wrong_types_and_extra_arguments() -> None:
    expected: list[dict[str, dict[str, list[Any]]]] = [
        {"weather.get": {"city": ["Paris"], "days": [2]}}
    ]
    wrong_type = match_function_calls(
        [FunctionCall("weather.get", {"city": "Paris", "days": 2.0})],
        expected,
        FUNCTIONS,
    )
    extra = match_function_calls(
        [FunctionCall("weather.get", {"city": "Paris", "days": 2, "unknown": True})],
        expected,
        FUNCTIONS,
    )
    assert wrong_type.matched is False
    assert extra.matched is False


def test_match_calls_allows_python_int_for_float_schema() -> None:
    functions = [
        {
            "name": "scale",
            "parameters": {
                "type": "dict",
                "properties": {"value": {"type": "float"}},
                "required": ["value"],
            },
        }
    ]
    result = match_function_calls(
        [FunctionCall("scale", {"value": 3})],
        [{"scale": {"value": [3.0]}}],
        functions,
    )
    assert result.matched is True


def test_frozen_web_search_hides_or_exposes_snippets() -> None:
    source = [{"subquestion": "Who?", "answer": "Ada", "source": "https://x"}]
    call = FunctionCall("search_engine_query", {"keywords": "who"})
    assert "Ada" in frozen_agentic_result(call, source=source, show_snippet=True)
    assert "Ada" not in frozen_agentic_result(call, source=source, show_snippet=False)

    fetch = FunctionCall("fetch_url_content", {"url": "https://x"})
    assert "Ada" in frozen_agentic_result(fetch, source=source, show_snippet=False)
