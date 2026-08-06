"""Conversion from benchmark JSON schemas to Inspect tool definitions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from inspect_ai.tool import ToolDef, ToolParams
from inspect_ai.util import JSONSchema

_TYPE_MAPPING = {
    "": None,
    "Any": None,
    "any": None,
    "array": "array",
    "Array": "array",
    "ArrayList": "array",
    "list": "array",
    "Queue": "array",
    "Stack": "array",
    "Bigint": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "Boolean": "boolean",
    "char": "string",
    "dict": "object",
    "HashMap": "object",
    "Hashtable": "object",
    "object": "object",
    "number": "number",
    "byte": "integer",
    "short": "integer",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "long": "integer",
    "String": "string",
    "string": "string",
    "tuple": "array",
}


def safe_tool_name(name: str) -> str:
    """Return a provider-compatible function name."""

    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return normalized[:64] or "tool"


def _json_schema(schema: Mapping[str, Any]) -> JSONSchema:
    raw_type = str(schema.get("type", ""))
    mapped_type = _TYPE_MAPPING.get(raw_type, "string")
    properties = schema.get("properties")
    items = schema.get("items")
    return JSONSchema(
        type=mapped_type,  # type: ignore[arg-type]
        description=(str(schema["description"]) if schema.get("description") else None),
        default=schema.get("default"),
        enum=list(schema["enum"]) if isinstance(schema.get("enum"), list) else None,
        items=_json_schema(items) if isinstance(items, Mapping) else None,
        properties=(
            {str(key): _json_schema(value) for key, value in properties.items()}
            if isinstance(properties, Mapping)
            else None
        ),
        required=(
            [str(value) for value in schema.get("required", [])]
            if mapped_type == "object"
            else None
        ),
        additionalProperties=(
            bool(schema.get("additionalProperties", False))
            if mapped_type == "object"
            else None
        ),
    )


def build_tool_definitions(
    functions: list[dict[str, Any]],
) -> tuple[list[ToolDef], dict[str, str]]:
    """Build non-executing Inspect tools and a safe-to-original name mapping."""

    definitions: list[ToolDef] = []
    name_mapping: dict[str, str] = {}

    for index, function in enumerate(functions):
        original_name = str(function["name"])
        safe_name = safe_tool_name(original_name)
        if safe_name in name_mapping and name_mapping[safe_name] != original_name:
            suffix = f"_{index}"
            safe_name = f"{safe_name[: 64 - len(suffix)]}{suffix}"
        name_mapping[safe_name] = original_name

        async def unavailable_tool(**_: Any) -> str:
            return "Tool execution is disabled for this evaluation."

        parameters = function.get("parameters", {})
        converted = _json_schema(parameters)
        definitions.append(
            ToolDef(
                unavailable_tool,
                name=safe_name,
                description=str(function.get("description", "")),
                parameters=ToolParams(
                    properties=converted.properties or {},
                    required=converted.required or [],
                    additionalProperties=bool(converted.additionalProperties),
                ),
                parallel=True,
            )
        )

    return definitions, name_mapping
