"""Reusable stateful execution helpers for function-calling benchmarks."""

from __future__ import annotations

import json
from typing import Any

from openbench.function_calling.matching import FunctionCall


def format_python_call(call: FunctionCall) -> str:
    """Serialize a normalized call without evaluating model-authored source."""

    arguments = ", ".join(f"{name}={value!r}" for name, value in call.arguments.items())
    return f"{call.name}({arguments})"


def frozen_agentic_result(
    call: FunctionCall,
    *,
    source: str | list[dict[str, Any]],
    show_snippet: bool,
) -> str:
    """Execute BFCL agentic tools against an immutable offline evidence set."""

    name = call.name
    if name == "search_engine_query" and isinstance(source, list):
        results = []
        for index, item in enumerate(source):
            result = {
                "title": str(item.get("subquestion", f"Result {index + 1}")),
                "url": str(item.get("source", "")),
            }
            if show_snippet:
                result["snippet"] = str(item.get("answer", ""))
            results.append(result)
        limit = int(call.arguments.get("max_results", 10))
        return json.dumps(results[:limit])

    if name == "fetch_url_content" and isinstance(source, list):
        url = str(call.arguments.get("url", ""))
        matches = [item for item in source if str(item.get("source", "")) == url]
        content = "\n".join(
            f"{item.get('subquestion', '')}: {item.get('answer', '')}"
            for item in matches
        )
        return json.dumps({"content": content or "URL not found in frozen corpus"})

    memory = str(source)
    if name == "memory_retrieve":
        return json.dumps({"memory_content": memory})
    if "retrieve" in name or "search" in name:
        return json.dumps({"results": [{"id": 0, "score": 1.0, "text": memory}]})
    if "list_keys" in name:
        return json.dumps({"keys": ["profile"]})
    return json.dumps({"status": "ok"})
