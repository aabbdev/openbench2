"""Provider-neutral primitives for function-calling evaluations."""

from openbench.function_calling.matching import (
    FunctionCall,
    FunctionMatch,
    match_function_calls,
)
from openbench.function_calling.parsing import parse_function_calls
from openbench.function_calling.schema import build_tool_definitions

__all__ = [
    "FunctionCall",
    "FunctionMatch",
    "build_tool_definitions",
    "match_function_calls",
    "parse_function_calls",
]
