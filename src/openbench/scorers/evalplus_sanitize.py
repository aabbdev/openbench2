"""EvalPlus's tree-sitter Python solution sanitizer.

Adapted under Apache-2.0 from ``evalplus/sanitize.py`` at commit
26d6d00bb1fd0fa37f39c99d5290da67891d1c5e. Copyright EvalPlus contributors.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Generator

import tree_sitter_python
from tree_sitter import Language, Node, Parser

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPES = {"import_statement", "import_from_statement"}
IDENTIFIER_TYPE = "identifier"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def _syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def code_extract(text: str) -> str:
    """Return the longest contiguous syntactically valid part of a response."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    lines = ansi_escape.sub("", text).split("\n")
    longest_pair = (0, 0)
    longest_so_far = 0
    for start in range(len(lines)):
        for end in range(start + 1, len(lines)):
            current = lines[start : end + 1]
            if _syntax_check("\n".join(current)):
                length = sum(1 for line in current if line.strip())
                if length > longest_so_far:
                    longest_so_far = length
                    longest_pair = (start, end)
    return "\n".join(lines[longest_pair[0] : longest_pair[1] + 1])


def _traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0
    visited_children = False
    while True:
        if not visited_children:
            current_node = cursor.node
            if current_node is not None:
                yield current_node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def _definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            assert child.text is not None
            return child.text.decode("utf8")
    raise ValueError("Definition has no identifier")


def _has_return(node: Node) -> bool:
    return any(item.type == RETURN_TYPE for item in _traverse_tree(node))


def _dependencies(nodes: list[tuple[str, Node]]) -> dict[str, set[str]]:
    def visit(node: Node, result: set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                assert child.text is not None
                result.add(child.text.decode("utf8"))
            else:
                visit(child, result)

    dependencies: dict[str, set[str]] = {}
    for name, node in nodes:
        dependencies[name] = set()
        visit(node, dependencies[name])
    return dependencies


def _reachable(entrypoint: str, graph: dict[str, set[str]]) -> set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        for neighbour in graph.get(current, set()):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def extract_target_code_or_empty(code: str, entrypoint: str | None = None) -> str:
    code = code_extract(code)
    code_bytes = code.encode("utf8")
    tree = Parser(Language(tree_sitter_python.language())).parse(code_bytes)
    class_names: set[str] = set()
    function_names: set[str] = set()
    variable_names: set[str] = set()
    import_nodes: list[Node] = []
    definition_nodes: list[tuple[str, Node]] = []

    for child in tree.root_node.children:
        if child.type in IMPORT_TYPES:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = _definition_name(child)
            if name not in class_names | variable_names | function_names:
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = _definition_name(child)
            if (
                name not in function_names | variable_names | class_names
                and _has_return(child)
            ):
                definition_nodes.append((name, child))
                function_names.add(name)
        elif (
            child.type == EXPRESSION_TYPE
            and child.children
            and child.children[0].type == ASSIGNMENT_TYPE
        ):
            assignment = child.children[0]
            name = _definition_name(assignment)
            if name not in variable_names | function_names | class_names:
                definition_nodes.append((name, assignment))
                variable_names.add(name)

    reachable = (
        _reachable(entrypoint, _dependencies(definition_nodes)) if entrypoint else None
    )
    output = b""
    for node in import_nodes:
        output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    for name, node in definition_nodes:
        if reachable is not None and name not in reachable:
            continue
        output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return output[:-1].decode("utf8")


def sanitize(code: str, entrypoint: str | None = None) -> str:
    sanitized = extract_target_code_or_empty(code, entrypoint).strip()
    return sanitized if sanitized else code_extract(code)
