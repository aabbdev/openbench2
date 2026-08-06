"""Container-side adapter around BFCL's pinned official multi-turn checker."""

from __future__ import annotations

import ast
import ipaddress
import json
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bfcl_eval.constants.default_prompts import (  # type: ignore[import-not-found]
    MEMORY_AGENT_SETTINGS,
    MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL,
    MEMORY_BACKEND_INSTRUCTION_UNIFIED,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # type: ignore[import-not-found]
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # type: ignore[import-not-found]
    execute_multi_turn_func_call,
)


def _execute_calls(payload: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    calls = [str(value) for value in payload["calls"]]
    _validate_web_fetches(calls)
    initial_config = payload["initial_config"]
    for config in initial_config.values():
        if "model_result_dir" in config:
            config["model_result_dir"] = Path(config["model_result_dir"])
    return execute_multi_turn_func_call(
        func_call_list=calls,
        initial_config=initial_config,
        involved_classes=payload["involved_classes"],
        model_name="openbench_generation",
        test_entry_id=payload["id"],
        long_context="long_context" in payload["category"],
        is_evaL_run=False,
    )


def _validate_web_fetches(calls: list[str]) -> None:
    """Prevent model-authored web calls from reaching private network services."""

    for call in calls:
        expression = ast.parse(call, mode="eval").body
        if not isinstance(expression, ast.Call):
            continue
        function_name = (
            expression.func.id if isinstance(expression.func, ast.Name) else ""
        )
        if function_name != "fetch_url_content":
            continue
        url_node = next(
            (keyword.value for keyword in expression.keywords if keyword.arg == "url"),
            expression.args[0] if expression.args else None,
        )
        if url_node is None:
            continue
        url = str(ast.literal_eval(url_node))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("BFCL web fetch requires an HTTP(S) URL")
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
        }
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ValueError("BFCL web fetch blocked a non-public destination")


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    previous_count = int(payload.get("previous_count", 0))
    outputs, _ = _execute_calls(payload)
    return {"outputs": outputs[previous_count:]}


def memory_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Replay a model's prerequisite calls and render BFCL's exact memory prompt."""

    outputs, involved_instances = _execute_calls(payload)
    if len(involved_instances) != 1:
        raise ValueError("BFCL memory workflows require exactly one backend instance")
    instance = next(iter(involved_instances.values()))
    memory_content = instance._dump_core_memory_to_context()
    scenario = str(payload["scenario"])
    template = (
        MEMORY_BACKEND_INSTRUCTION_UNIFIED
        if "rec_sum" in str(payload["category"])
        else MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL
    )
    return {
        "outputs": outputs[int(payload.get("previous_count", 0)) :],
        "system_prompt": template.format(
            scenario_setting=MEMORY_AGENT_SETTINGS[scenario],
            memory_content=memory_content,
        ),
    }


def score(payload: dict[str, Any]) -> dict[str, Any]:
    result = multi_turn_checker(
        payload["model_turn_calls"],
        payload["ground_truth"],
        payload["test_entry"],
        payload["category"],
        "openbench",
    )
    if result.get("valid"):
        result = multi_turn_irrelevance_checker(
            payload["model_turn_calls"], payload["ground_truth"]
        )
    return result


def dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    operation = payload.pop("operation")
    if operation == "execute":
        return execute(payload)
    if operation == "memory_context":
        return memory_context(payload)
    if operation == "score":
        return score(payload)
    raise ValueError(f"Unsupported BFCL runner operation: {operation}")


def serve(socket_path: Path) -> None:
    """Serve requests while retaining official backend instances in memory."""

    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen()
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                chunks = []
                while chunk := connection.recv(1024 * 1024):
                    chunks.append(chunk)
                request = json.loads(b"".join(chunks).decode())
                if request.get("operation") == "shutdown":
                    connection.sendall(b'{"stopped": true}')
                    return
                try:
                    response = dispatch(request)
                except Exception as error:
                    response = {"runner_error": str(error)}
                connection.sendall(json.dumps(response, default=str).encode())
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def client(socket_path: Path, payload_path: Path) -> None:
    payload = payload_path.read_bytes()
    payload_path.unlink()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(socket_path))
    with connection:
        connection.sendall(payload)
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := connection.recv(1024 * 1024):
            chunks.append(chunk)
    print(b"".join(chunks).decode())


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(Path(sys.argv[2]))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--client":
        client(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: bfcl_runner PAYLOAD.json")
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text())
    payload_path.unlink()
    result = dispatch(payload)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
