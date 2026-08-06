"""Pinned BFCL v4 single-turn dataset loader."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from platformdirs import user_cache_dir

BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_LICENSE = "Apache-2.0"
BFCL_BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    f"{BFCL_REVISION}/berkeley-function-call-leaderboard/bfcl_eval/data"
)

SINGLE_TURN_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)

MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)

AGENTIC_CATEGORIES = (
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
    "web_search_base",
    "web_search_no_snippet",
)

_MULTI_TURN_COUNT = 200

_FUNCTION_DOCS = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}

_FILE_SHA256 = {
    "BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "BFCL_v4_simple_java.json": "13d2303a125b08754f0e41995b9273b5005fa8ed8ebfaa24ef53b4d83c4b5c6e",
    "BFCL_v4_simple_javascript.json": "329e67fedf79a6243d93dbda4b388d12bd2d31f1f2163d92cb6ef676d1764f44",
    "BFCL_v4_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "BFCL_v4_parallel.json": "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
    "BFCL_v4_parallel_multiple.json": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "BFCL_v4_irrelevance.json": "2b6ed4c2e992cdcf5f1678a701851f944bef7550ee026ed1ddb89efed5be01a6",
    "BFCL_v4_live_simple.json": "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
    "BFCL_v4_live_multiple.json": "fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8",
    "BFCL_v4_live_parallel.json": "6c26e9fdc3350cf596e6d1ea9c179cbff834761bccf562f4141ed29a839ca421",
    "BFCL_v4_live_parallel_multiple.json": "21d4b9319c1faac431e22757b367ea28917fe467364c3a4b17f16ec06d4f6e79",
    "BFCL_v4_live_irrelevance.json": "6559fda2beaceb609a2cd2e504c65b4a56cb448e1ef88fddfd199e163d163349",
    "BFCL_v4_live_relevance.json": "e03f9e241657a137cba48a89ee12f47bf3fcb7e4f6274263e9c699a0c974203a",
    "possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",
    "possible_answer/BFCL_v4_simple_java.json": "78f25616084044fa05bbfcee68e03f6ececb222bdd5cb3b7783a675fb3366e35",
    "possible_answer/BFCL_v4_simple_javascript.json": "e2f9f2e51d88e0c8056ffbf1a3dd3d02eb032532d2b5d98c9cc9003385bdd56b",
    "possible_answer/BFCL_v4_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
    "possible_answer/BFCL_v4_parallel.json": "8a6aa19c1adddc6a5a2f7e40f9dbf30cc7e95815e7b830c90589ab318229e0f0",
    "possible_answer/BFCL_v4_parallel_multiple.json": "5ebf24f458c1f16300c05505d83d6f0a1b68b79be273a033febd0d4f840507e3",
    "possible_answer/BFCL_v4_live_simple.json": "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce",
    "possible_answer/BFCL_v4_live_multiple.json": "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
    "possible_answer/BFCL_v4_live_parallel.json": "8a9f189ff0e832ebbbbdade1fd95a7dbcc67406e9177df3f0aad76f59ab00350",
    "possible_answer/BFCL_v4_live_parallel_multiple.json": "f5b5f360556c5feb51db46fb9f56ee4b304f4b45b161599bbb14161c98a2873f",
    "BFCL_v4_multi_turn_base.json": "1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a",
    "BFCL_v4_multi_turn_miss_func.json": "87d28ce10e37d864b72de85d5732eef2a867b241d6c1c99b4ae682c9e3ea921c",
    "BFCL_v4_multi_turn_miss_param.json": "f0c66dda3795f5f53e3e1c0cc8ba0246b6761c8f58bdba8317203bf451ab8838",
    "BFCL_v4_multi_turn_long_context.json": "78c3268c5cc8e97c0f4ec6c811b3b9a2bba14323b1830b7a874b06d822749324",
    "possible_answer/BFCL_v4_multi_turn_base.json": "1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc",
    "possible_answer/BFCL_v4_multi_turn_miss_func.json": "69e679b806d1c871b05393a4b95583bb973248e5b8d96c2d7f4ca05e29fc32e6",
    "possible_answer/BFCL_v4_multi_turn_miss_param.json": "59c442901779e2c31c33abcd566d032e03736e5ad8069de2fe05489873046ecf",
    "possible_answer/BFCL_v4_multi_turn_long_context.json": "e82aa0e839c39d23c64a05834f7ae024d7c9738ec21738ecee78dc876e3d0d18",
    "BFCL_v4_memory.json": "40fc21d4528af53c6b44204def89e81515d0101654229e2c2e82bbf5f047b14f",
    "BFCL_v4_web_search.json": "6fc41d96d003dc849028966a782923560d2fc127ed2088aa967f06daaafa4268",
    "possible_answer/BFCL_v4_memory.json": "2355cf8d842f94af6bcb7bfa6ad2f9e472bc6d825d6ecd45702cfc41e27d7e5d",
    "possible_answer/BFCL_v4_web_search.json": "771cab45fdad5744563456801d4623a42c5a358f10514b9b9105d2f6052b4999",
    "multi_turn_func_doc/gorilla_file_system.json": "c4c1b741c71e2a17c97a5dc9c4a91d89978c4eaece56494d14272d5df6c650e9",
    "multi_turn_func_doc/math_api.json": "83fa31708c89442bdcf12ac4dfbe3be8663ec9183a1b1524a9fda98164bc7e0b",
    "multi_turn_func_doc/memory_kv.json": "96480cd9cbd3d4a34cd8f78879bc6622731768a26e080f7a34782e36e3402287",
    "multi_turn_func_doc/memory_rec_sum.json": "4ceff946df00983c7f0b95d1b70ae402247f3312fb898bac9ecfa1de52f5a783",
    "multi_turn_func_doc/memory_vector.json": "917908ca99fdd01e203274b7ec6eb2347fa91d57f3c6341e20945cc9c9d746cf",
    "multi_turn_func_doc/message_api.json": "4d58ea933a5d2b280d7a52366617fb47fecd333d7b0e08e724db6fa12fb5f847",
    "multi_turn_func_doc/posting_api.json": "87f9fc404a06e4107d7c366f1399c596440e84c16cb119faf374b2f532d86e8b",
    "multi_turn_func_doc/ticket_api.json": "31324e0380782664fe82ba05bd23517808ad55692b45a5fe7c3d4d395fe4f0f8",
    "multi_turn_func_doc/trading_bot.json": "1a7933fd8f0cb8fbec38aeae05cb8c24132747cad4904fe1c13ac3d01fc22c2d",
    "multi_turn_func_doc/travel_booking.json": "f17b950c13adddf41d0848077df58788252e4c2e7cad5cfa71c8c4bf04f57b26",
    "multi_turn_func_doc/vehicle_control.json": "0c8a66292844874ef7b168f343bc394d8615d2d9e1f4387999a9ee23011eac78",
    "multi_turn_func_doc/web_search.json": "61fcee411e35f7ff67415e18cd67276615cf06e1c8841a683d2d997dbb46eac5",
}

_COUNTS = {
    "simple_python": 400,
    "simple_java": 100,
    "simple_javascript": 50,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "irrelevance": 240,
    "live_simple": 258,
    "live_multiple": 1053,
    "live_parallel": 16,
    "live_parallel_multiple": 24,
    "live_irrelevance": 884,
    "live_relevance": 16,
}


def _cache_dir() -> Path:
    return Path(user_cache_dir("openbench")) / "bfcl" / BFCL_REVISION


def _ensure_file(relative_path: str) -> Path:
    expected = _FILE_SHA256[relative_path]
    path = _cache_dir() / relative_path
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"{BFCL_BASE_URL}/{relative_path}", timeout=120
    ) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected:
        raise ValueError(
            f"BFCL checksum mismatch for {relative_path}: expected {expected}, got {digest}"
        )
    path.write_bytes(content)
    return path


def _load_jsonl(relative_path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _ensure_file(relative_path).read_text().splitlines()
    ]


def _messages(question: list[list[dict[str, str]]]) -> list[Any]:
    messages: list[Any] = []
    for message in question[0]:
        content = str(message["content"])
        if message["role"] == "system":
            messages.append(ChatMessageSystem(content=content))
        else:
            messages.append(ChatMessageUser(content=content))
    return messages


def get_bfcl_v4_single_turn_dataset(
    categories: list[str] | tuple[str, ...] | None = None,
) -> MemoryDataset:
    """Load pinned BFCL v4 single-turn categories and public ground truths."""

    selected = tuple(categories or SINGLE_TURN_CATEGORIES)
    unknown = set(selected) - set(SINGLE_TURN_CATEGORIES)
    if unknown:
        raise ValueError(
            f"Unsupported BFCL v4 single-turn categories: {sorted(unknown)}"
        )

    samples: list[Sample] = []
    for category in selected:
        questions = _load_jsonl(f"BFCL_v4_{category}.json")
        if category in {"irrelevance", "live_irrelevance", "live_relevance"}:
            answers_by_id: dict[str, list[dict[str, Any]]] = {}
        else:
            answers = _load_jsonl(f"possible_answer/BFCL_v4_{category}.json")
            answers_by_id = {
                str(answer["id"]): list(answer["ground_truth"]) for answer in answers
            }
        if len(questions) != _COUNTS[category]:
            raise ValueError(
                f"BFCL {category} expected {_COUNTS[category]} rows, got {len(questions)}"
            )
        for question in questions:
            sample_id = str(question["id"])
            expected = answers_by_id.get(sample_id, [])
            samples.append(
                Sample(
                    id=sample_id,
                    input=_messages(question["question"]),
                    target=json.dumps(expected),
                    metadata={
                        "category": category,
                        "functions": question["function"],
                        "expected_calls": expected,
                        "bfcl_revision": BFCL_REVISION,
                        "license": BFCL_LICENSE,
                    },
                )
            )
    return MemoryDataset(samples=samples, name="bfcl_v4_single_turn")


def _load_function_docs(class_names: list[str]) -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    for class_name in class_names:
        functions.extend(
            _load_jsonl(f"multi_turn_func_doc/{_FUNCTION_DOCS[class_name]}")
        )
    return functions


def get_bfcl_v4_multi_turn_dataset(
    categories: list[str] | tuple[str, ...] | None = None,
) -> MemoryDataset:
    """Load the four pinned BFCL v4 multi-turn categories."""

    selected = tuple(categories or MULTI_TURN_CATEGORIES)
    unknown = set(selected) - set(MULTI_TURN_CATEGORIES)
    if unknown:
        raise ValueError(f"Unsupported BFCL multi-turn categories: {sorted(unknown)}")

    samples: list[Sample] = []
    for category in selected:
        questions = _load_jsonl(f"BFCL_v4_{category}.json")
        answers = _load_jsonl(f"possible_answer/BFCL_v4_{category}.json")
        answers_by_id = {str(row["id"]): row["ground_truth"] for row in answers}
        if len(questions) != _MULTI_TURN_COUNT:
            raise ValueError(
                f"BFCL {category} expected {_MULTI_TURN_COUNT} rows, got {len(questions)}"
            )
        for question in questions:
            sample_id = str(question["id"])
            all_functions = _load_function_docs(question["involved_classes"])
            missed: dict[str, list[dict[str, Any]]] = {}
            for turn, names in question.get("missed_function", {}).items():
                missed[str(turn)] = [
                    function for function in all_functions if function["name"] in names
                ]
            missed_names = {
                function["name"]
                for functions in missed.values()
                for function in functions
            }
            initial_functions = [
                function
                for function in all_functions
                if function["name"] not in missed_names
                and function["name"] not in question.get("excluded_function", [])
            ]
            turns = question["question"]
            expected = answers_by_id[sample_id]
            samples.append(
                Sample(
                    id=sample_id,
                    input=_messages([turns[0]]),
                    target=json.dumps(expected),
                    metadata={
                        "category": category,
                        "turns": turns,
                        "functions": initial_functions,
                        "missed_functions": missed,
                        "initial_config": question.get("initial_config", {}),
                        "involved_classes": question["involved_classes"],
                        "ground_truth": expected,
                        "bfcl_revision": BFCL_REVISION,
                        "license": BFCL_LICENSE,
                    },
                )
            )
    return MemoryDataset(samples=samples, name="bfcl_v4_multi_turn")


def get_bfcl_v4_agentic_dataset(
    categories: list[str] | tuple[str, ...] | None = None,
) -> MemoryDataset:
    """Load reproducible offline variants of BFCL v4 memory and web-search."""

    selected = tuple(categories or AGENTIC_CATEGORIES)
    unknown = set(selected) - set(AGENTIC_CATEGORIES)
    if unknown:
        raise ValueError(f"Unsupported BFCL agentic categories: {sorted(unknown)}")

    memory_questions = _load_jsonl("BFCL_v4_memory.json")
    memory_answers = {
        str(row["id"]): row
        for row in _load_jsonl("possible_answer/BFCL_v4_memory.json")
    }
    web_questions = _load_jsonl("BFCL_v4_web_search.json")
    web_answers = {
        str(row["id"]): row
        for row in _load_jsonl("possible_answer/BFCL_v4_web_search.json")
    }
    samples: list[Sample] = []
    for category in selected:
        is_memory = category.startswith("memory_")
        questions = memory_questions if is_memory else web_questions
        answers = memory_answers if is_memory else web_answers
        class_name = (
            f"MemoryAPI_{category.removeprefix('memory_')}"
            if is_memory
            else "WebSearchAPI"
        )
        functions = _load_function_docs([class_name])
        for question in questions:
            source_id = str(question["id"])
            answer = answers[source_id]
            sample_id = source_id.replace(
                "memory" if is_memory else "web_search", category
            )
            samples.append(
                Sample(
                    id=sample_id,
                    input=_messages(question["question"]),
                    target=json.dumps(answer["ground_truth"]),
                    metadata={
                        "category": category,
                        "functions": functions,
                        "expected_answers": answer["ground_truth"],
                        "frozen_source": answer["source"],
                        "show_snippet": category != "web_search_no_snippet",
                        "bfcl_revision": BFCL_REVISION,
                        "license": BFCL_LICENSE,
                        "offline_adaptation": True,
                    },
                )
            )
    return MemoryDataset(samples=samples, name="bfcl_v4_agentic_offline")
