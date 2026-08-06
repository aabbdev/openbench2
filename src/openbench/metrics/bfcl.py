"""Official-style category aggregation for BFCL v4 single-turn tasks."""

from __future__ import annotations

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric


def _category_accuracy(scores: list[SampleScore]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in scores:
        if sample.sample_metadata is None:
            continue
        grouped[str(sample.sample_metadata["category"])].append(sample.score.as_float())
    return {
        category: sum(values) / len(values)
        for category, values in grouped.items()
        if values
    }


@metric
def bfcl_v4_single_turn_metrics() -> Metric:
    def calculate(scores: list[SampleScore]) -> Value:
        grouped: dict[str, list[float]] = defaultdict(list)
        for sample in scores:
            if sample.sample_metadata is not None:
                grouped[str(sample.sample_metadata["category"])].append(
                    sample.score.as_float()
                )
        accuracy = _category_accuracy(scores)

        def mean(categories: list[str]) -> float:
            values = [
                accuracy[category] for category in categories if category in accuracy
            ]
            return sum(values) / len(values) if values else 0.0

        simple_categories = ["simple_python", "simple_java", "simple_javascript"]
        has_simple = any(name in accuracy for name in simple_categories)
        simple_non_live = mean(simple_categories)
        non_live_values = [
            *([simple_non_live] if has_simple else []),
            *(
                accuracy[name]
                for name in ["multiple", "parallel", "parallel_multiple"]
                if name in accuracy
            ),
        ]
        non_live = (
            sum(non_live_values) / len(non_live_values) if non_live_values else 0.0
        )

        live_categories = [
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ]
        live_total = sum(len(grouped[name]) for name in live_categories)
        live = (
            sum(
                accuracy.get(name, 0.0) * len(grouped[name]) for name in live_categories
            )
            / live_total
            if live_total
            else 0.0
        )
        irrelevance = mean(["irrelevance", "live_irrelevance"])
        sections = [
            *([non_live] if non_live_values else []),
            *([live] if live_total else []),
            *(
                [irrelevance]
                if "irrelevance" in accuracy or "live_irrelevance" in accuracy
                else []
            ),
        ]
        single_turn = sum(sections) / len(sections) if sections else 0.0
        return {
            "single_turn": single_turn,
            "non_live": non_live,
            "live": live,
            "irrelevance": irrelevance,
            "relevance": accuracy.get("live_relevance", 0.0),
            **accuracy,
        }

    return calculate


@metric
def bfcl_v4_multi_turn_metrics() -> Metric:
    def calculate(scores: list[SampleScore]) -> Value:
        accuracy = _category_accuracy(scores)
        values = list(accuracy.values())
        return {
            "multi_turn": sum(values) / len(values) if values else 0.0,
            **accuracy,
        }

    return calculate


@metric
def bfcl_v4_agentic_metrics() -> Metric:
    def calculate(scores: list[SampleScore]) -> Value:
        accuracy = _category_accuracy(scores)

        def mean(names: list[str]) -> float:
            values = [accuracy[name] for name in names if name in accuracy]
            return sum(values) / len(values) if values else 0.0

        web = mean(["web_search_base", "web_search_no_snippet"])
        memory = mean(["memory_kv", "memory_vector", "memory_rec_sum"])
        sections = [
            *([web] if any(name.startswith("web_search") for name in accuracy) else []),
            *([memory] if any(name.startswith("memory_") for name in accuracy) else []),
        ]
        return {
            "agentic": sum(sections) / len(sections) if sections else 0.0,
            "web_search": web,
            "memory": memory,
            **accuracy,
        }

    return calculate


@metric
def bfcl_v4_offline_metrics() -> Metric:
    def calculate(scores: list[SampleScore]) -> Value:
        accuracy = _category_accuracy(scores)

        def mean(names: list[str]) -> float:
            values = [accuracy[name] for name in names if name in accuracy]
            return sum(values) / len(values) if values else 0.0

        simple = mean(["simple_python", "simple_java", "simple_javascript"])
        non_live = (
            sum(
                [
                    simple,
                    accuracy["multiple"],
                    accuracy["parallel"],
                    accuracy["parallel_multiple"],
                ]
            )
            / 4
        )
        live_names = [
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ]
        counts: dict[str, int] = defaultdict(int)
        for sample in scores:
            if sample.sample_metadata is not None:
                counts[str(sample.sample_metadata["category"])] += 1
        live_count = sum(counts[name] for name in live_names)
        live = sum(accuracy[name] * counts[name] for name in live_names) / live_count
        irrelevance = mean(["irrelevance", "live_irrelevance"])
        multi_turn = mean(
            [
                "multi_turn_base",
                "multi_turn_miss_func",
                "multi_turn_miss_param",
                "multi_turn_long_context",
            ]
        )
        web = mean(["web_search_base", "web_search_no_snippet"])
        memory = mean(["memory_kv", "memory_vector", "memory_rec_sum"])
        agentic = (web + memory) / 2
        overall = (
            0.1 * non_live
            + 0.1 * live
            + 0.1 * irrelevance
            + 0.3 * multi_turn
            + 0.4 * agentic
        )
        return {
            "overall_offline": overall,
            "non_live": non_live,
            "live": live,
            "irrelevance": irrelevance,
            "multi_turn": multi_turn,
            "agentic": agentic,
            "web_search": web,
            "memory": memory,
            **accuracy,
        }

    return calculate
