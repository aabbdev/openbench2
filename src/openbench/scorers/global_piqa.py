"""Generation scorer and hierarchical macro metric for Global PIQA."""

import re
from collections import defaultdict
from collections.abc import Callable

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState


@metric
def global_piqa_macro_accuracy() -> Metric:
    """Average examples by language, then languages by component, then components."""

    def calculate(scores: list[SampleScore]) -> Value:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample in scores:
            metadata = sample.score.metadata or {}
            value = 1.0 if sample.score.value == CORRECT else 0.0
            grouped[(metadata["component"], metadata["language"])].append(value)

        components: dict[str, list[float]] = defaultdict(list)
        for (component, _), values in grouped.items():
            components[component].append(sum(values) / len(values))
        component_scores = [sum(values) / len(values) for values in components.values()]
        return (
            sum(component_scores) / len(component_scores) if component_scores else 0.0
        )

    return calculate


@scorer(metrics=[global_piqa_macro_accuracy()])
def global_piqa_scorer() -> Callable:
    """Extract the answer letter using the official generation-mode patterns."""

    async def score(state: TaskState, target: Target) -> Score:
        patterns = (
            r"[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)[^A-D]*([A-D])",
            r"[Aa]nswer\s*:[^A-D]*([A-D])",
            r"\\boxed\{([A-D])\}",
        )
        matches = [
            match
            for pattern in patterns
            for match in re.findall(pattern, state.output.completion)
        ]
        answer = matches[-1].upper() if matches else ""
        return Score(
            value=CORRECT if answer == target.text.upper() else INCORRECT,
            answer=answer,
            metadata={
                "component": state.metadata["component"],
                "language": state.metadata["language"],
            },
        )

    return score
