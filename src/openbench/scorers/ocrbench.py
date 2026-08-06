"""Official OCRBench v1 substring scorer."""

from collections.abc import Callable

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from openbench.metrics.grouped import grouped


def normalize_ocrbench_text(text: str, *, hme: bool) -> str:
    """Apply the normalization from OCRBench's reference evaluation script."""
    normalized = (
        text.replace("\n", "") if hme else text.lower().strip().replace("\n", " ")
    )
    return normalized.replace(" ", "") if hme else normalized


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(group_key="component", metric=accuracy(), all=False),
        grouped(group_key="question_type", metric=accuracy(), all=False),
    ]
)
def ocrbench_scorer() -> Callable:
    """Score each sample when any reference is a substring of the prediction."""

    async def score(state: TaskState, target: Target) -> Score:
        hme = state.metadata.get("dataset_name") == "HME100k"
        prediction = normalize_ocrbench_text(state.output.completion, hme=hme)
        references = [normalize_ocrbench_text(value, hme=hme) for value in target]
        correct = any(reference in prediction for reference in references)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=state.output.completion,
            explanation="OCRBench v1 normalized substring match",
        )

    return score
