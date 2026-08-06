"""Global PIQA generation-mode evaluation."""

from collections.abc import Iterable

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.global_piqa import get_global_piqa_dataset
from openbench.scorers.global_piqa import global_piqa_scorer


@task
def global_piqa(
    components: Iterable[str] = ("nonparallel", "parallel"),
    languages: Iterable[str] | None = None,
) -> Task:
    """Evaluate both Global PIQA components using the official generation mode."""
    return Task(
        dataset=get_global_piqa_dataset(components=components, languages=languages),
        solver=generate(),
        scorer=global_piqa_scorer(),
        config=GenerateConfig(temperature=0.8, top_p=0.95, max_tokens=2048),
        name="global_piqa",
    )
