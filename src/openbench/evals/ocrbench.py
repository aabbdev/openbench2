"""OCRBench v1 evaluation."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.ocrbench import get_ocrbench_dataset
from openbench.scorers.ocrbench import ocrbench_scorer


@task
def ocrbench() -> Task:
    """Evaluate OCRBench v1, kept distinct from OCRBench v2."""
    return Task(
        dataset=get_ocrbench_dataset(),
        solver=generate(),
        scorer=ocrbench_scorer(),
        config=GenerateConfig(temperature=0.0, max_tokens=100),
        name="ocrbench",
    )
