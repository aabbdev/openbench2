"""Official OJBench language tracks with DMOJ scoring."""

from pathlib import Path
from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai.solver import generate

from openbench.datasets.ojbench import get_ojbench_dataset
from openbench.scorers.ojbench import ojbench_scorer

COMPOSE_PATH = (Path(__file__).parent / "compose.yaml").resolve()


def _ojbench_task(language: Literal["python", "cpp"], total_timeout: int) -> Task:
    return Task(
        name=f"ojbench_{language}",
        dataset=get_ojbench_dataset(language),
        solver=generate(),
        scorer=ojbench_scorer(total_timeout=total_timeout),
        sandbox=("docker", str(COMPOSE_PATH)),
        epochs=Epochs(8, reducer=["mean", "pass_at_1", "pass_at_8"]),
    )


@task
def ojbench_python(total_timeout: int = 3600) -> Task:
    """Evaluate the 232 official Python prompts with eight samples each."""

    return _ojbench_task("python", total_timeout)


@task
def ojbench_cpp(total_timeout: int = 3600) -> Task:
    """Evaluate the 232 official C++ prompts with eight samples each."""

    return _ojbench_task("cpp", total_timeout)
