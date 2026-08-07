"""Public LiveBench coding release 2024-11-25."""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.livebench import get_livebench_coding_dataset
from openbench.scorers.livecodebench import livebench_coding_scorer

COMPOSE_PATH = (Path(__file__).parents[1] / "livecodebench" / "compose.yaml").resolve()


@task
def livebench_coding_2024_11_25(
    test_timeout: int = 6,
    total_timeout: int = 600,
) -> Task:
    """Run the latest fully public LiveBench coding release."""

    return Task(
        name="livebench_coding_2024_11_25",
        dataset=get_livebench_coding_dataset(),
        solver=generate(),
        scorer=livebench_coding_scorer(
            test_timeout=test_timeout,
            total_timeout=total_timeout,
        ),
        sandbox=("docker", str(COMPOSE_PATH)),
        config=GenerateConfig(temperature=0, max_tokens=4096),
    )
