"""LiveCodeBench v6 code-generation evaluation."""

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.livecodebench import get_livecodebench_v6_dataset
from openbench.scorers.livecodebench import livecodebench_scorer

TASK_DIR = Path(__file__).parent
COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()


@task
def livecodebench_v6(
    start_date: str | None = None,
    end_date: str | None = None,
    test_timeout: int = 6,
    total_timeout: int = 600,
) -> Task:
    """Run the pinned cumulative LiveCodeBench release-v6 benchmark.

    The official protocol samples 10 completions at temperature 0.2 and reports
    pass@1 and pass@5. Optional date boundaries are inclusive.
    """

    return Task(
        name="livecodebench_v6",
        dataset=get_livecodebench_v6_dataset(
            start_date=start_date,
            end_date=end_date,
        ),
        solver=generate(),
        scorer=livecodebench_scorer(
            test_timeout=test_timeout,
            total_timeout=total_timeout,
        ),
        sandbox=("docker", str(COMPOSE_PATH)),
        epochs=Epochs(10, reducer=["mean", "pass_at_1", "pass_at_5"]),
        config=GenerateConfig(
            temperature=0.2,
            top_p=0.95,
            max_tokens=2000,
            stop_seqs=["###"],
        ),
    )
