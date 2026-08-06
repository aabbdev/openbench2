"""HumanEval+ and MBPP+ with isolated differential testing."""

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.evalplus import get_evalplus_dataset
from openbench.scorers.evalplus import evalplus_scorer

COMPOSE_PATH = (Path(__file__).parent / "compose.yaml").resolve()


def _evalplus_task(dataset: str, epochs: int, total_timeout: int) -> Task:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    reducers = ["mean", "pass_at_1"]
    if epochs >= 10:
        reducers.append("pass_at_10")
    if epochs >= 100:
        reducers.append("pass_at_100")
    return Task(
        name=f"{dataset}_plus",
        dataset=get_evalplus_dataset(dataset),
        solver=generate(),
        scorer=evalplus_scorer(total_timeout=total_timeout),
        sandbox=("docker", str(COMPOSE_PATH)),
        epochs=Epochs(epochs, reducer=reducers),
        config=GenerateConfig(temperature=0.2, top_p=0.95, max_tokens=2048),
    )


@task
def humaneval_plus(epochs: int = 1, total_timeout: int = 900) -> Task:
    """Evaluate HumanEval+ v0.1.10 base and augmented tests."""
    return _evalplus_task("humaneval", epochs, total_timeout)


@task
def mbpp_plus(epochs: int = 1, total_timeout: int = 900) -> Task:
    """Evaluate MBPP+ v0.2.0 base and augmented tests."""
    return _evalplus_task("mbpp", epochs, total_timeout)
