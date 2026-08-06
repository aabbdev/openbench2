"""C-Eval Chinese academic multiple-choice benchmark."""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.ceval import HARD_SUBJECTS, get_ceval_dataset
from openbench.scorers.mcq import create_mcq_scorer


def _ceval_task(*, hard: bool, split: str, shots: int) -> Task:
    return Task(
        dataset=get_ceval_dataset(
            subjects=HARD_SUBJECTS if hard else None,
            split=split,
            shots=shots,
        ),
        solver=generate(),
        scorer=create_mcq_scorer(group_keys=["category", "subject"])(),
        config=GenerateConfig(temperature=0.0, max_tokens=32),
        name="ceval_hard" if hard else "ceval",
    )


@task
def ceval(split: str = "val", shots: int = 5) -> Task:
    """Evaluate all 52 C-Eval subjects with official 0/5-shot prompting."""
    return _ceval_task(hard=False, split=split, shots=shots)


@task
def ceval_hard(split: str = "val", shots: int = 5) -> Task:
    """Evaluate the official eight-subject C-Eval Hard subset."""
    return _ceval_task(hard=True, split=split, shots=shots)
