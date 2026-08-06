from inspect_ai import Task, task

from openbench.evals.matharena.matharena import matharena_task


@task
def aime_2026() -> Task:
    """Evaluate the combined AIME I and II 2026 dataset."""
    return matharena_task(
        dataset_path="MathArena/aime_2026",
        revision="d2de22f3c656b4f56cf8981212186377d1e23bc3",
        instruction=(
            "Please reason step by step, and put your final answer within "
            "\\boxed{{}}.\nThe answer is an integer between 0 and 999 inclusive."
        ),
        default_temperature=0.6,
        default_max_tokens=8000,
        default_epochs=4,
        name="aime_2026",
    )
