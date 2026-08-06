from inspect_ai import Task, task

from openbench.evals.matharena.matharena import matharena_task


@task
def hmmt_nov_2025() -> Task:
    """Evaluate HMMT November 2025."""
    return matharena_task(
        dataset_path="MathArena/hmmt_nov_2025",
        revision="118dbfb45c4c9467c672268ed55166642897aa46",
        instruction=(
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        ),
        default_temperature=0.6,
        default_max_tokens=16000,
        default_epochs=4,
        name="hmmt_nov_2025",
    )
