from inspect_ai import Task, task

from openbench.evals.matharena.matharena import matharena_task


@task
def hmmt_feb_2026() -> Task:
    """Evaluate HMMT February 2026."""
    return matharena_task(
        dataset_path="MathArena/hmmt_feb_2026",
        revision="02fba4f74d8e68e73e66a02d540fd979c05c274c",
        instruction=(
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        ),
        default_temperature=0.6,
        default_max_tokens=16000,
        default_epochs=4,
        name="hmmt_feb_2026",
    )
