"""GSM-Hard, the numerically perturbed GSM8K evaluation set."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.scorers.grade_school_math import numeric_tolerance_scorer

DATASET_REVISION = "960448f73503112d4226baeb8eb41d3fb5ae2506"
PROMPT_TEMPLATE = """Solve this math problem. Show your reasoning, then put only the
numeric result after `Answer:` on the final line.

{question}"""


def record_to_sample(record: dict) -> Sample:
    """Convert a canonical GSM-Hard record to an Inspect sample."""
    return Sample(
        input=PROMPT_TEMPLATE.format(question=record["input"]),
        target=str(record["target"]),
        metadata={"answer_prefix": "Answer"},
    )


@task
def gsm8k_hard() -> Task:
    """Evaluate the 1,319-example GSM-Hard dataset from the PAL authors."""
    return Task(
        dataset=hf_dataset(
            path="reasoning-machines/gsm-hard",
            revision=DATASET_REVISION,
            split="train",
            sample_fields=record_to_sample,
        ),
        solver=generate(),
        scorer=numeric_tolerance_scorer(tolerance=1e-3),
        config=GenerateConfig(temperature=0.0, max_tokens=2048),
        name="gsm8k_hard",
    )
