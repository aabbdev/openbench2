import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from openbench.config import BENCHMARKS
from openbench.evals.gsm8k_hard import DATASET_REVISION, record_to_sample
from openbench.scorers.grade_school_math import numeric_tolerance_scorer


def test_gsm8k_hard_record_conversion():
    sample = record_to_sample({"input": "What is 2 + 2?", "target": 4.0})

    assert "What is 2 + 2?" in sample.input
    assert sample.target == "4.0"
    assert sample.metadata == {"answer_prefix": "Answer"}


@pytest.mark.asyncio
async def test_gsm8k_hard_uses_official_strict_tolerance():
    scorer = numeric_tolerance_scorer(tolerance=1e-3)
    state = TaskState(
        model="mock",
        sample_id="1",
        epoch=1,
        input="question",
        messages=[],
        output=ModelOutput.from_content("mock", "Answer: 1.0009"),
        metadata={"answer_prefix": "Answer"},
    )

    assert (await scorer(state, Target("1.0"))).value == 1.0
    # The canonical PAL scorer uses binary floats, where 1.001 - 1.0 is
    # marginally below 0.001. Use a value unambiguously outside the threshold.
    state.output = ModelOutput.from_content("mock", "Answer: 1.0011")
    assert (await scorer(state, Target("1.0"))).value == 0.0


def test_gsm8k_hard_registry_and_revision():
    assert BENCHMARKS["gsm8k_hard"].function_name == "gsm8k_hard"
    assert DATASET_REVISION == "960448f73503112d4226baeb8eb41d3fb5ae2506"
