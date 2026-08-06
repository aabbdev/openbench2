from unittest.mock import patch

from openbench.config import BENCHMARKS, EVAL_GROUPS
from openbench.evals.bigbench_hard import (
    BBH_DATASET_REVISION,
    _bbh_free_response_task,
    _free_response_record,
)


MISSING_CONFIGS = {
    "boolean_expressions",
    "dyck_languages",
    "formal_fallacies",
    "hyperbaton",
    "multistep_arithmetic_two",
    "object_counting",
    "penguins_in_a_table",
    "web_of_lies",
    "word_sorting",
}


def test_bbh_free_response_keeps_full_target():
    sample = _free_response_record({"input": "Sort words", "target": "a b c"})

    assert sample.target == "a b c"
    assert "So the answer is" in sample.input


def test_bbh_new_tasks_are_pinned():
    with patch(
        "openbench.evals.bigbench_hard.hf_dataset",
        return_value=[_free_response_record({"input": "x", "target": "y"})],
    ) as load:
        _bbh_free_response_task("word_sorting")

    assert load.call_args.kwargs["revision"] == BBH_DATASET_REVISION
    assert load.call_args.kwargs["name"] == "word_sorting"


def test_bbh_complete_registry_and_group():
    expected_ids = {f"bbh_{name}" for name in MISSING_CONFIGS}

    assert expected_ids <= BENCHMARKS.keys()
    assert expected_ids <= set(EVAL_GROUPS["bbh"].benchmarks)
    assert len(EVAL_GROUPS["bbh"].benchmarks) == 27
