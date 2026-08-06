from unittest.mock import patch

import pytest

from openbench.config import BENCHMARKS
from openbench.datasets.ceval import (
    DATASET_REVISION,
    HARD_SUBJECTS,
    SUBJECTS,
    get_ceval_dataset,
)


def _record(record_id: int, answer: str = "B") -> dict:
    return {
        "id": record_id,
        "question": "测试题",
        "A": "甲",
        "B": "乙",
        "C": "丙",
        "D": "丁",
        "answer": answer,
        "explanation": "",
    }


def test_ceval_subject_inventory():
    assert len(SUBJECTS) == 52
    assert len(HARD_SUBJECTS) == 8
    assert set(HARD_SUBJECTS) <= SUBJECTS.keys()


def test_ceval_five_shot_prompt_and_pinning():
    def fake_load(path, *, name, split, revision):
        assert path == "ceval/ceval-exam"
        assert name == "computer_network"
        assert revision == DATASET_REVISION
        return [_record(i) for i in range(5)] if split == "dev" else [_record(9)]

    with patch("openbench.datasets.ceval.load_dataset", side_effect=fake_load):
        dataset = get_ceval_dataset(subjects=["computer_network"], shots=5)

    sample = list(dataset)[0]
    assert sample.input.count("答案：B") == 5
    assert sample.input.endswith("答案：")
    assert sample.target == "B"
    assert sample.metadata == {"subject": "computer_network", "category": "STEM"}


@pytest.mark.parametrize("shots", [1, 4, 10])
def test_ceval_rejects_nonstandard_shot_counts(shots):
    with pytest.raises(ValueError, match="0-shot or 5-shot"):
        get_ceval_dataset(subjects=["computer_network"], shots=shots)


def test_ceval_registry_entries():
    assert BENCHMARKS["ceval"].function_name == "ceval"
    assert BENCHMARKS["ceval_hard"].function_name == "ceval_hard"
