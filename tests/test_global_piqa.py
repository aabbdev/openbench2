from unittest.mock import patch

from openbench.config import BENCHMARKS
from openbench.datasets.global_piqa import DATASETS, get_global_piqa_dataset


def test_global_piqa_loads_both_pinned_components_and_prompts():
    nonparallel = {
        "prompt": "Situation",
        "solution0": "A0",
        "solution1": "B0",
        "label": 1,
        "example_id": "n1",
    }
    parallel = {
        **nonparallel,
        "solution2": "C0",
        "solution3": "D0",
        "label": 2,
        "example_id": "p1",
    }

    def fake_load(path, *, name, split, revision):
        assert name == "eng_latn"
        assert split == "test"
        assert (
            revision
            == DATASETS["nonparallel" if "nonparallel" in path else "parallel"][1]
        )
        return [nonparallel if "nonparallel" in path else parallel]

    with (
        patch(
            "openbench.datasets.global_piqa.get_dataset_config_names",
            return_value=["eng_latn"],
        ),
        patch("openbench.datasets.global_piqa.load_dataset", side_effect=fake_load),
    ):
        samples = list(get_global_piqa_dataset())

    assert [sample.target for sample in samples] == ["B", "C"]
    assert "one of A or B" in samples[0].input
    assert "A, B, C, or D" in samples[1].input
    assert {sample.metadata["component"] for sample in samples} == {
        "parallel",
        "nonparallel",
    }


def test_global_piqa_registry_entry():
    assert BENCHMARKS["global_piqa"].function_name == "global_piqa"
