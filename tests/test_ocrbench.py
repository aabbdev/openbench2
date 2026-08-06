from unittest.mock import patch

from PIL import Image

from openbench.config import BENCHMARKS
from openbench.datasets.ocrbench import DATASET_REVISION, get_ocrbench_dataset
from openbench.scorers.ocrbench import normalize_ocrbench_text


def test_ocrbench_loader_is_v1_and_pinned():
    rows = [
        {
            "dataset": "IIIT5K",
            "question": "what is written in the image?",
            "question_type": "Regular Text Recognition",
            "answer": ["CENTRE"],
            "image": Image.new("RGB", (2, 2), "white"),
        }
    ]
    with patch("openbench.datasets.ocrbench.load_dataset", return_value=rows) as load:
        sample = list(get_ocrbench_dataset())[0]

    assert load.call_args.kwargs == {
        "split": "test",
        "revision": DATASET_REVISION,
    }
    assert load.call_args.args == ("echo840/OCRBench",)
    assert sample.target == ["CENTRE"]
    assert sample.metadata["component"] == "text_recognition"


def test_ocrbench_normalization_modes():
    assert normalize_ocrbench_text("  CenTre\nText  ", hme=False) == "centre text"
    assert normalize_ocrbench_text(" x + y \n", hme=True) == "x+y"


def test_ocrbench_v1_and_v2_are_distinct_registry_entries():
    assert BENCHMARKS["ocrbench"].function_name == "ocrbench"
    assert BENCHMARKS["ocrbenchv2"].function_name == "ocrbenchv2"
