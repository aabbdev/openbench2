from unittest.mock import patch

import pytest

from openbench.config import BENCHMARKS, EVAL_GROUPS
from openbench.evals.matharena.aime_2026.aime_2026 import aime_2026
from openbench.evals.matharena.hmmt_feb_2026.hmmt_feb_2026 import hmmt_feb_2026
from openbench.evals.matharena.hmmt_nov_2025.hmmt_nov_2025 import hmmt_nov_2025
from openbench.scorers.robust_boxed import _exact_arithmetic_value, extract_boxed_answer


@pytest.mark.parametrize(
    ("factory", "path", "revision"),
    [
        (
            aime_2026,
            "MathArena/aime_2026",
            "d2de22f3c656b4f56cf8981212186377d1e23bc3",
        ),
        (
            hmmt_nov_2025,
            "MathArena/hmmt_nov_2025",
            "118dbfb45c4c9467c672268ed55166642897aa46",
        ),
        (
            hmmt_feb_2026,
            "MathArena/hmmt_feb_2026",
            "02fba4f74d8e68e73e66a02d540fd979c05c274c",
        ),
    ],
)
def test_new_matharena_factories_are_pinned(factory, path, revision):
    with patch(
        f"{factory.__module__}.matharena_task", return_value=object()
    ) as task_factory:
        factory.__wrapped__()

    assert task_factory.call_args.kwargs["dataset_path"] == path
    assert task_factory.call_args.kwargs["revision"] == revision


def test_new_matharena_registry_entries_and_group():
    expected = {"aime_2026", "hmmt_nov_2025", "hmmt_feb_2026"}

    assert expected <= BENCHMARKS.keys()
    assert expected <= set(EVAL_GROUPS["matharena"].benchmarks)


def test_matharena_fraction_answer_parsing():
    answer = extract_boxed_answer(r"Reasoning... \boxed{-\frac{1}{21}}")

    assert answer == r"-\frac{1}{21}"
    assert _exact_arithmetic_value(answer) == _exact_arithmetic_value("-1/21")
