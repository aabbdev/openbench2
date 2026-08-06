"""Global PIQA v1 loader for parallel and non-parallel components."""

from __future__ import annotations

from collections.abc import Iterable

from datasets import get_dataset_config_names, load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import MemoryDataset, Sample

DATASETS = {
    "nonparallel": (
        "mrlbenchmarks/global-piqa-nonparallel",
        "6777742fa3634c0583cda3b7f8a482ea7b1b0937",
    ),
    "parallel": (
        "mrlbenchmarks/global-piqa-parallel",
        "b0b18516a8bc2cb1106bce3dd4db32848ca715ea",
    ),
}


def _prompt(record: dict, component: str) -> str:
    choices = [
        record[f"solution{index}"]
        for index in range(2 if component == "nonparallel" else 4)
    ]
    options = "\n\n".join(
        f"Option {chr(65 + index)}: {choice}" for index, choice in enumerate(choices)
    )
    if component == "nonparallel":
        return (
            "Given the following situation, which option is more likely to be correct?\n\n"
            f"Situation:\n{record['prompt']}\n\n{options}\n\n"
            'Your response should end with "The best answer is: [answer_letter]" '
            "where [answer_letter] is one of A or B."
        )
    return (
        f"{record['prompt']}\n\n{options}\n\n"
        'Your response should end with "The best answer is: [answer_letter]" '
        "where [answer_letter] is one of A, B, C, or D."
    )


def get_global_piqa_dataset(
    *,
    components: Iterable[str] = ("nonparallel", "parallel"),
    languages: Iterable[str] | None = None,
) -> MemoryDataset:
    """Load Global PIQA generation mode with immutable source revisions."""
    selected_components = tuple(components)
    unknown = sorted(set(selected_components) - DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown Global PIQA components: {unknown}")

    selected_languages = set(languages) if languages is not None else None
    samples: list[Sample] = []
    for component in selected_components:
        path, revision = DATASETS[component]
        configs = get_dataset_config_names(path, revision=revision)
        if selected_languages is not None:
            configs = [config for config in configs if config in selected_languages]
        for language in configs:
            records = load_dataset(
                path,
                name=language,
                split="test",
                revision=revision,
            )
            for index, record in enumerate(records):
                target = chr(65 + int(record["label"]))
                samples.append(
                    Sample(
                        id=f"{component}-{language}-{record.get('example_id', index)}",
                        input=_prompt(record, component),
                        target=target,
                        metadata={
                            "component": component,
                            "language": language,
                            "example_id": record.get("example_id"),
                        },
                    )
                )
    return MemoryDataset(samples=samples, name="global_piqa")
