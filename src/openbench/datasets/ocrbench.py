"""OCRBench v1 dataset loader."""

from __future__ import annotations

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

from openbench.utils.image import image_bytes_to_data_uri, pil_image_to_bytes

DATASET_REVISION = "92a54bd1384387c178d5a07140a2d85e0a3d12e1"


def _component(question_type: str) -> str:
    if "Recognition" in question_type and "Expression" not in question_type:
        return "text_recognition"
    if "Scene Text" in question_type:
        return "scene_text_vqa"
    if "Doc" in question_type:
        return "document_vqa"
    if "Information Extraction" in question_type:
        return "key_information_extraction"
    if "Expression" in question_type:
        return "handwritten_math_expression"
    return question_type


def get_ocrbench_dataset() -> MemoryDataset:
    """Load the official 1,000-example OCRBench v1 test set."""
    records = load_dataset(
        "echo840/OCRBench",
        split="test",
        revision=DATASET_REVISION,
    )
    samples: list[Sample] = []
    for index, record in enumerate(records):
        image = record["image"]
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image_uri = image_bytes_to_data_uri(pil_image_to_bytes(image, format="JPEG"))
        question_type = str(record["question_type"])
        samples.append(
            Sample(
                id=f"ocrbench-{index}",
                input=[
                    ChatMessageUser(
                        content=[
                            ContentImage(image=image_uri),
                            ContentText(text=str(record["question"])),
                        ]
                    )
                ],
                target=list(record["answer"]),
                metadata={
                    "dataset_name": str(record["dataset"]),
                    "question_type": question_type,
                    "component": _component(question_type),
                },
            )
        )
    return MemoryDataset(samples=samples, name="OCRBench")
