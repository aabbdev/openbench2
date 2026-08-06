"""Pinned C-Eval dataset loader with official Chinese prompts."""

from __future__ import annotations

from collections.abc import Iterable

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import MemoryDataset, Sample

DATASET_PATH = "ceval/ceval-exam"
DATASET_REVISION = "617524a00b307ff6f9933702f724131fe12ca7ce"

# subject: (Chinese display name, official super-category)
SUBJECTS = {
    "computer_network": ("计算机网络", "STEM"),
    "operating_system": ("操作系统", "STEM"),
    "computer_architecture": ("计算机组成", "STEM"),
    "college_programming": ("大学编程", "STEM"),
    "college_physics": ("大学物理", "STEM"),
    "college_chemistry": ("大学化学", "STEM"),
    "advanced_mathematics": ("高等数学", "STEM"),
    "probability_and_statistics": ("概率统计", "STEM"),
    "discrete_mathematics": ("离散数学", "STEM"),
    "electrical_engineer": ("注册电气工程师", "STEM"),
    "metrology_engineer": ("注册计量师", "STEM"),
    "high_school_mathematics": ("高中数学", "STEM"),
    "high_school_physics": ("高中物理", "STEM"),
    "high_school_chemistry": ("高中化学", "STEM"),
    "high_school_biology": ("高中生物", "STEM"),
    "middle_school_mathematics": ("初中数学", "STEM"),
    "middle_school_biology": ("初中生物", "STEM"),
    "middle_school_physics": ("初中物理", "STEM"),
    "middle_school_chemistry": ("初中化学", "STEM"),
    "veterinary_medicine": ("兽医学", "STEM"),
    "college_economics": ("大学经济学", "Social Science"),
    "business_administration": ("工商管理", "Social Science"),
    "marxism": ("马克思主义基本原理", "Social Science"),
    "mao_zedong_thought": (
        "毛泽东思想和中国特色社会主义理论体系概论",
        "Social Science",
    ),
    "education_science": ("教育学", "Social Science"),
    "teacher_qualification": ("教师资格", "Social Science"),
    "high_school_politics": ("高中政治", "Social Science"),
    "high_school_geography": ("高中地理", "Social Science"),
    "middle_school_politics": ("初中政治", "Social Science"),
    "middle_school_geography": ("初中地理", "Social Science"),
    "modern_chinese_history": ("近代史纲要", "Humanities"),
    "ideological_and_moral_cultivation": ("思想道德修养与法律基础", "Humanities"),
    "logic": ("逻辑学", "Humanities"),
    "law": ("法学", "Humanities"),
    "chinese_language_and_literature": ("中国语言文学", "Humanities"),
    "art_studies": ("艺术学", "Humanities"),
    "professional_tour_guide": ("导游资格", "Humanities"),
    "legal_professional": ("法律职业资格", "Humanities"),
    "high_school_chinese": ("高中语文", "Humanities"),
    "high_school_history": ("高中历史", "Humanities"),
    "middle_school_history": ("初中历史", "Humanities"),
    "civil_servant": ("公务员", "Other"),
    "sports_science": ("体育学", "Other"),
    "plant_protection": ("植物保护", "Other"),
    "basic_medicine": ("基础医学", "Other"),
    "clinical_medicine": ("临床医学", "Other"),
    "urban_and_rural_planner": ("注册城乡规划师", "Other"),
    "accountant": ("注册会计师", "Other"),
    "fire_engineer": ("注册消防工程师", "Other"),
    "environmental_impact_assessment_engineer": ("环境影响评价工程师", "Other"),
    "tax_accountant": ("税务师", "Other"),
    "physician": ("医师资格", "Other"),
}

HARD_SUBJECTS = (
    "advanced_mathematics",
    "discrete_mathematics",
    "probability_and_statistics",
    "college_chemistry",
    "college_physics",
    "high_school_mathematics",
    "high_school_chemistry",
    "high_school_physics",
)


def _format_question(record: dict, answer: str | None = None) -> str:
    text = (
        f"{record['question']}\n"
        f"A. {record['A']}\nB. {record['B']}\n"
        f"C. {record['C']}\nD. {record['D']}\n答案："
    )
    return text + (answer if answer is not None else "")


def get_ceval_dataset(
    *,
    subjects: Iterable[str] | None = None,
    split: str = "val",
    shots: int = 5,
) -> MemoryDataset:
    """Load C-Eval subjects and apply the official answer-only prompt."""
    if split not in {"val", "test"}:
        raise ValueError("C-Eval evaluation split must be 'val' or 'test'")
    if shots not in {0, 5}:
        raise ValueError("C-Eval officially supports 0-shot or 5-shot prompting")

    selected = tuple(subjects) if subjects is not None else tuple(SUBJECTS)
    unknown = sorted(set(selected) - SUBJECTS.keys())
    if unknown:
        raise ValueError(f"Unknown C-Eval subjects: {unknown}")

    samples: list[Sample] = []
    for subject in selected:
        chinese_name, category = SUBJECTS[subject]
        prefix = (
            f"以下是中国关于{chinese_name}考试的单项选择题，请选出其中的正确答案。\n\n"
        )
        if shots:
            dev = load_dataset(
                DATASET_PATH,
                name=subject,
                split="dev",
                revision=DATASET_REVISION,
            )
            prefix += "\n\n".join(
                _format_question(record, str(record["answer"])) for record in dev
            )
            prefix += "\n\n"

        records = load_dataset(
            DATASET_PATH,
            name=subject,
            split=split,
            revision=DATASET_REVISION,
        )
        for record in records:
            samples.append(
                Sample(
                    id=f"{subject}-{record['id']}",
                    input=prefix + _format_question(record),
                    target=str(record["answer"]),
                    metadata={"subject": subject, "category": category},
                )
            )

    return MemoryDataset(samples=samples, name="ceval")
