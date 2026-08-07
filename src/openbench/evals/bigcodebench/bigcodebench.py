"""BigCodeBench code-generation evaluation."""

from pathlib import Path
from platform import machine as platform_machine
from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from openbench.datasets.bigcodebench import get_bigcodebench_dataset
from openbench.scorers.bigcodebench import bigcodebench_scorer

TASK_DIR = Path(__file__).parent
OFFICIAL_COMPOSE_PATH = (TASK_DIR / "compose.yaml").resolve()
ARM64_COMPOSE_PATH = (TASK_DIR / "compose.arm64.yaml").resolve()
Runtime = Literal["auto", "official", "arm64"]


def compose_path_for_runtime(runtime: Runtime) -> Path:
    """Select the BigCodeBench sandbox compose file for the local runtime."""

    if runtime == "official":
        return OFFICIAL_COMPOSE_PATH
    if runtime == "arm64":
        return ARM64_COMPOSE_PATH
    if runtime == "auto":
        host_machine = platform_machine().lower()
        if host_machine in {"arm64", "aarch64"}:
            return ARM64_COMPOSE_PATH
        return OFFICIAL_COMPOSE_PATH
    raise ValueError("runtime must be one of: auto, official, arm64")


@task
def bigcodebench(
    split: Literal["complete", "instruct"] = "instruct",
    subset: Literal["full", "hard"] = "full",
    runtime: Runtime = "auto",
    epochs: int = 1,
    limit: int | None = None,
    total_timeout: int = 900,
) -> Task:
    """Run BigCodeBench v0.1.4 with the official Docker evaluator.

    The official protocol supports `complete` and `instruct` splits and `full`
    and `hard` subsets. `runtime="official"` uses the pinned upstream amd64
    image; `runtime="arm64"` builds OpenBench's source-equivalent arm64 scorer
    image for Apple Silicon machines. `runtime="auto"` selects arm64 only on
    arm64/aarch64 hosts. Greedy decoding uses temperature 0 with one sample by
    default; increasing `epochs` enables Inspect pass@k reducers.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    reducers = ["mean", "pass_at_1"]
    if epochs >= 5:
        reducers.append("pass_at_5")
    if epochs >= 10:
        reducers.append("pass_at_10")

    return Task(
        name=f"bigcodebench_{split}_{subset}",
        dataset=get_bigcodebench_dataset(split=split, subset=subset, limit=limit),
        solver=generate(),
        scorer=bigcodebench_scorer(total_timeout=total_timeout),
        sandbox=("docker", str(compose_path_for_runtime(runtime))),
        epochs=Epochs(epochs, reducer=reducers),
        config=GenerateConfig(
            temperature=0,
            top_p=0.95,
            max_tokens=1280,
        ),
    )
