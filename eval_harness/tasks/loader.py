import json
from pathlib import Path
from eval_harness.core.models import TaskDefinition, TestCase


def load_task_cases(task_def: TaskDefinition) -> list[TestCase]:
    """Loads and returns TestCase models from a task dataset file."""
    path = Path(task_def.dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases = []
    for raw in raw_cases:
        # Format input prompt using task template
        formatted_input = task_def.prompt_template.format(input=raw["input"])
        cases.append(
            TestCase(
                case_id=raw["case_id"],
                input=formatted_input,
                reference_answer=raw.get("reference_answer"),
                category=raw["category"],
                expected_behavior=raw["expected_behavior"],
            )
        )
    return cases