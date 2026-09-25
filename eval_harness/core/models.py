from typing import Any, Optional
from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    score: float
    verdict: str


class TestCase(BaseModel):
    case_id: str
    input: str
    reference_answer: Optional[str] = None
    category: str  # "correctness", "guardrail_harmful", "guardrail_benign"
    expected_behavior: str
    human_label: Optional[str] = None
    is_contested: bool = False


class TaskDefinition(BaseModel):
    task_id: str
    dataset_path: str
    prompt_template: str = "{input}"
    system_prompt: Optional[str] = None


class RunManifest(BaseModel):
    run_id: str
    task_id: str
    model_name: str
    temperature: float = 0.0
    seed: int = 42
    runs_per_case: int = 1


class TestCaseResult(BaseModel):
    case_id: str
    responses: list[str]
    scores: dict[str, ScoreResult] = Field(default_factory=dict)