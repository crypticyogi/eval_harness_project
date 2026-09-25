import asyncio
import json
from pathlib import Path
from eval_harness.backends.mock_backend import MockBackend
from eval_harness.core.models import RunManifest, TaskDefinition
from eval_harness.runner import EvalRunner
from eval_harness.scorers import (
    ExactMatchScorer,
    ExactMatchAgreementScorer,
    RegexRefusalScorer,
)
from eval_harness.tasks.loader import load_task_cases


async def main():
    print("=== Running eval_harness Benchmark Suite ===")

    # 1. Define Task & Manifest
    task_def = TaskDefinition(
        task_id="safety_and_math_v1",
        dataset_path="eval_harness/tasks/sample_safety_math.json",
        prompt_template="User prompt: {input}",
    )

    manifest = RunManifest(
        run_id="run_001",
        task_id=task_def.task_id,
        model_name="mock_llm",
        temperature=0.0,
        runs_per_case=2,
    )

    # 2. Load Cases
    cases = load_task_cases(task_def)
    print(f"Loaded {len(cases)} test cases from {task_def.dataset_path}")

    # 3. Setup Runner & Scorers
    runner = EvalRunner(
        backend=MockBackend(),
        case_scorers=[RegexRefusalScorer(), ExactMatchScorer()],
        group_scorers=[ExactMatchAgreementScorer()],
        max_concurrency=3,
        cache_db_path=".eval_cache.db",
    )

    # 4. Execute Suite
    results = await runner.run_suite(cases, manifest)

    # 5. Export Run Artifact
    output_dir = Path("eval_runs")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{manifest.run_id}_report.json"

    export_data = {
        "manifest": manifest.model_dump(),
        "task": task_def.model_dump(),
        "results": [r.model_dump() for r in results],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n✅ Evaluation complete! Run report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())