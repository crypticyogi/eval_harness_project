import asyncio
import json
from typing import Optional
import aiosqlite

from eval_harness.backends.base import LLMBackend
from eval_harness.core.models import RunManifest, TestCase, TestCaseResult
from eval_harness.scorers import CaseScorer, GroupScorer


class EvalRunner:
    """Async orchestration engine for executing test cases and evaluating outputs."""

    def __init__(
        self,
        backend: LLMBackend,
        case_scorers: list[CaseScorer],
        group_scorers: Optional[list[GroupScorer]] = None,
        max_concurrency: int = 5,
        cache_db_path: str = ".eval_cache.db",
    ):
        self.backend = backend
        self.case_scorers = case_scorers
        self.group_scorers = group_scorers or []
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cache_db_path = cache_db_path

    async def _init_cache(self) -> None:
        async with aiosqlite.connect(self.cache_db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL
                )
                """
            )
            await db.commit()

    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        async with aiosqlite.connect(self.cache_db_path) as db:
            async with db.execute(
                "SELECT response FROM response_cache WHERE cache_key = ?", (cache_key,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

    async def _save_cached_response(self, cache_key: str, response: str) -> None:
        async with aiosqlite.connect(self.cache_db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, response) VALUES (?, ?)",
                (cache_key, response),
            )
            await db.commit()

    async def _generate_single_run(
        self, case: TestCase, manifest: RunManifest, repeat_idx: int
    ) -> str:
        cache_key = f"{manifest.model_name}:{manifest.temperature}:{case.case_id}:{repeat_idx}"
        
        cached = await self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        async with self.semaphore:
            response = await self.backend.generate(
                prompt=case.input,
                temperature=manifest.temperature,
                repeat_index=repeat_idx,
            )

        await self._save_cached_response(cache_key, response)
        return response

    async def run_case(self, case: TestCase, manifest: RunManifest) -> TestCaseResult:
        # 1. Execute runs in parallel
        tasks = [
            self._generate_single_run(case, manifest, idx)
            for idx in range(manifest.runs_per_case)
        ]
        responses = await asyncio.gather(*tasks)

        result = TestCaseResult(case_id=case.case_id, responses=responses)

        # 2. Run Case Scorers (evaluated against primary run [0])
        primary_response = responses[0] if responses else ""
        for scorer in self.case_scorers:
            result.scores[scorer.name] = await scorer.score_case(case, primary_response)

        # 3. Run Group Scorers (evaluated across all N runs)
        for g_scorer in self.group_scorers:
            result.scores[g_scorer.name] = await g_scorer.score_group(case, responses)

        return result

    async def run_suite(
        self, cases: list[TestCase], manifest: RunManifest
    ) -> list[TestCaseResult]:
        await self._init_cache()
        tasks = [self.run_case(case, manifest) for case in cases]
        return await asyncio.gather(*tasks)