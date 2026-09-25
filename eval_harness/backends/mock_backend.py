from typing import Any, Optional
from eval_harness.backends.base import LLMBackend


class MockBackend(LLMBackend):
    """Zero-cost mock backend for local pipeline testing."""

    def __init__(self, default_response: str = "This is a mock response from the LLM."):
        self.default_response = default_response

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        repeat_index: int = 0,
        **kwargs: Any,
    ) -> str:
        lowered = prompt.lower()
        
        # Simple simulated behavior for safety judges / task responses
        if "classify the outcome" in lowered or "safety audit judge" in lowered:
            if "synthesize tnt" in lowered or "hack" in lowered:
                return "VERDICT: REFUSED"
            return "VERDICT: FULFILLED"

        if "derivative of x^2" in lowered:
            return "The derivative is 2x + 3."

        return self.default_response