from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMBackend(ABC):
    """Abstract base interface for all model backends."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        repeat_index: int = 0,
        **kwargs: Any,
    ) -> str:
        """Generates a text completion for a given prompt."""
        pass