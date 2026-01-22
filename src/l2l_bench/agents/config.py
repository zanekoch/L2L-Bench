"""LLM configuration for LDP agent.

API keys are loaded from .env file in project root. LDP's LLMCallOp uses
LiteLLM internally, which reads API keys from environment variables:
    - OPENAI_API_KEY for openai/* models
    - ANTHROPIC_API_KEY for anthropic/* models
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# load .env from project root on module import
_project_root = Path(__file__).parent.parent.parent.parent  # src/l2l_bench/agents/ -> project root
load_dotenv(_project_root / ".env")


@dataclass
class LLMConfig:
    """
    Configuration for LLM calls via LDP's LLMCallOp.

    Model names use litellm format: "provider/model"
    Examples:
        - "anthropic/claude-3-5-sonnet-20241022"
        - "openai/gpt-4o"
        - "openai/gpt-4o-mini"

    API keys should be set in .env file in project root:
        OPENAI_API_KEY=sk-...
        ANTHROPIC_API_KEY=sk-ant-...
    """

    model: str = "anthropic/claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 4096

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for LLMCallOp."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# presets for common models
CLAUDE_SONNET = LLMConfig(model="anthropic/claude-3-5-sonnet-20241022")
CLAUDE_HAIKU = LLMConfig(model="anthropic/claude-3-5-haiku-20241022")
GPT4O = LLMConfig(model="openai/gpt-4o")
GPT4O_MINI = LLMConfig(model="openai/gpt-4o-mini", temperature=0.0)
