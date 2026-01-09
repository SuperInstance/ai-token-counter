"""
AI Token Counter - Accurate token counting for multiple LLM providers.

Supports:
- OpenAI (GPT-4, GPT-3.5, GPT-4o, etc.)
- Anthropic (Claude 3, Claude 3.5, Claude 3.7)
- Google (Gemini Pro, Gemini Flash)
- Meta (Llama 2, Llama 3, Llama 3.1, Llama 3.2, Llama 3.3)
- Mistral (Mistral, Mixtral)
- DeepSeek
- And more...
"""

from .counter import (
    TokenCounter,
    count_tokens,
    count_message_tokens,
    count_string_tokens,
    estimate_cost,
    get_tokenizer_for_model,
    count_conversation_tokens,
)
from .streaming import (
    StreamingTokenCounter,
    StreamTokenCount,
    StreamMetrics,
    StreamChunkType,
    TokenBudget,
    count_streaming_tokens,
    count_batch_tokens,
)

__version__ = "1.1.0"
__all__ = [
    "TokenCounter",
    "count_tokens",
    "count_message_tokens",
    "count_string_tokens",
    "estimate_cost",
    "get_tokenizer_for_model",
    "count_conversation_tokens",
    "StreamingTokenCounter",
    "StreamTokenCount",
    "StreamMetrics",
    "StreamChunkType",
    "TokenBudget",
    "count_streaming_tokens",
    "count_batch_tokens",
]
