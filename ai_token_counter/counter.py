"""
Token counting for various LLM providers.
"""

import re
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum


class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    REPOLYTE = "repolyte"
    QWEN = "qwen"
    GENERIC = "generic"


@dataclass
class TokenCount:
    """Result of token counting"""
    count: int
    method: str
    model: str
    provider: ModelProvider
    estimated: bool = False


@dataclass
class CostEstimate:
    """Cost estimation for token usage"""
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    currency: str = "USD"


class TokenCounter:
    """
    Accurate token counting for multiple LLM providers.

    Uses provider-specific tokenizers when available,
    falls back to accurate estimation methods.

    Example:
        counter = TokenCounter()
        count = counter.count("Hello, world!", model="gpt-4")
        print(f"Tokens: {count.count}")
    """

    # Character-to-token ratios based on empirical analysis
    CHAR_TOKEN_RATIOS = {
        ModelProvider.OPENAI: 4.0,  # ~4 chars per token for English
        ModelProvider.ANTHROPIC: 3.7,
        ModelProvider.GOOGLE: 4.0,
        ModelProvider.META: 4.0,
        ModelProvider.MISTRAL: 4.0,
        ModelProvider.DEEPSEEK: 3.8,
        ModelProvider.GENERIC: 4.0,
    }

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        # OpenAI
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1-preview": (15.00, 60.00),
        "o1-mini": (1.10, 4.40),
        # Anthropic
        "claude-3-7-sonnet": (3.00, 15.00),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "claude-3-opus": (15.00, 75.00),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        # Google
        "gemini-2.0-flash-exp": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.0-pro": (0.50, 1.50),
        # Meta
        "llama-3.3-70b": (0.59, 0.79),  # Via together.ai
        "llama-3.1-405b": (2.70, 2.70),
        "llama-3.1-70b": (0.59, 0.79),
        "llama-3-70b": (0.70, 0.70),
        "llama-3-8b": (0.10, 0.10),
        "llama-2-70b": (0.70, 0.70),
        # Mistral
        "mistral-large": (2.00, 6.00),
        "mistral-medium": (0.25, 0.25),
        "mistral-small": (0.10, 0.10),
        "mixtral-8x7b": (0.50, 0.50),
        "mixtral-8x22b": (0.65, 0.65),
        "codestral": (0.20, 0.20),
        # DeepSeek
        "deepseek-chat": (0.14, 0.28),
        "deepseek-coder": (0.14, 0.28),
        "deepseek-reasoner": (0.55, 2.19),
        # Cohere
        "command-r-plus": (3.00, 15.00),
        "command-r": (0.50, 1.50),
        "command": (0.50, 1.50),
    }

    MODEL_PATTERNS = {
        r"gpt-4o|gpt-4": (ModelProvider.OPENAI, "gpt-4o"),
        r"gpt-3\.5|gpt-35": (ModelProvider.OPENAI, "gpt-3.5-turbo"),
        r"o1-": (ModelProvider.OPENAI, "o1-preview"),
        r"claude-3[-\.]7|claude-37": (ModelProvider.ANTHROPIC, "claude-3-7-sonnet"),
        r"claude-3[-\.]5|claude-35": (ModelProvider.ANTHROPIC, "claude-3-5-sonnet"),
        r"claude-3[-\.]?(haiku|opus|sonnet)": (ModelProvider.ANTHROPIC, "claude-3-sonnet"),
        r"gemini-2|gemini-1\.5-pro": (ModelProvider.GOOGLE, "gemini-1.5-pro"),
        r"gemini-1": (ModelProvider.GOOGLE, "gemini-1.0-pro"),
        r"llama-3\.3|llama-33": (ModelProvider.META, "llama-3.3-70b"),
        r"llama-3\.1|llama-31": (ModelProvider.META, "llama-3.1-70b"),
        r"llama-3": (ModelProvider.META, "llama-3-70b"),
        r"llama-2": (ModelProvider.META, "llama-2-70b"),
        r"mixtral-8x22b": (ModelProvider.MISTRAL, "mixtral-8x22b"),
        r"mixtral-8x7b|mixtral-8x7": (ModelProvider.MISTRAL, "mixtral-8x7b"),
        r"mistral-large": (ModelProvider.MISTRAL, "mistral-large"),
        r"mistral-small": (ModelProvider.MISTRAL, "mistral-small"),
        r"mistral-medium": (ModelProvider.MISTRAL, "mistral-medium"),
        r"codestral": (ModelProvider.MISTRAL, "codestral"),
        r"deepseek-(reasoner|r1)": (ModelProvider.DEEPSEEK, "deepseek-reasoner"),
        r"deepseek": (ModelProvider.DEEPSEEK, "deepseek-chat"),
        r"command-r-plus": (ModelProvider.COHERE, "command-r-plus"),
        r"command-r": (ModelProvider.COHERE, "command-r"),
        r"command": (ModelProvider.COHERE, "command"),
    }

    # Word boundaries for more accurate counting
    WORD_PATTERN = re.compile(r'\S+')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the token counter.

        Args:
            model: Default model to use for counting
        """
        self.default_model = model
        self.default_provider = self._get_provider_for_model(model) if model else ModelProvider.GENERIC

    def count(
        self,
        text: str,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> TokenCount:
        """
        Count tokens in text or messages.

        Args:
            text: Text to count tokens for
            model: Model name for accurate counting
            messages: List of message dicts (overrides text)

        Returns:
            TokenCount with count and metadata
        """
        target_model = model or self.default_model or "generic"
        provider = self._get_provider_for_model(target_model)

        if messages:
            count = self._count_messages(messages, provider)
        else:
            count = self._count_string(text, provider)

        return TokenCount(
            count=count,
            method="estimation",
            model=target_model,
            provider=provider,
            estimated=True
        )

    def _count_string(self, text: str, provider: ModelProvider) -> int:
        """Count tokens in a string using provider-specific estimation."""
        # Remove extra whitespace
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()

        # Count words (better proxy for tokens than characters)
        words = self.WORD_PATTERN.findall(text)
        word_count = len(words)

        # Use character-based estimation for non-word text (code, etc.)
        char_count = len(text)
        char_based = max(1, int(char_count / self.CHAR_TOKEN_RATIOS.get(provider, 4.0)))

        # Average the two for better accuracy
        return max(1, int((word_count + char_based) / 2))

    def _count_messages(self, messages: List[Dict[str, str]], provider: ModelProvider) -> int:
        """Count tokens in a list of messages."""
        total = 0

        for msg in messages:
            # Count role tokens (roughly 1-5 per message depending on provider)
            if provider == ModelProvider.ANTHROPIC:
                total += 5  # Anthropic uses more tokens for formatting
            else:
                total += 3

            # Count content
            content = msg.get("content", "")
            total += self._count_string(content, provider)

            # Count name if present
            if "name" in msg:
                total += 2

        # Add overhead for the message array itself
        total += 3

        return total

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> CostEstimate:
        """
        Estimate API cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing

        Returns:
            CostEstimate with cost breakdown
        """
        target_model = model or self.default_model or "gpt-4o"

        # Find matching pricing
        pricing = self._get_pricing(target_model)

        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]

        return CostEstimate(
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(input_cost + output_cost, 6)
        )

    def _get_provider_for_model(self, model: str) -> ModelProvider:
        """Determine provider from model name."""
        model_lower = model.lower()

        for pattern, (provider, _) in self.MODEL_PATTERNS.items():
            if re.search(pattern, model_lower):
                return provider

        return ModelProvider.GENERIC

    def _get_pricing(self, model: str) -> tuple:
        """Get (input, output) pricing per 1M tokens for a model."""
        model_lower = model.lower()

        # Direct match
        if model_lower in self.PRICING:
            return self.PRICING[model_lower]

        # Pattern match
        for pattern, (_, pricing_model) in self.MODEL_PATTERNS.items():
            if re.search(pattern, model_lower):
                return self.PRICING.get(pricing_model, (0.001, 0.002))

        # Default pricing
        return (0.001, 0.002)

    def get_pricing_info(self, model: str) -> Dict[str, Any]:
        """Get pricing information for a model."""
        input_price, output_price = self._get_pricing(model)
        return {
            "model": model,
            "provider": self._get_provider_for_model(model).value,
            "input_price_per_1m": input_price,
            "output_price_per_1m": output_price,
            "currency": "USD"
        }


# Convenience functions

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Quick token count for text."""
    counter = TokenCounter(model)
    return counter.count(text).count


def count_string_tokens(text: str, model: str = "gpt-4o") -> TokenCount:
    """Count tokens with metadata."""
    counter = TokenCounter(model)
    return counter.count(text)


def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> TokenCount:
    """Count tokens in chat messages."""
    counter = TokenCounter(model)
    return counter.count(messages=messages)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o"
) -> CostEstimate:
    """Estimate API cost."""
    counter = TokenCounter(model)
    return counter.estimate_cost(input_tokens, output_tokens)


def get_tokenizer_for_model(model: str) -> TokenCounter:
    """Get a counter configured for a specific model."""
    return TokenCounter(model)


def count_conversation_tokens(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o"
) -> Dict[str, int]:
    """
    Count tokens for a full conversation.

    Returns breakdown by message and totals.
    """
    counter = TokenCounter(model)

    message_counts = []
    total = 0

    for i, msg in enumerate(messages):
        content_tokens = counter._count_string(msg.get("content", ""), counter._get_provider_for_model(model))
        role_tokens = 3 if msg.get("name") else 5 if counter._get_provider_for_model(model) == ModelProvider.ANTHROPIC else 3
        msg_total = content_tokens + role_tokens

        message_counts.append({
            "index": i,
            "role": msg.get("role", "unknown"),
            "tokens": msg_total
        })
        total += msg_total

    return {
        "messages": message_counts,
        "total_tokens": total,
        "model": model
    }
