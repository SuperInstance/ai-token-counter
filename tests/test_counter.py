"""
Tests for Token Counter
"""

import pytest
from ai_token_counter import (
    TokenCounter,
    ModelProvider,
    TokenCount,
    CostEstimate,
    count_tokens,
    count_string_tokens,
    count_message_tokens,
    estimate_cost,
    get_tokenizer_for_model,
    count_conversation_tokens
)


class TestTokenCounter:
    """Test TokenCounter class"""

    def test_initialization(self):
        """Test counter initialization"""
        counter = TokenCounter()
        assert counter.default_model is None
        assert counter.default_provider == ModelProvider.GENERIC

    def test_initialization_with_model(self):
        """Test counter initialization with model"""
        counter = TokenCounter(model="gpt-4o")
        assert counter.default_model == "gpt-4o"
        assert counter.default_provider == ModelProvider.OPENAI

    def test_count_simple_text(self):
        """Test counting simple text"""
        counter = TokenCounter(model="gpt-4o")
        result = counter.count("Hello, world!")
        assert result.count > 0
        assert result.model == "gpt-4o"
        assert result.provider == ModelProvider.OPENAI

    def test_count_empty_text(self):
        """Test counting empty text"""
        counter = TokenCounter()
        result = counter.count("")
        assert result.count == 0

    def test_count_long_text(self):
        """Test counting longer text"""
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = counter.count(text)
        assert result.count > 10

    def test_count_with_code(self):
        """Test counting code (should have different ratio)"""
        counter = TokenCounter()
        code = "def hello():\n    print('Hello, world!')\n    return True"
        result = counter.count(code)
        assert result.count > 0

    def test_count_messages_openai(self):
        """Test counting OpenAI-format messages"""
        counter = TokenCounter(model="gpt-4o")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = counter.count(messages=messages)
        assert result.count > 0
        assert result.provider == ModelProvider.OPENAI

    def test_count_messages_anthropic(self):
        """Test counting Anthropic-format messages"""
        counter = TokenCounter(model="claude-3-5-sonnet")
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        result = counter.count(messages=messages)
        assert result.count > 0
        assert result.provider == ModelProvider.ANTHROPIC

    def test_estimate_cost_gpt4o(self):
        """Test cost estimation for GPT-4o"""
        counter = TokenCounter(model="gpt-4o")
        cost = counter.estimate_cost(1000, 500)
        assert cost.input_cost_usd > 0
        assert cost.output_cost_usd > 0
        assert cost.total_cost_usd == cost.input_cost_usd + cost.output_cost_usd

    def test_estimate_cost_claude(self):
        """Test cost estimation for Claude"""
        counter = TokenCounter(model="claude-3-opus")
        cost = counter.estimate_cost(1000, 500)
        assert cost.input_cost_usd > 0
        assert cost.output_cost_usd > 0

    def test_get_pricing_info(self):
        """Test getting pricing information"""
        counter = TokenCounter()
        pricing = counter.get_pricing_info("gpt-4o")
        assert pricing["model"] == "gpt-4o"
        assert pricing["provider"] == "openai"
        assert "input_price_per_1m" in pricing
        assert "output_price_per_1m" in pricing

    def test_unknown_model_defaults(self):
        """Test that unknown models use generic defaults"""
        counter = TokenCounter()
        result = counter.count("test text", model="unknown-model-xyz")
        assert result.count > 0
        assert result.provider == ModelProvider.GENERIC


class TestModelProviderDetection:
    """Test model provider detection"""

    def test_openai_models(self):
        """Test OpenAI model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("gpt-4o") == ModelProvider.OPENAI
        assert counter._get_provider_for_model("gpt-4-turbo") == ModelProvider.OPENAI
        assert counter._get_provider_for_model("gpt-3.5-turbo") == ModelProvider.OPENAI
        assert counter._get_provider_for_model("o1-preview") == ModelProvider.OPENAI

    def test_anthropic_models(self):
        """Test Anthropic model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("claude-3-opus") == ModelProvider.ANTHROPIC
        assert counter._get_provider_for_model("claude-3.5-sonnet") == ModelProvider.ANTHROPIC
        assert counter._get_provider_for_model("claude-3-haiku") == ModelProvider.ANTHROPIC

    def test_google_models(self):
        """Test Google model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("gemini-1.5-pro") == ModelProvider.GOOGLE
        assert counter._get_provider_for_model("gemini-2.0-flash") == ModelProvider.GOOGLE

    def test_meta_models(self):
        """Test Meta model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("llama-3-70b") == ModelProvider.META
        assert counter._get_provider_for_model("llama-3.1-405b") == ModelProvider.META

    def test_mistral_models(self):
        """Test Mistral model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("mixtral-8x7b") == ModelProvider.MISTRAL
        assert counter._get_provider_for_model("mistral-large") == ModelProvider.MISTRAL

    def test_deepseek_models(self):
        """Test DeepSeek model detection"""
        counter = TokenCounter()
        assert counter._get_provider_for_model("deepseek-chat") == ModelProvider.DEEPSEEK
        assert counter._get_provider_for_model("deepseek-coder") == ModelProvider.DEEPSEEK


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_count_tokens(self):
        """Test count_tokens function"""
        count = count_tokens("Hello, world!", model="gpt-4o")
        assert isinstance(count, int)
        assert count > 0

    def test_count_string_tokens(self):
        """Test count_string_tokens function"""
        result = count_string_tokens("Hello, world!", model="claude-3-opus")
        assert isinstance(result, TokenCount)
        assert result.count > 0

    def test_count_message_tokens_func(self):
        """Test count_message_tokens function"""
        messages = [{"role": "user", "content": "Hello!"}]
        result = count_message_tokens(messages, model="gpt-4o")
        assert isinstance(result, TokenCount)
        assert result.count > 0

    def test_estimate_cost_func(self):
        """Test estimate_cost function"""
        cost = estimate_cost(1000, 500, model="gpt-4o")
        assert isinstance(cost, CostEstimate)
        assert cost.total_cost_usd > 0

    def test_get_tokenizer_for_model(self):
        """Test get_tokenizer_for_model function"""
        counter = get_tokenizer_for_model("gpt-4o")
        assert isinstance(counter, TokenCounter)
        assert counter.default_model == "gpt-4o"

    def test_count_conversation_tokens(self):
        """Test count_conversation_tokens function"""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = count_conversation_tokens(messages, model="gpt-4o")
        assert "messages" in result
        assert "total_tokens" in result
        assert "model" in result
        assert len(result["messages"]) == 3
        assert result["total_tokens"] > 0


class TestCostCalculations:
    """Test cost calculation accuracy"""

    def test_gpt4o_pricing(self):
        """Test GPT-4o pricing"""
        counter = TokenCounter(model="gpt-4o")
        # GPT-4o: $2.50/1M input, $10.00/1M output
        cost = counter.estimate_cost(1_000_000, 1_000_000)
        assert cost.input_cost_usd == 2.50
        assert cost.output_cost_usd == 10.00
        assert cost.total_cost_usd == 12.50

    def test_claude_opus_pricing(self):
        """Test Claude Opus pricing"""
        counter = TokenCounter(model="claude-3-opus")
        # Claude Opus: $15/1M input, $75/1M output
        cost = counter.estimate_cost(1_000_000, 1_000_000)
        assert cost.input_cost_usd == 15.00
        assert cost.output_cost_usd == 75.00
        assert cost.total_cost_usd == 90.00

    def test_gemini_pricing(self):
        """Test Gemini pricing"""
        counter = TokenCounter(model="gemini-1.5-flash")
        # Gemini Flash: $0.075/1M input, $0.30/1M output
        cost = counter.estimate_cost(1_000_000, 1_000_000)
        assert cost.input_cost_usd == 0.075
        assert cost.output_cost_usd == 0.30

    def test_zero_tokens(self):
        """Test cost with zero tokens"""
        counter = TokenCounter(model="gpt-4o")
        cost = counter.estimate_cost(0, 0)
        assert cost.total_cost_usd == 0


class TestEdgeCases:
    """Test edge cases and special inputs"""

    def test_whitespace_only(self):
        """Test counting whitespace-only text"""
        counter = TokenCounter()
        result = counter.count("   \n\n\t   ")
        assert result.count == 0

    def test_unicode_text(self):
        """Test counting unicode text"""
        counter = TokenCounter()
        result = counter.count("Hello 世界 🌍")
        assert result.count > 0

    def test_very_long_text(self):
        """Test counting very long text"""
        counter = TokenCounter()
        text = "word " * 10000
        result = counter.count(text)
        assert result.count > 0

    def test_messages_with_name(self):
        """Test counting messages with name field"""
        counter = TokenCounter(model="gpt-4o")
        messages = [
            {"role": "user", "name": "Alice", "content": "Hello!"}
        ]
        result = counter.count(messages=messages)
        # Should count extra tokens for name field
        assert result.count > 3

    def test_empty_messages(self):
        """Test counting empty message list"""
        counter = TokenCounter()
        result = counter.count(messages=[])
        # Should still count message array overhead
        assert result.count >= 0


class TestTokenCount:
    """Test TokenCount dataclass"""

    def test_token_count_creation(self):
        """Test creating TokenCount"""
        count = TokenCount(
            count=10,
            method="estimation",
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            estimated=True
        )
        assert count.count == 10
        assert count.method == "estimation"
        assert count.model == "gpt-4o"
        assert count.provider == ModelProvider.OPENAI
        assert count.estimated is True


class TestCostEstimate:
    """Test CostEstimate dataclass"""

    def test_cost_estimate_creation(self):
        """Test creating CostEstimate"""
        cost = CostEstimate(
            input_cost_usd=0.01,
            output_cost_usd=0.02,
            total_cost_usd=0.03
        )
        assert cost.input_cost_usd == 0.01
        assert cost.output_cost_usd == 0.02
        assert cost.total_cost_usd == 0.03
        assert cost.currency == "USD"
