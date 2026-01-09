# AI Token Counter

Accurate token counting for multiple LLM providers.

## Features

- **Multi-Provider Support**: OpenAI (GPT-4, GPT-3.5, GPT-4o, o1), Anthropic (Claude 3, 3.5, 3.7), Google (Gemini), Meta (Llama 2, 3, 3.1, 3.2, 3.3), Mistral, DeepSeek, Cohere
- **Cost Estimation**: Built-in pricing for accurate cost calculations
- **Message Support**: Count tokens in chat messages with proper overhead
- **Simple API**: Easy-to-use functions for quick counting

## Installation

```bash
pip install ai-token-counter
```

## Quick Start

```python
from ai_token_counter import count_tokens, estimate_cost

# Count tokens in text
count = count_tokens("Hello, world!", model="gpt-4o")
print(f"Tokens: {count}")

# Estimate cost
cost = estimate_cost(input_tokens=1000, output_tokens=500, model="gpt-4o")
print(f"Cost: ${cost.total_cost_usd:.6f}")

# Count chat messages
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]
from ai_token_counter import count_message_tokens
result = count_message_tokens(messages, model="claude-3-5-sonnet")
print(f"Message tokens: {result.count}")
```

## Supported Models

### OpenAI
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
- o1-preview, o1-mini

### Anthropic
- claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku
- claude-3-opus, claude-3-sonnet, claude-3-haiku

### Google
- gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash
- gemini-1.0-pro

### Meta
- llama-3.3-70b, llama-3.1-405b, llama-3.1-70b
- llama-3-70b, llama-3-8b, llama-2-70b

### Mistral
- mistral-large, mistral-medium, mistral-small
- mixtral-8x7b, mixtral-8x22b, codestral

### DeepSeek
- deepseek-chat, deepseek-coder, deepseek-reasoner

### Cohere
- command-r-plus, command-r, command

## API Reference

### TokenCounter Class

```python
from ai_token_counter import TokenCounter

counter = TokenCounter(model="gpt-4o")

# Count text
result = counter.count("Your text here")
print(result.count)  # Token count
print(result.provider)  # Provider
print(result.estimated)  # Whether estimated

# Count messages
messages = [{"role": "user", "content": "Hello"}]
result = counter.count(messages=messages)

# Estimate cost
cost = counter.estimate_cost(input_tokens=1000, output_tokens=500)
print(cost.total_cost_usd)

# Get pricing info
pricing = counter.get_pricing_info("gpt-4o")
```

### Convenience Functions

```python
from ai_token_counter import (
    count_tokens,
    count_string_tokens,
    count_message_tokens,
    estimate_cost,
    count_conversation_tokens
)

# Quick count
count = count_tokens("text", model="gpt-4o")

# Full result
result = count_string_tokens("text", model="claude-3-opus")

# Messages
result = count_message_tokens(messages, model="gemini-1.5-pro")

# Cost
cost = estimate_cost(1000, 500, model="llama-3-70b")

# Conversation breakdown
breakdown = count_conversation_tokens(messages)
for msg in breakdown["messages"]:
    print(f"Message {msg['index']} ({msg['role']}): {msg['tokens']} tokens")
print(f"Total: {breakdown['total_tokens']} tokens")
```

## License

MIT License - see LICENSE file for details.
