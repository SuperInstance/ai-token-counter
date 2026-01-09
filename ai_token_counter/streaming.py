"""
Streaming token counting for real-time monitoring.
"""

import re
from typing import Dict, List, Optional, Callable, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum

from .counter import TokenCounter, ModelProvider, TokenCount


class StreamChunkType(str, Enum):
    """Types of chunks in a stream."""
    TOKEN = "token"
    WORD = "word"
    CHARACTER = "character"
    DELTA = "delta"


@dataclass
class StreamTokenCount:
    """Token count for a streaming response."""
    total_tokens: int
    delta_tokens: int
    text_length: int
    complete: bool = False
    model: str = "generic"


@dataclass
class StreamMetrics:
    """Metrics for a streaming session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    chunks_processed: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "chunks_processed": self.chunks_processed,
            "duration_seconds": self.end_time - self.start_time if self.end_time > self.start_time else 0,
            "tokens_per_second": self.tokens_per_second,
        }


class StreamingTokenCounter:
    """
    Count tokens in streaming LLM responses in real-time.

    Example:
        counter = StreamingTokenCounter()

        # Count tokens as chunks arrive
        for chunk in response_stream:
            count_info = counter.update(chunk)
            print(f"Tokens so far: {count_info.total_tokens}")

        # Get final count
        final = counter.finalize()
        print(f"Total tokens: {final.total_output_tokens}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        input_tokens: Optional[int] = None,
        chunk_type: StreamChunkType = StreamChunkType.DELTA
    ):
        """
        Initialize the streaming counter.

        Args:
            model: Model name for accurate counting
            input_tokens: Number of input tokens (for cost calculation)
            chunk_type: How to interpret incoming chunks
        """
        self.counter = TokenCounter(model)
        self.model = model or "generic"
        self.provider = self.counter._get_provider_for_model(self.model)
        self.chunk_type = chunk_type

        # Accumulated text from all chunks
        self._accumulated_text = ""

        # Track tokens seen so far
        self._previous_token_count = 0

        # Metrics
        self.metrics = StreamMetrics(
            total_input_tokens=input_tokens or 0
        )

    def update(self, chunk: str) -> StreamTokenCount:
        """
        Update token count with a new chunk.

        Args:
            chunk: Text chunk from the stream

        Returns:
            StreamTokenCount with current counts
        """
        # Accumulate text
        self._accumulated_text += chunk
        self.metrics.chunks_processed += 1

        # Count tokens in accumulated text
        current_count = self.counter._count_string(self._accumulated_text, self.provider)
        delta = current_count - self._previous_token_count

        # Handle delta chunks (SSE-style updates)
        if self.chunk_type == StreamChunkType.DELTA:
            # For delta chunks, count the chunk itself
            delta_count = self.counter._count_string(chunk, self.provider)
            current_count = self._previous_token_count + delta_count
            delta = delta_count

        self._previous_token_count = current_count

        return StreamTokenCount(
            total_tokens=current_count,
            delta_tokens=delta,
            text_length=len(self._accumulated_text),
            model=self.model
        )

    def update_sse(self, sse_data: str) -> StreamTokenCount:
        """
        Update from Server-Sent Events format.

        Args:
            sse_data: Raw SSE data string

        Returns:
            StreamTokenCount with current counts
        """
        # Extract content from SSE format
        # Format: "data: {..."content": "..."}"
        content_match = re.search(r'"content"\s*:\s*"([^"]*)"', sse_data)

        if content_match:
            content = content_match.group(1)
            # Unescape JSON
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            return self.update(content)

        # No content found, return count without increment
        return StreamTokenCount(
            total_tokens=self._previous_token_count,
            delta_tokens=0,
            text_length=len(self._accumulated_text),
            model=self.model
        )

    def finalize(self) -> StreamMetrics:
        """
        Finalize the counting session.

        Returns:
            StreamMetrics with final statistics
        """
        import time

        self.metrics.total_output_tokens = self._previous_token_count
        self.metrics.end_time = time.time()

        if self.metrics.start_time > 0:
            duration = self.metrics.end_time - self.metrics.start_time
            if duration > 0:
                self.metrics.tokens_per_second = self.metrics.total_output_tokens / duration

        return self.metrics

    def reset(self) -> None:
        """Reset the counter for a new stream."""
        self._accumulated_text = ""
        self._previous_token_count = 0
        self.metrics = StreamMetrics(
            total_input_tokens=self.metrics.total_input_tokens
        )

    def get_current_count(self) -> int:
        """Get current token count."""
        return self._previous_token_count

    def get_accumulated_text(self) -> str:
        """Get all accumulated text so far."""
        return self._accumulated_text


class TokenBudget:
    """
    Monitor and enforce token budgets during streaming.

    Example:
        budget = TokenBudget(max_output_tokens=1000)

        for chunk in stream:
            if budget.is_exceeded():
                break
            budget.update(chunk)
    """

    def __init__(
        self,
        max_output_tokens: int,
        max_input_tokens: Optional[int] = None,
        warning_threshold: float = 0.9
    ):
        """
        Initialize the token budget.

        Args:
            max_output_tokens: Maximum output tokens allowed
            max_input_tokens: Maximum input tokens (for validation)
            warning_threshold: Fraction of budget to trigger warning (0.0-1.0)
        """
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self.warning_threshold = warning_threshold

        self._output_tokens = 0
        self._input_tokens = 0
        self._warnings_triggered = []

    def update(self, tokens: int, is_output: bool = True) -> bool:
        """
        Update the budget tracker.

        Args:
            tokens: Number of tokens to add
            is_output: True if output tokens, False if input

        Returns:
            True if still within budget, False if exceeded
        """
        if is_output:
            self._output_tokens += tokens
        else:
            self._input_tokens += tokens

        # Check warnings
        output_ratio = self._output_tokens / self.max_output_tokens
        if output_ratio >= self.warning_threshold and "output_warning" not in self._warnings_triggered:
            self._warnings_triggered.append("output_warning")

        return not self.is_exceeded()

    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        if self._output_tokens >= self.max_output_tokens:
            return True

        if self.max_input_tokens and self._input_tokens >= self.max_input_tokens:
            return True

        return False

    def remaining_output_tokens(self) -> int:
        """Get remaining output tokens."""
        return max(0, self.max_output_tokens - self._output_tokens)

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "max_output_tokens": self.max_output_tokens,
            "remaining_output_tokens": self.remaining_output_tokens(),
            "output_usage_ratio": self._output_tokens / self.max_output_tokens,
            "warnings_triggered": self._warnings_triggered,
            "exceeded": self.is_exceeded()
        }


def count_streaming_tokens(
    stream: Iterator[str],
    model: str = "gpt-4o",
    callback: Optional[Callable[[StreamTokenCount], None]] = None
) -> StreamMetrics:
    """
    Count tokens from a streaming response iterator.

    Args:
        stream: Iterator yielding text chunks
        model: Model name for counting
        callback: Optional function called with each count update

    Returns:
        StreamMetrics with final statistics
    """
    import time

    counter = StreamingTokenCounter(model)
    counter.metrics.start_time = time.time()

    for chunk in stream:
        count_info = counter.update(chunk)
        if callback:
            callback(count_info)

    return counter.finalize()


def count_batch_tokens(
    texts: List[str],
    model: str = "gpt-4o",
    show_progress: bool = False
) -> List[TokenCount]:
    """
    Count tokens for multiple texts efficiently.

    Args:
        texts: List of texts to count
        model: Model name for counting
        show_progress: Whether to show progress (for large batches)

    Returns:
        List of TokenCount results
    """
    counter = TokenCounter(model)
    results = []

    for i, text in enumerate(texts):
        result = counter.count(text, model)
        results.append(result)

        if show_progress and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(texts)} texts")

    return results
