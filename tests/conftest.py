"""Shared pytest configurations and fixtures for the AssistArena testing environment.

Provides a pre-configured TestClient, stadium ID constants, and simulated Mock clients
to test GenAI live operations path with local offline isolation.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from backend import copilot
from backend.server import app, rate_limiter

STADIUM_ID = "new-york-new-jersey"


@pytest.fixture(autouse=True)
def _reset_copilot_client() -> Iterator[None]:  # type: ignore
    """Ensure the copilot client mock is reset around each test case."""
    copilot._reset_genai_client()
    yield
    copilot._reset_genai_client()


@pytest.fixture
def stadium_id() -> str:
    """Return a valid target stadium identifier."""
    return STADIUM_ID


@pytest.fixture
def test_client() -> Iterator[TestClient]:  # type: ignore
    """Provide a TestClient with a flushed rate limit bucket registry."""
    rate_limiter.flush()
    with TestClient(app) as tc:
        yield tc
    rate_limiter.flush()


# --- Mock Non-Streaming GenAI Interfaces ---

class _MockCall:
    def __init__(self, name: str, args: dict[str, Any]) -> None:  # type: ignore
        self.name = name
        self.args = args


class _MockCandidate:
    def __init__(self, content: types.Content | None = None) -> None:
        self.content = content


class MockResponse:
    """Simulated GenerateContentResponse from Google GenAI."""

    def __init__(
        self,
        *,
        function_calls: list[types.FunctionCall] | None = None,
        text: str | None = None,
        model_turn: types.Content | None = None,
    ) -> None:
        self.function_calls = function_calls or []
        self.text = text
        self.candidates = [_MockCandidate(model_turn)] if model_turn else []


class MockGenAIClient:
    """A client simulating sequential responses from GenAI model targets."""

    def __init__(self, responses: Sequence[MockResponse]) -> None:
        self._responses = list(responses)

        class _Models:
            def __init__(self, outer: "MockGenAIClient") -> None:
                self._outer = outer

            def generate_content(
                self, *, model: str, contents: Any, config: Any,  # type: ignore
            ) -> MockResponse:
                return self._outer._responses.pop(0)

        self.models = _Models(self)


@pytest.fixture
def make_function_call() -> Any:  # type: ignore
    """Build a mock call specification."""
    return _MockCall


@pytest.fixture
def install_mock_client(monkeypatch: pytest.MonkeyPatch) -> Any:  # type: ignore
    """Install a mocked non-streaming client instance."""
    def _install(responses: Sequence[MockResponse]) -> MockGenAIClient:
        c = MockGenAIClient(responses)
        monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: c)
        return c
    return _install


# --- Mock Streaming GenAI Interfaces ---

class MockChunk:
    """A chunk simulating individual NDJSON streamed token responses."""

    def __init__(self, *, parts: Sequence[types.Part] | None = None) -> None:
        c = types.Content(role="model", parts=parts) if parts is not None else None
        self.candidates = [_MockCandidate(c)] if c is not None else []


class MockStreamingClient:
    """A client simulating sequential streamed token listings."""

    def __init__(self, turns: Sequence[Sequence[MockChunk]]) -> None:
        self._turns = list(turns)
        self.history_captures: list[Any] = []  # type: ignore

        class _Models:
            def __init__(self, outer: "MockStreamingClient") -> None:
                self._outer = outer

            def generate_content_stream(
                self, *, model: str, contents: Any, config: Any,  # type: ignore
            ) -> Iterator[MockChunk]:
                self._outer.history_captures.append(list(contents))  # type: ignore
                return iter(self._outer._turns.pop(0))

        self.models = _Models(self)


@pytest.fixture
def text_chunk() -> Any:  # type: ignore
    """Fixture providing factory for text-based MockChunks."""
    return lambda t: MockChunk(parts=[types.Part(text=t)])


@pytest.fixture
def call_chunk() -> Any:  # type: ignore
    """Fixture providing factory for action function-call MockChunks."""
    return lambda name, args: MockChunk(
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


@pytest.fixture
def empty_chunk() -> Any:  # type: ignore
    """Fixture providing factory for empty/blocked MockChunks."""
    return lambda: MockChunk(parts=[])


@pytest.fixture
def install_mock_stream_client(monkeypatch: pytest.MonkeyPatch) -> Any:  # type: ignore
    """Install a mocked streaming client instance."""
    def _install(turns: Sequence[Sequence[MockChunk]]) -> MockStreamingClient:
        c = MockStreamingClient(turns)
        monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: c)
        return c
    return _install
