"""Tests for backend.copilot: Google GenAI client routing and fallbacks."""

from typing import Any

import pytest
from google.genai import errors, types

from backend import copilot

STADIUM = "new-york-new-jersey"


class _MockCall:
    def __init__(self, name: str, args: dict[str, Any]) -> None:  # type: ignore
        self.name = name
        self.args = args


class _MockCandidate:
    def __init__(self, content: types.Content | None = None) -> None:
        self.content = content


class _MockResponse:
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


class _MockClient:
    def __init__(self, responses: list[_MockResponse]) -> None:
        self._responses = list(responses)
        self.contents_snapshots: list[Any] = []  # type: ignore

        class _Models:
            def __init__(self, outer: "_MockClient") -> None:
                self._outer = outer

            def generate_content(
                self, *, model: str, contents: Any, config: Any,  # type: ignore
            ) -> _MockResponse:
                self._outer.contents_snapshots.append(list(contents))  # type: ignore
                return self._outer._responses.pop(0)

        self.models = _Models(self)


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:  # type: ignore
    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: client)


def _with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_function_call_roundtrip_appends_model_turn_and_one_user_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    model_turn = types.Content(role="model", parts=[types.Part(text="thinking")])
    first = _MockResponse(
        function_calls=[
            _MockCall("find_support_services", {"stadium_id": STADIUM, "need": "physical"}),  # type: ignore
            _MockCall("query_realtime_status", {"stadium_id": STADIUM}),  # type: ignore
        ],
        model_turn=model_turn,
    )
    final = _MockResponse(text="MetLife has wheelchair access on all levels.")
    client = _MockClient([first, final])
    _patch_client(monkeypatch, client)

    reply = copilot.consult(
        "wheelchair access at MetLife?", profile={"stadium_id": STADIUM, "language": "en"},
    )

    assert reply.mode == "live"
    assert reply.text == "MetLife has wheelchair access on all levels."
    assert reply.actions_invoked == ["find_support_services", "query_realtime_status"]

    second = client.contents_snapshots[1]
    assert second[1] is model_turn
    func_content = second[2]
    assert func_content.role == "user"
    assert len(func_content.parts) == 2


def test_plain_text_response_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient([_MockResponse(text="The opening match is on 2026-06-11.")])
    _patch_client(monkeypatch, client)

    reply = copilot.consult("when is the opening match?", profile={"language": "en"})

    assert reply.mode == "live"
    assert reply.text == "The opening match is on 2026-06-11."
    assert reply.actions_invoked == []
    assert len(client.contents_snapshots) == 1


def test_blocked_none_text_returns_polite_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient([_MockResponse(text=None)])
    _patch_client(monkeypatch, client)

    reply = copilot.consult("something blocked", profile={"language": "es"})

    assert reply.mode == "live"
    assert reply.text == copilot._DECLINE_MESSAGES["es"]


def test_function_call_without_model_content_breaks_loop_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient(
        [_MockResponse(function_calls=[_MockCall("fetch_stadium_info", {"stadium_id": STADIUM})])],  # type: ignore
    )
    _patch_client(monkeypatch, client)

    reply = copilot.consult(
        "tell me about MetLife", profile={"stadium_id": STADIUM, "language": "en"},
    )

    assert reply.mode == "live"
    assert reply.text == copilot._DECLINE_MESSAGES["en"]
    assert reply.actions_invoked == []


def test_decline_in_unsupported_language_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient([_MockResponse(text=None)])
    _patch_client(monkeypatch, client)

    reply = copilot.consult("something blocked", profile={"language": "de"})

    assert reply.text == copilot._DECLINE_MESSAGES["en"]


def test_history_turns_without_text_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient([_MockResponse(text="ok")])
    _patch_client(monkeypatch, client)

    copilot.consult(
        "next question",
        profile={"language": "en"},
        history=[{"role": "user", "text": ""}, {"role": "assistant"}],
    )

    assert len(client.contents_snapshots[0]) == 1


def test_client_error_401_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(401, {"error": {"message": "bad key"}})

    _patch_client(monkeypatch, _RaisingClient())

    reply = copilot.consult(
        "wheelchair?", profile={"stadium_id": STADIUM, "language": "en"},
    )

    assert reply.mode == "offline"
    assert reply.text


def test_model_not_found_404_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(404, {"error": {"message": "model not found"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = copilot.consult("hello", profile={"language": "en"})
    assert reply.mode == "offline"
    assert reply.text


def test_rate_limit_429_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(429, {"error": {"message": "rate limited"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = copilot.consult("hello", profile={"language": "en"})
    assert reply.mode == "offline"


def test_server_error_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ServerError(503, {"error": {"message": "unavailable"}})

    _patch_client(monkeypatch, _RaisingClient())
    reply = copilot.consult("hello", profile={"language": "en"})
    assert reply.mode == "offline"


def test_connection_error_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise ConnectionError("network down")

    _patch_client(monkeypatch, _RaisingClient())
    reply = copilot.consult("hello", profile={"language": "en"})
    assert reply.mode == "offline"


def test_non_fallback_client_error_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(400, {"error": {"message": "our bug"}})

    _patch_client(monkeypatch, _RaisingClient())
    with pytest.raises(errors.ClientError):
        copilot.consult("hello", profile={"language": "en"})


def test_no_api_key_goes_straight_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def _boom(*a: Any, **k: Any) -> None:
        raise AssertionError("Client must not be constructed without a key")

    monkeypatch.setattr("backend.copilot.genai.Client", _boom)

    reply = copilot.consult(
        "quietest gate?",
        profile={"stadium_id": STADIUM, "language": "en", "needs": ["cal"]},
    )
    assert reply.mode == "offline"
    assert reply.text


def test_gemini_client_is_constructed_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    constructed = []

    def _factory(*a: Any, **k: Any) -> Any:
        client = _MockClient([_MockResponse(text="one"), _MockResponse(text="two")])
        constructed.append(client)
        return client

    monkeypatch.setattr("backend.copilot.genai.Client", _factory)

    assert copilot.consult("first", profile={"language": "en"}).text == "one"
    assert copilot.consult("second", profile={"language": "en"}).text == "two"
    assert len(constructed) == 1


def test_tool_loop_stops_at_iteration_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    model_turn = types.Content(role="model", parts=[types.Part(text="loop")])
    responses = [
        _MockResponse(
            function_calls=[_MockCall("fetch_stadium_info", {"stadium_id": STADIUM})],  # type: ignore
            model_turn=model_turn,
        )
        for _ in range(copilot._MAX_ACTION_LOOPS)
    ]
    client = _MockClient(responses)
    _patch_client(monkeypatch, client)

    reply = copilot.consult("loop forever", profile={"language": "en"})

    assert len(client.contents_snapshots) == copilot._MAX_ACTION_LOOPS
    assert len(reply.actions_invoked) == copilot._MAX_ACTION_LOOPS
    assert reply.text == copilot._DECLINE_MESSAGES["en"]


def test_history_round_trips_as_alternating_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _MockClient([_MockResponse(text="ok")])
    _patch_client(monkeypatch, client)

    copilot.consult(
        "and the nearest restroom?",
        profile={"stadium_id": STADIUM, "language": "en"},
        history=[
            {"role": "user", "text": "wheelchair access?"},
            {"role": "assistant", "text": "Yes, on all levels."},
        ],
    )

    contents = client.contents_snapshots[0]
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
