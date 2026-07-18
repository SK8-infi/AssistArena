"""Tests verifying the server FastAPI API routing, streaming, and rate limiting."""

import json
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

from backend import copilot
from backend.server import (
    RATE_LIMIT_CEILING,
    RateLimitBucket,
    RedisRateLimitBucket,
    _initialize_rate_limiter,
    rate_limiter,
)
from tests.conftest import MockChunk

STADIUM = "new-york-new-jersey"
UNKNOWN = "atlantis"


@pytest.fixture(autouse=True)
def _offline_env_and_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rate_limiter.flush()
    yield
    rate_limiter.flush()


# --- Endpoint REST API Inquiries ---

def test_chat_happy_path_offline(test_client: TestClient) -> None:
    resp = test_client.post(
        "/api/consult",
        json={
            "query": "wheelchair access?",
            "profile": {"language": "en", "needs": ["physical"], "stadium_id": STADIUM},
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert body["stadium_id"] == STADIUM


def test_chat_without_stadium_asks_to_pick_one(test_client: TestClient) -> None:
    resp = test_client.post(
        "/api/consult",
        json={"query": "which gate is accessible?", "profile": {"language": "en"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert body["answer"].strip()
    assert body["stadium_id"] is None


def test_chat_live_mode_passthrough(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.server.copilot.consult",
        lambda *a, **k: copilot.CopilotResult(text="Live answer.", mode="live"),
    )
    resp = test_client.post("/api/consult", json={"query": "hi", "profile": {"stadium_id": STADIUM}})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "live"
    assert resp.json()["answer"] == "Live answer."


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "hi", "profile": {"needs": ["fly"]}},
        {"query": "hi", "profile": {"language": "english"}},
        {"query": "hi", "history": [{"role": "user", "text": "t"}] * 21},
        {"query": "hi", "history": [{"role": "bot", "text": "t"}]},
        {"query": "hi", "junk": 1},
    ],
)
def test_chat_rejects_bad_input_with_422(test_client: TestClient, payload: dict[str, Any]) -> None:
    assert test_client.post("/api/consult", json=payload).status_code == 422


def test_list_stadiums(test_client: TestClient) -> None:
    resp = test_client.get("/api/stadiums")
    assert resp.status_code == 200
    stadiums = resp.json()["stadiums"]
    assert len(stadiums) == 16
    for s in stadiums:
        assert {"id", "stadiumName", "city", "country", "spectatorCapacity"} <= set(s)


def test_get_stadium_by_id(test_client: TestClient) -> None:
    resp = test_client.get(f"/api/stadiums/{STADIUM}")
    assert resp.status_code == 200
    assert resp.json()["stadiumName"] == "MetLife Stadium"


def test_get_unknown_stadium_404(test_client: TestClient) -> None:
    resp = test_client.get("/api/stadiums/atlantis")
    assert resp.status_code == 404


def test_search_stadiums_by_city_case_insensitive(test_client: TestClient) -> None:
    resp = test_client.get("/api/stadiums/search", params={"q": "MEXICO"})
    assert resp.status_code == 200
    stadiums = resp.json()["stadiums"]
    assert any(s["id"] == "mexico-city" for s in stadiums)
    for s in stadiums:
        assert {"id", "stadiumName", "city", "country", "spectatorCapacity"} <= set(s)


def test_search_stadiums_no_match_returns_empty_list(test_client: TestClient) -> None:
    resp = test_client.get("/api/stadiums/search", params={"q": "atlantis"})
    assert resp.status_code == 200
    assert resp.json() == {"stadiums": []}


def test_search_stadiums_validates_query(test_client: TestClient) -> None:
    assert test_client.get("/api/stadiums/search").status_code == 422
    assert test_client.get("/api/stadiums/search", params={"q": ""}).status_code == 422
    too_long = "x" * 65
    assert test_client.get("/api/stadiums/search", params={"q": too_long}).status_code == 422


def test_healthz_reports_offline_without_key(test_client: TestClient) -> None:
    resp = test_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "llm": "offline"}


def test_healthz_reports_live_with_key_but_never_echoes_it(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_key = "test-fake-key-never-real"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    resp = test_client.get("/healthz")
    assert resp.json()["llm"] == "live"
    assert fake_key not in resp.text


@pytest.mark.parametrize("path", ["/", "/api/stadiums", "/healthz"])
def test_security_headers_present(test_client: TestClient, path: str) -> None:
    headers = test_client.get(path).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_no_wildcard_cors(test_client: TestClient) -> None:
    headers = test_client.get("/api/stadiums").headers
    assert headers.get("access-control-allow-origin") != "*"


def test_rate_limit_429_after_burst(test_client: TestClient) -> None:
    payload = {"query": "hi", "profile": {"stadium_id": STADIUM}}
    for _ in range(RATE_LIMIT_CEILING):
        assert test_client.post("/api/consult", json=payload).status_code == 200
    assert test_client.post("/api/consult", json=payload).status_code == 429


def test_index_served(test_client: TestClient) -> None:
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "AssistArena" in resp.text


def test_static_assets_served(test_client: TestClient) -> None:
    assert test_client.get("/frontend/main.js").status_code == 200
    assert test_client.get("/frontend/theme.css").status_code == 200


# --- Rate Limiting Pruning & Threads ---

def test_prune_evicts_full_buckets_and_keeps_active_ones() -> None:
    limiter = RateLimitBucket(RATE_LIMIT_CEILING, 60.0, prune_limit=10)
    with limiter._lock:
        now = time.monotonic()
        for i in range(8):
            limiter._buckets[f"full-{i}"] = (limiter.capacity, now)
        for i in range(6):
            limiter._buckets[f"empty-{i}"] = (0.0, now)

    assert limiter.acquire("brand-new-ip") is True
    remaining = set(limiter._buckets)
    assert not any(k.startswith("full-") for k in remaining)
    assert sum(k.startswith("empty-") for k in remaining) == 6


def test_prune_is_behaviourally_a_no_op_for_evicted_ips() -> None:
    limiter = RateLimitBucket(RATE_LIMIT_CEILING, 60.0, prune_limit=5)
    with limiter._lock:
        for i in range(10):
            limiter._buckets[f"full-{i}"] = (limiter.capacity, time.monotonic())
    limiter.acquire("trigger")
    assert "full-0" not in limiter._buckets
    assert limiter.acquire("full-0") is True


def test_no_prune_below_threshold() -> None:
    limiter = RateLimitBucket(RATE_LIMIT_CEILING, 60.0, prune_limit=1000)
    for i in range(50):
        limiter.acquire(f"ip-{i}")
    assert len(limiter._buckets) == 50


def test_concurrent_allow_on_one_key_hands_out_exactly_capacity() -> None:
    capacity = 50
    limiter = RateLimitBucket(capacity, refill_window=10_000)
    attempts = 200
    barrier = threading.Barrier(attempts)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        ok = limiter.acquire("same-ip")
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(attempts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads)
    assert len(results) == attempts
    assert sum(results) == capacity


class _MockRedis:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, float]] = {}

    def register_script(self, _lua: str) -> Any:  # type: ignore
        def run(keys: Sequence[str], args: Sequence[float]) -> int:
            cap, rate, now, _ttl = args
            key = keys[0]
            bucket = self.store.get(key)
            tokens = bucket["tokens"] if bucket else cap
            ts = bucket["ts"] if bucket else now
            tokens = min(cap, tokens + max(0.0, now - ts) * rate)
            allowed = 0
            if tokens >= 1:
                tokens -= 1
                allowed = 1
            self.store[key] = {"tokens": tokens, "ts": now}
            return allowed
        return run

    def scan_iter(self, match: str | None = None) -> list[str]:
        prefix = match.rstrip("*") if match else ""
        return [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)


class _PingableMock(_MockRedis):
    def ping(self) -> bool:
        return True


def test_redis_limiter_allows_up_to_capacity_then_blocks() -> None:
    fake = _MockRedis()
    limiter = RedisRateLimitBucket(fake, RATE_LIMIT_CEILING, 60.0)  # type: ignore
    allowed = sum(limiter.acquire("1.2.3.4") for _ in range(RATE_LIMIT_CEILING))
    assert allowed == RATE_LIMIT_CEILING
    assert limiter.acquire("1.2.3.4") is False
    assert limiter.acquire("5.6.7.8") is True


def test_redis_limiter_namespaces_keys_and_reset_is_scoped() -> None:
    fake = _MockRedis()
    fake.store["unrelated:key"] = {"tokens": 1.0, "ts": 0.0}
    limiter = RedisRateLimitBucket(fake, RATE_LIMIT_CEILING, 60.0)  # type: ignore
    limiter.acquire("9.9.9.9")
    assert any(k.startswith("assistarena:rl:") for k in fake.store)
    limiter.flush()
    assert "unrelated:key" in fake.store
    assert not any(k.startswith("assistarena:rl:") for k in fake.store)


def test_redis_reset_with_no_keys_never_calls_delete() -> None:
    fake = _MockRedis()
    calls: list[Any] = []  # type: ignore
    fake.delete = lambda *keys: calls.append(keys)  # type: ignore
    limiter = RedisRateLimitBucket(fake, RATE_LIMIT_CEILING, 60.0)  # type: ignore
    limiter.flush()
    assert calls == []


def test_initialize_rate_limiter_defaults_to_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(_initialize_rate_limiter(), RateLimitBucket)


def test_initialize_rate_limiter_falls_back_when_redis_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6390/0")
    assert isinstance(_initialize_rate_limiter(), RateLimitBucket)


def test_initialize_rate_limiter_uses_redis_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import types as std_types
    fake_module = std_types.SimpleNamespace(
        Redis=std_types.SimpleNamespace(from_url=lambda *a, **k: _PingableMock()),
    )
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    limiter = _initialize_rate_limiter()
    assert isinstance(limiter, RedisRateLimitBucket)
    assert limiter.acquire("1.1.1.1") is True


# --- Streaming Chat Path Execution ---

def _with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_offline_stream_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    events = list(
        copilot.consult_stream("wheelchair access?", {"stadium_id": STADIUM, "language": "en"}),
    )
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]
    assert len(events) == 2


def test_live_direct_text_streams_incrementally(
    monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any, text_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    install_mock_stream_client([[text_chunk("The opening "), text_chunk("match is 2026-06-11.")]])

    events = list(copilot.consult_stream("when is kickoff?", {"language": "en"}))

    assert events[0] == ("meta", "live")
    deltas = [payload for kind, payload in events if kind == "delta"]
    assert len(deltas) == 2
    assert "".join(deltas) == "The opening match is 2026-06-11."


def test_live_tool_round_then_streamed_answer(
    monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any, text_chunk: Any, call_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    client = install_mock_stream_client(
        [
            [call_chunk("fetch_stadium_info", {"stadium_id": STADIUM})],
            [text_chunk("MetLife Stadium "), text_chunk("hosts the final.")],
        ],
    )

    events = list(
        copilot.consult_stream("tell me about MetLife", {"stadium_id": STADIUM, "language": "en"}),
    )

    assert events[0] == ("meta", "live")
    assert "".join(p for k, p in events if k == "delta") == "MetLife Stadium hosts the final."

    second = client.history_captures[1]
    assert len(second) == 3
    assert second[1].role == "model"
    assert second[2].role == "user"
    assert len(second[2].parts) == 1


def test_streamed_thought_parts_are_excluded_from_visible_text(
    monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    chunk = MockChunk(
        parts=[
            types.Part(text="internal reasoning", thought=True),
            types.Part(text="Gate A is quietest."),
        ],
    )
    install_mock_stream_client([[chunk]])

    events = list(copilot.consult_stream("quietest gate?", {"language": "en"}))

    deltas = "".join(payload for kind, payload in events if kind == "delta")
    assert deltas == "Gate A is quietest."


def test_live_blocked_no_text_yields_decline(
    monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any, empty_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    install_mock_stream_client([[empty_chunk()]])

    events = list(copilot.consult_stream("something blocked", {"language": "es"}))

    assert events[0] == ("meta", "live")
    assert events[-1] == ("delta", copilot._DECLINE_MESSAGES["es"])


def test_auth_error_before_any_text_falls_back_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(401, {"error": {"message": "bad key"}})

    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(
        copilot.consult_stream("hi", {"stadium_id": STADIUM, "language": "en"}),
    )
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]


def test_model_not_found_404_before_any_text_falls_back_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(404, {"error": {"message": "model not found"}})

    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(copilot.consult_stream("hi", {"language": "en"}))
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]


def test_server_error_before_any_text_falls_back_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ServerError(503, {"error": {"message": "unavailable"}})

    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(copilot.consult_stream("hi", {"language": "en"}))
    assert events[0] == ("meta", "offline")


def test_streamed_tool_loop_stops_at_iteration_cap(
    monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any, call_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    turns = [
        [call_chunk("fetch_stadium_info", {"stadium_id": STADIUM})]
        for _ in range(copilot._MAX_ACTION_LOOPS)
    ]
    client = install_mock_stream_client(turns)

    events = list(copilot.consult_stream("loop forever", {"language": "en"}))

    assert len(client.history_captures) == copilot._MAX_ACTION_LOOPS
    assert events[0] == ("meta", "live")
    assert events[-1] == ("delta", copilot._DECLINE_MESSAGES["en"])


def test_stream_400_client_error_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model: str, contents: Any, config: Any) -> None:  # type: ignore
                raise errors.ClientError(400, {"error": {"message": "our bug"}})

    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: _RaisingClient())

    with pytest.raises(errors.ClientError):
        list(copilot.consult_stream("hi", {"language": "en"}))


def test_midstream_failure_after_first_delta_ends_stream_gracefully(
    monkeypatch: pytest.MonkeyPatch, text_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)

    def _chunks() -> Iterator[Any]:  # type: ignore
        yield text_chunk("Partial ")
        raise ConnectionError("connection dropped mid-stream")

    class _Client:
        class models:
            @staticmethod
            def generate_content_stream(*, model: str, contents: Any, config: Any) -> Iterator[Any]:  # type: ignore
                return _chunks()

    monkeypatch.setattr("backend.copilot.genai.Client", lambda *a, **k: _Client())

    events = list(copilot.consult_stream("hi", {"language": "en"}))
    assert events == [("meta", "live"), ("delta", "Partial ")]


def test_empty_live_stream_yields_live_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    monkeypatch.setattr(
        "backend.copilot._stream_live_consultation", lambda *a, **k: iter(()),
    )

    events = list(copilot.consult_stream("hi", {"language": "en"}))
    assert events == [("meta", "live"), ("delta", copilot._DECLINE_MESSAGES["en"])]


# --- Endpoint Stream validation ---

def test_endpoint_streams_ndjson_offline(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    res = test_client.post(
        "/api/consult/stream",
        json={
            "query": "wheelchair access?",
            "profile": {"stadium_id": STADIUM, "language": "en", "needs": []},
            "history": [],
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]

    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[0] == {"type": "meta", "mode": "offline", "stadium_id": STADIUM}
    assert any(f["type"] == "delta" and f["text"] for f in frames)


def test_endpoint_streams_live_deltas(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch, install_mock_stream_client: Any, text_chunk: Any,  # type: ignore
) -> None:
    _with_key(monkeypatch)
    install_mock_stream_client([[text_chunk("Gate A "), text_chunk("is quietest.")]])

    res = test_client.post(
        "/api/consult/stream",
        json={
            "query": "quietest gate?",
            "profile": {"stadium_id": STADIUM, "language": "en", "needs": []},
            "history": [],
        },
    )
    assert res.status_code == 200
    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[0]["type"] == "meta" and frames[0]["mode"] == "live"
    text = "".join(f["text"] for f in frames if f["type"] == "delta")
    assert text == "Gate A is quietest."


def test_endpoint_emits_error_frame_and_no_traceback_when_stream_raises(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_key(monkeypatch)

    def _broken_stream(*a: Any, **k: Any) -> Iterator[Any]:  # type: ignore
        yield ("meta", "offline")
        raise RuntimeError("boom-sentinel")

    monkeypatch.setattr("backend.server.copilot.consult_stream", _broken_stream)

    res = test_client.post("/api/consult/stream", json={"query": "hi"})
    assert res.status_code == 200
    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[-1] == {"type": "error"}
    assert "boom-sentinel" not in res.text


def test_endpoint_rate_limited_returns_429(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_key(monkeypatch)
    payload = {
        "query": "hi",
        "profile": {"stadium_id": STADIUM, "language": "en", "needs": []},
        "history": [],
    }
    statuses = {test_client.post("/api/consult/stream", json=payload).status_code for _ in range(50)}
    assert 429 in statuses
