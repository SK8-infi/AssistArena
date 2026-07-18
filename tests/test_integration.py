"""Full-stack integration tests for AssistArena APIs and copilot functions."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.models import ConsultRequest
from tests.conftest import MockResponse


def test_live_roundtrip_through_api(
    test_client: TestClient, install_mock_client: Any, make_function_call: Any, stadium_id: str, monkeypatch: pytest.MonkeyPatch,  # type: ignore
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    install_mock_client(
        [
            MockResponse(
                function_calls=[
                    make_function_call(
                        "find_support_services",
                        {"stadium_id": stadium_id, "need": "physical"},
                    ),
                ],
                model_turn={"role": "model"},  # type: ignore
            ),
            MockResponse(text="Wheelchair access is available on all levels."),
        ],
    )

    resp = test_client.post(
        "/api/consult",
        json={
            "query": "wheelchair access at MetLife?",
            "profile": {"language": "en", "needs": ["physical"], "stadium_id": stadium_id},
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "live"
    assert body["answer"] == "Wheelchair access is available on all levels."
    assert body["stadium_id"] == stadium_id


def test_offline_roundtrip_through_api(test_client: TestClient, stadium_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    resp = test_client.post(
        "/api/consult",
        json={"query": "where is the nursing room?", "profile": {"stadium_id": stadium_id}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert body["answer"].strip()


def test_consult_request_schema_defaults() -> None:
    req = ConsultRequest(query="hi")
    assert req.profile.language == "en"
    assert req.profile.needs == []
    assert req.profile.stadium_id is None
    assert req.history == []
