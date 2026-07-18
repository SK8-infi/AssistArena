"""Tests for checking models field validators and edge cases in copilot module."""

import pytest
from pydantic import ValidationError

from backend.copilot import _resolve_decline_lang
from backend.models import ConsultProfile, ConsultRequest, ConversationTurn


def test_consult_profile_validation_errors() -> None:
    """Verify that validator triggers on invalid profiles."""
    with pytest.raises(ValidationError, match="needs list cannot exceed 4 items"):
        ConsultProfile(needs=["physical", "visual", "auditory", "calm", "physical"])

    with pytest.raises(ValidationError, match="stadium_id cannot exceed 64 characters"):
        ConsultProfile(stadium_id="a" * 65)


def test_conversation_turn_validation_errors() -> None:
    """Verify that validator triggers on excessively long turns."""
    with pytest.raises(ValidationError, match="text cannot exceed 2000 characters"):
        ConversationTurn(role="user", text="a" * 2001)


def test_consult_request_validation_errors() -> None:
    """Verify validator limits conversation history to 20 turns."""
    history = [ConversationTurn(role="user", text="hello")] * 21
    with pytest.raises(ValidationError, match="history cannot exceed 20 turns"):
        ConsultRequest(query="hello", history=history)


def test_consult_request_query_validation() -> None:
    """Verify validator enforces query bounds."""
    with pytest.raises(ValidationError, match="query must be between 1 and 2000 characters"):
        ConsultRequest(query="")

    with pytest.raises(ValidationError, match="query must be between 1 and 2000 characters"):
        ConsultRequest(query="a" * 2001)


def test_resolve_decline_lang_unsupported() -> None:
    """Verify language fallbacks to en for unsupported or invalid language types."""
    assert _resolve_decline_lang({"language": "de"}) == "en"
    assert _resolve_decline_lang({"language": 123}) == "en"
