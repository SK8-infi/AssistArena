"""Data models and validation rules for the AssistArena API layer.

Defines Pydantic structures for user consultation requests, history items,
stadium summaries, and health status indicators.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

LitNeed = Literal["physical", "visual", "auditory", "calm"]


class ConsultProfile(BaseModel):
    """spectator profile context sent with each request."""

    model_config = {"extra": "ignore"}

    language: str = Field(default="en")
    needs: list[LitNeed] = Field(default_factory=list)
    stadium_id: str | None = Field(default=None)

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Strip whitespace, lowercase, and ensure it is a 2-letter language code."""
        val = v.strip().lower()
        if len(val) != 2:
            raise ValueError("language must be a 2-letter code")
        return val

    @field_validator("needs")
    @classmethod
    def validate_needs(cls, v: list[LitNeed]) -> list[LitNeed]:
        """Ensure needs has at most 4 elements."""
        if len(v) > 4:
            raise ValueError("needs list cannot exceed 4 items")
        return v

    @field_validator("stadium_id")
    @classmethod
    def validate_stadium_id(cls, v: str | None) -> str | None:
        """Ensure stadium_id does not exceed 64 characters."""
        if v is not None and len(v) > 64:
            raise ValueError("stadium_id cannot exceed 64 characters")
        return v


class ConversationTurn(BaseModel):
    """A prior turn in the conversation, round-tripped from the client."""

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not longer than 2000 characters."""
        if len(v) > 2000:
            raise ValueError("text cannot exceed 2000 characters")
        return v


class ConsultRequest(BaseModel):
    """Payload format for consultation requests."""

    model_config = {"extra": "forbid"}

    query: str
    profile: ConsultProfile = Field(default_factory=ConsultProfile)
    history: list[ConversationTurn] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Strip whitespace and validate query length constraints."""
        val = v.strip()
        if len(val) < 1 or len(val) > 2000:
            raise ValueError("query must be between 1 and 2000 characters")
        return val

    @field_validator("history")
    @classmethod
    def validate_history(cls, v: list[ConversationTurn]) -> list[ConversationTurn]:
        """Ensure history has at most 20 elements."""
        if len(v) > 20:
            raise ValueError("history cannot exceed 20 turns")
        return v


class ConsultResponse(BaseModel):
    """Server response envelop for consultations."""

    answer: str
    mode: Literal["live", "offline"]
    stadium_id: str | None = None


class StadiumSummary(BaseModel):
    """Public summary fields of a stadium."""

    id: str
    stadiumName: str  # noqa: N815
    city: str
    country: str
    spectatorCapacity: int  # noqa: N815


class StadiumList(BaseModel):
    """List response container for stadiums."""

    stadiums: list[StadiumSummary]


class ServiceHealth(BaseModel):
    """Liveness probe details."""

    status: Literal["ok"]
    llm: Literal["live", "offline"]
