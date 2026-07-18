"""Tests verifying the deterministic local fallback engine."""

from typing import Any

import pytest

from backend import actions, database
from backend.fallback import fallback_answer


def profile(language: str = "en", stadium_id: str | None = None, needs: list[str] | None = None) -> dict[str, Any]:  # type: ignore
    return {"language": language, "needs": needs or [], "stadium_id": stadium_id}


# --- support ---

def test_english_support_question_mentions_facilities() -> None:
    ans = fallback_answer(
        "Is there wheelchair access for my seat?",
        profile("en", "new-york-new-jersey"),
    )
    assert "Stadium assistance details for MetLife Stadium" in ans
    assert "Accessible entrances" in ans
    assert "MetLife Gate" in ans
    assert "wheelchair" in ans.lower()


def test_calm_question_routes_to_calm_fields() -> None:
    ans = fallback_answer(
        "Where is the quietest place for my autistic son?",
        profile("en", "los-angeles"),
    )
    assert "Sensory and calm support" in ans
    assert "sensory" in ans.lower()


def test_unverified_stadium_gets_caveat() -> None:
    ans = fallback_answer("Is there a ramp?", profile("en", "dallas"))
    assert "not been officially verified" in ans


# --- languages ---

def test_spanish_baby_care_question_answered_in_spanish() -> None:
    ans = fallback_answer(
        "¿Dónde está el área de lactancia?", profile("es", "mexico-city"),
    )
    assert "lactancia" in ans
    assert "Ubicación" in ans
    assert "Nursing room near Puerta 1" in ans
    assert not ans.startswith("Yes")


def test_french_question_answered_in_french() -> None:
    ans = fallback_answer(
        "Où sont les toilettes accessibles ?", profile("fr", "toronto"),
    )
    assert "assistance" in ans.lower()
    assert "Toilettes accessibles" in ans
    assert "vérifiés" in ans


def test_unknown_language_code_falls_back_to_english() -> None:
    ans = fallback_answer("Is there wheelchair access?", profile("de", "dallas"))
    assert "Stadium assistance details for AT&T Stadium" in ans


def test_language_detected_from_message_when_profile_has_none() -> None:
    ans = fallback_answer("hola", {})
    assert "Copa Mundial" in ans


def test_unmatched_message_without_profile_language_falls_back_to_english() -> None:
    ans = fallback_answer("qwertyuiop", {})
    assert "I can assist" in ans


# --- arabic ---

def test_arabic_greeting_uses_arabic_template() -> None:
    ans = fallback_answer("السلام عليكم", profile("ar", None))
    assert "AssistArena" in ans
    assert "البطولة" in ans
    assert "مرحبا" in ans


def test_arabic_wheelchair_question_routes_and_renders_in_arabic() -> None:
    ans = fallback_answer(
        "أين مسار الكرسي المتحرك؟", profile("ar", "new-york-new-jersey"),
    )
    assert "المساعدة في ملعب MetLife Stadium" in ans
    assert "المداخل المتاحة" in ans
    assert "MetLife Gate" in ans


def test_arabic_schedule_final_date() -> None:
    ans = fallback_answer("متى النهائي؟", profile("ar", "new-york-new-jersey"))
    assert "المباراة النهائية" in ans
    assert "2026-07-19" in ans


def test_arabic_refreshments_localized() -> None:
    ans = fallback_answer("أين الماء؟", profile("ar", "guadalajara"))
    assert "الماء في Estadio Akron" in ans


def test_arabic_unverified_stadium_gets_caveat_in_arabic() -> None:
    ans = fallback_answer("هل يوجد منحدر؟", profile("ar", "dallas"))
    assert "لم يتم التحقق" in ans


def test_arabic_no_stadium_prompt_is_localized() -> None:
    ans = fallback_answer("هل يوجد غرفة حسية؟", profile("ar", None))
    assert "يرجى اختيار" in ans
    assert "MetLife Stadium" in ans


def test_arabic_clitic_prefixed_keyword_still_matches() -> None:
    ans = fallback_answer(
        "أريد مكاناً هادئاً لطفلي المصاب بالتوحد", profile("ar", "los-angeles"),
    )
    assert "الدعم الحسي" in ans


def test_arabic_profile_language_overrides_english_message() -> None:
    ans = fallback_answer("Is there wheelchair access?", profile("ar", "dallas"))
    assert "المساعدة في ملعب AT&T Stadium" in ans


def test_arabic_answers_are_deterministic() -> None:
    msg, prof = "أين المصعد؟", profile("ar", "los-angeles")
    assert fallback_answer(msg, prof) == fallback_answer(msg, prof)


# --- profile needs ---

def test_profile_need_used_when_message_has_no_need_keyword() -> None:
    ans = fallback_answer(
        "Is the stadium accessible?",
        profile("en", "new-york-new-jersey", ["auditory"]),
    )
    assert "Auditory assistance" in ans


def test_invalid_profile_need_falls_back_to_standard() -> None:
    ans = fallback_answer(
        "Is the stadium accessible?",
        profile("en", "new-york-new-jersey", ["flying"]),
    )
    assert "Accessible entrances" in ans


# --- amenities ---

def test_generic_amenities_question_lists_all_three_amenities() -> None:
    from backend.fallback import _render_amenity_answer

    stadium = database.find_stadium_by_id("new-york-new-jersey")
    assert stadium is not None
    ans = _render_amenity_answer(stadium, "amenities", "en")
    assert "baby care room" in ans
    assert "First aid" in ans
    assert "Prayer and multi-faith" in ans


# --- directions ---

def test_directions_question_mentions_an_entrance() -> None:
    ans = fallback_answer(
        "Which gate should I use to get in?", profile("en", "seattle"),
    )
    assert "Recommended entrance" in ans
    assert "Northwest Gate" in ans or "Southeast Gate" in ans


def test_directions_answer_warns_about_lift_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    real = actions.query_realtime_status
    monkeypatch.setattr(
        "backend.fallback.actions.query_realtime_status",
        lambda stadium_id, hour=None: real(stadium_id, hour=1),
    )
    ans = fallback_answer(
        "Which gate should I use?", profile("en", "mexico-city"),
    )
    assert "out of service" in ans


# --- no stadium ---

def test_no_stadium_asks_user_to_pick_one_with_examples() -> None:
    ans = fallback_answer("Is there a sensory room?", profile("en", None))
    assert "select a stadium" in ans
    assert "Estadio Azteca" in ans
    assert "MetLife Stadium" in ans


def test_no_stadium_prompt_is_localized() -> None:
    ans = fallback_answer(
        "¿Hay rampa para silla de ruedas?", profile("es", None),
    )
    assert "elija primero un estadio" in ans


def test_unknown_stadium_id_treated_as_no_stadium() -> None:
    ans = fallback_answer("Is there a ramp?", profile("en", "atlantis"))
    assert "select a stadium" in ans


# --- other intents ---

def test_greeting() -> None:
    ans = fallback_answer("Hello!", profile("en", None))
    assert "AssistArena" in ans


def test_fallback_help_for_unmatched_message() -> None:
    ans = fallback_answer("asdfghjkl", profile("en", "dallas"))
    assert "I can assist" in ans


def test_schedule_final_date() -> None:
    ans = fallback_answer(
        "When is the final?", profile("en", "new-york-new-jersey"),
    )
    assert "2026-07-19" in ans


def test_schedule_opening_match_for_hosting_stadium() -> None:
    ans = fallback_answer(
        "When is the opening match?", profile("en", "mexico-city"),
    )
    assert "opening match" in ans
    assert "2026-06-11" in ans


def test_schedule_works_without_stadium_in_french() -> None:
    ans = fallback_answer("Quand a lieu la finale ?", profile("fr", None))
    assert "2026-06-11" in ans
    assert "2026-07-19" in ans
    assert "finale" in ans


def test_refreshments_spanish() -> None:
    ans = fallback_answer("¿Dónde hay agua?", profile("es", "guadalajara"))
    assert "Agua en Estadio Akron" in ans


# --- determinism ---

def test_offline_answers_are_deterministic() -> None:
    cases = [
        ("Is there wheelchair access?", profile("en", "dallas")),
        ("¿Dónde está el área de lactancia?", profile("es", "mexico-city")),
        ("Hello!", profile("en", None)),
    ]
    for message, prof in cases:
        assert fallback_answer(message, prof) == fallback_answer(message, prof)


def test_no_emoji_in_answers() -> None:
    ans = fallback_answer(
        "Is there wheelchair access?", profile("en", "dallas"),
    )
    assert all(ord(ch) < 0x2600 for ch in ans)


def test_find_congestion_level_missing() -> None:
    from backend.fallback import _find_congestion_level
    res = _find_congestion_level([{"entrance": "Gate A", "congestion": "high"}], "Gate B")
    assert res == "low"


def test_detect_need_preference_from_profile_edge_case() -> None:
    from backend.fallback import _detect_need_preference
    res = _detect_need_preference("hello", {"needs": ["unknown"]})
    assert res == "standard"
    res2 = _detect_need_preference("hello", {"needs": "not-a-list"})
    assert res2 == "standard"
