"""Local deterministic fallback handler for AssistArena.

Used when the live Gemini API is unreachable or no API keys are configured.
Routes message keywords to intents and builds responses using localized templates.
"""

import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend import actions, database

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "assets" / "fallback_config.json"
with _CONFIG_DIR.open(encoding="utf-8") as _f:
    _FALLBACK_CONFIG = json.load(_f)

SUPPORTED_LANGS: tuple[str, ...] = tuple(_FALLBACK_CONFIG["SUPPORTED_LANGS"])
_INTENT_ORDER: tuple[str, ...] = tuple(_FALLBACK_CONFIG["INTENT_ORDER"])
_INTENT_KEYWORDS: dict[str, dict[str, list[str]]] = _FALLBACK_CONFIG["INTENT_KEYWORDS"]
_SUPPORT_KEYWORDS: dict[str, list[str]] = _FALLBACK_CONFIG["SUPPORT_KEYWORDS"]
_AMENITY_KEYWORDS: dict[str, list[str]] = _FALLBACK_CONFIG["AMENITY_KEYWORDS"]
_TRAFFIC_LABELS: dict[str, dict[str, str]] = _FALLBACK_CONFIG["TRAFFIC_LABELS"]
_FIELD_TRANSLATIONS: dict[str, dict[str, str]] = _FALLBACK_CONFIG["FIELD_TRANSLATIONS"]
_TEMPLATES: dict[str, dict[str, str]] = _FALLBACK_CONFIG["TEMPLATES"]


def _strip_accents(text: str) -> str:
    """Normalize text by lowecasing and removing accent diacritics."""
    normalized_form = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in normalized_form if not unicodedata.combining(c))


_ARABIC_PREFIXES = "وفبكل"


_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _has_keyword(normalized_message: str, keyword: str) -> bool:
    """Match word boundary, allowing for Arabic proclitics (e.g. wa/fa/bi/ka/li)."""
    regex_kw = re.escape(_strip_accents(keyword))
    pattern_str = rf"\b[{_ARABIC_PREFIXES}]{{0,2}}{regex_kw}\b"
    if pattern_str not in _REGEX_CACHE:
        _REGEX_CACHE[pattern_str] = re.compile(pattern_str)
    return _REGEX_CACHE[pattern_str].search(normalized_message) is not None


def _resolve_intent(normalized_message: str) -> tuple[str, str | None]:
    """Parse message intents. Returns (intent_name, language_code)."""
    for intent in _INTENT_ORDER:
        for lang in SUPPORTED_LANGS:
            for kw in _INTENT_KEYWORDS[intent][lang]:
                if _has_keyword(normalized_message, kw):
                    return intent, lang
    return "help", None


def _detect_lang(lang_code: object, detected: str | None) -> str:
    """Determine language fallback prioritizing profile preferences."""
    if isinstance(lang_code, str) and lang_code.strip():
        code = lang_code.strip().lower()[:2]
        return code if code in _TEMPLATES else "en"
    return detected if detected in _TEMPLATES else "en"


def _detect_keyword_category(msg: str) -> str | None:
    """Detect matching support needs keywords inside a message."""
    for category, keywords in _SUPPORT_KEYWORDS.items():
        if any(_has_keyword(msg, kw) for kw in keywords):
            return category
    return None


def _detect_profile_category(needs: object) -> str | None:
    """Detect matching support needs inside a profile dictionary."""
    if isinstance(needs, (list, tuple)):
        for raw in needs:
            if isinstance(raw, str) and raw.lower() in actions.PERMITTED_NEEDS:
                return raw.lower()
    return None


def _detect_need_preference(normalized_message: str, profile: Mapping[str, Any]) -> str:
    """Map message keywords or profile needs to a specific assistance needs category."""
    cat = _detect_keyword_category(normalized_message)
    if cat is not None:
        return cat
    p_cat = _detect_profile_category(profile.get("needs"))
    if p_cat is not None:
        return p_cat
    return "standard"


def _normalize_punctuation(fragment: str) -> str:
    """Add a period if the string does not end in terminal punctuation."""
    fragment = fragment.strip()
    return fragment if fragment.endswith((".", "!", "?")) else fragment + "."


def _strip_trailing_period(fragment: str) -> str:
    """Strip trailing periods to fit inside template parameters."""
    return fragment.strip().rstrip(".")


def _build_stadium_examples() -> str:
    """Build stadium name list examples for prompt fallbacks."""
    return ", ".join(
        f"{s['stadiumName']} ({s['id']})" for s in database.list_all_stadiums()[:3]
    )


def _render_facilities(facilities: Mapping[str, Any], labels: Mapping[str, str]) -> list[str]:
    """Map facility detail dictionaries to formatted template strings."""
    parts = []
    for field, val in facilities.items():
        if field == "entrances":
            names = ", ".join(e["entranceName"] for e in val if e["isAccessible"])
            parts.append(f"{labels['entrances']}: {names}.")
        else:
            parts.append(f"{labels[field]}: {_normalize_punctuation(val)}")
    return parts


def _render_support_answer(stadium_id: str, need: str, lang: str) -> str:
    """Format and join support services for a selected stadium."""
    res = actions.find_support_services(stadium_id, need=need)
    tpl, field_labels = _TEMPLATES[lang], _FIELD_TRANSLATIONS[lang]
    parts = [tpl["support_intro"].format(stadium=res["stadium_name"])]
    parts.extend(_render_facilities(res["facilities"], field_labels))
    if not res["isVerified"]:
        parts.append(tpl["unverified"])
    return " ".join(parts)


def _render_amenity_answer(
    stadium: database.Stadium, normalized_message: str, lang: str,
) -> str:
    """Find specific stadium amenities matching keywords, or list all."""
    tpl = _TEMPLATES[lang]
    amenities = stadium["amenities"]
    for service, keywords in _AMENITY_KEYWORDS.items():
        if any(_has_keyword(normalized_message, kw) for kw in keywords):
            return tpl[service].format(
                stadium=stadium["stadiumName"], value=_strip_trailing_period(amenities[service]),
            )
    return " ".join(
        tpl[service].format(
            stadium=stadium["stadiumName"], value=_strip_trailing_period(amenities[service]),
        )
        for service in _AMENITY_KEYWORDS
    )


def _find_congestion_level(traffic: list[dict[str, Any]], name: str) -> str:
    """Locate the congestion level descriptor for an entrance name."""
    for item in traffic:
        if item.get("entrance") == name:
            return str(item.get("congestion", "low"))
    return "low"


def _render_directions_answer(stadium_id: str, lang: str) -> str:
    """Render simulated live traffic status and calm directions for a stadium."""
    tpl = _TEMPLATES[lang]
    status = actions.query_realtime_status(stadium_id)
    info = actions.fetch_stadium_info(stadium_id)
    entrance_name = status["calmest_entrance"]
    entrance = actions._find_entrance(info["entrances"], entrance_name) or {}
    level = _find_congestion_level(status["entrance_traffic"], entrance_name)
    calm_details = actions.find_support_services(stadium_id, need="calm")
    parts = [
        tpl["directions"].format(
            stadium=info["stadiumName"],
            entrance=entrance_name,
            guidance=_strip_trailing_period(entrance.get("guidance", "")),
            level=_TRAFFIC_LABELS[lang][level],
            hint=_normalize_punctuation(calm_details["facilities"]["calmPathHint"]),
        ),
    ]
    if status["lift_outage"] is not None:
        parts.append(
            tpl["outage"].format(entrance=status["lift_outage"]["entrance"]),
        )
    return " ".join(parts)


def _render_stadium_schedule(stadium: database.Stadium, tpl: Mapping[str, str]) -> list[str]:
    """Retrieve stadium-specific opening and final dates, if applicable."""
    parts = []
    metadata = actions.fetch_stadium_info(stadium["id"])["schedule"]
    if "hosts_opening_match" in metadata:
        parts.append(
            tpl["schedule_opening"].format(
                stadium=stadium["stadiumName"], date=metadata["hosts_opening_match"],
            ),
        )
    if "hosts_final" in metadata:
        parts.append(
            tpl["schedule_final"].format(
                stadium=stadium["stadiumName"], date=metadata["hosts_final"],
            ),
        )
    return parts


def _render_general_schedule(db: Mapping[str, Any], tpl: Mapping[str, str]) -> str:
    """Format global opening and final match timelines."""
    opening, final = db["openingMatch"], db["final"]
    open_stadium = database.find_stadium_by_id(opening["stadiumId"])
    final_stadium = database.find_stadium_by_id(final["stadiumId"])
    return tpl["schedule_general"].format(
        open_date=opening["date"],
        open_stadium=open_stadium["stadiumName"] if open_stadium else "?",
        final_date=final["date"],
        final_stadium=final_stadium["stadiumName"] if final_stadium else "?",
    )


def _render_schedule_answer(stadium: database.Stadium | None, lang: str) -> str:
    """Retrieve match timeline details."""
    tpl = _TEMPLATES[lang]
    parts: list[str] = []
    if stadium is not None:
        parts.extend(_render_stadium_schedule(stadium, tpl))
    db = database.load_stadium_data()["tournament"]
    if not parts:
        parts.append(_render_general_schedule(db, tpl))
    parts.append(
        tpl["tickets"].format(
            categories=", ".join(db["accessibility_tickets"]["categories"]),
        ),
    )
    return " ".join(parts)


def _process_stadium_intent(
    intent: str,
    stadium: database.Stadium,
    normalized_message: str,
    profile: Mapping[str, Any],
    lang: str,
) -> str:
    """Forward stadium-specific query to corresponding formatter."""
    tpl = _TEMPLATES[lang]
    if intent == "support":
        return _render_support_answer(
            stadium["id"], _detect_need_preference(normalized_message, profile), lang,
        )
    if intent == "amenities":
        return _render_amenity_answer(stadium, normalized_message, lang)
    if intent == "refreshments":
        water_val = _strip_trailing_period(stadium["amenities"]["drinkingWater"])
        return tpl["refreshments"].format(
            stadium=stadium["stadiumName"],
            water=water_val,
        )
    return _render_directions_answer(stadium["id"], lang)


def _resolve_profile_stadium(profile: Mapping[str, Any]) -> database.Stadium | None:
    """Retrieve database stadium matching profile configurations."""
    sid = profile.get("stadium_id")
    if isinstance(sid, str):
        return database.find_stadium_by_id(sid)
    return None


def fallback_answer(message: str, profile: Mapping[str, Any] | None = None) -> str:
    """Deterministically resolve messages when operating in offline/fallback mode."""
    profile_dict = profile or {}
    normalized = _strip_accents(message or "")
    intent, detected_lang = _resolve_intent(normalized)
    lang = _detect_lang(profile_dict.get("language"), detected_lang)
    tpl = _TEMPLATES[lang]

    if intent == "greeting":
        return tpl["greeting"]
    if intent == "help":
        return tpl["help"]

    stadium = _resolve_profile_stadium(profile_dict)
    if intent == "schedule":
        return _render_schedule_answer(stadium, lang)

    if stadium is None:
        return tpl["pick_stadium"].format(examples=_build_stadium_examples())

    return _process_stadium_intent(intent, stadium, normalized, profile_dict, lang)
