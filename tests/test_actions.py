"""Tests for backend.actions: operations tools and the dispatcher function."""

import json

import pytest

from backend import actions, database
from backend.actions import (
    TRAFFIC_STATES,
    compile_arrival_guide,
    dispatch_action,
    fetch_stadium_info,
    find_support_services,
    query_realtime_status,
)

VALID = "new-york-new-jersey"
UNKNOWN = "atlantis"

ACTION_CALLS = {
    "fetch_stadium_info": lambda sid: fetch_stadium_info(sid),
    "find_support_services": lambda sid: find_support_services(sid),
    "query_realtime_status": lambda sid: query_realtime_status(sid, hour=13),
    "compile_arrival_guide": lambda sid: compile_arrival_guide(sid, ["physical"], hour=13),
}


@pytest.mark.parametrize("name", sorted(ACTION_CALLS))
def test_every_action_works_for_valid_stadium(name: str) -> None:
    result = ACTION_CALLS[name](VALID)
    assert isinstance(result, dict)
    assert "error" not in result
    json.dumps(result)


@pytest.mark.parametrize("name", sorted(ACTION_CALLS))
def test_every_action_returns_error_payload_for_unknown_stadium(name: str) -> None:
    result = ACTION_CALLS[name](UNKNOWN)
    assert "error" in result
    assert UNKNOWN in result["error"]


# --- fetch_stadium_info ---

def test_fetch_stadium_info_basic_fields() -> None:
    info = fetch_stadium_info("dallas")
    assert info["stadiumName"] == "AT&T Stadium"
    assert info["city"].startswith("Arlington")
    assert info["country"] == "USA"
    assert info["capacity_annotation"] == "approximate tournament capacity"
    assert [e["entranceName"] for e in info["entrances"]] == ["Entry A", "Entry E", "Entry K"]


def test_fetch_stadium_info_schedule_hosting() -> None:
    opening = fetch_stadium_info("mexico-city")["schedule"]
    assert opening["hosts_opening_match"] == "2026-06-11"
    assert "hosts_final" not in opening

    final = fetch_stadium_info(VALID)["schedule"]
    assert final["hosts_final"] == "2026-07-19"
    assert "hosts_opening_match" not in final

    neither = fetch_stadium_info("dallas")["schedule"]
    assert "hosts_opening_match" not in neither
    assert "hosts_final" not in neither
    assert len(neither["access_ticket_categories"]) == 3


def test_fetch_stadium_info_does_not_expose_shared_cached_dicts() -> None:
    info = fetch_stadium_info("dallas")
    info["entrances"][0]["entranceName"] = "MUTATED"
    assert database.find_stadium_by_id("dallas")["support"]["entrances"][0]["entranceName"] == "Entry A"


# --- find_support_services ---

SUPPORT_FIELDS = {
    "physical": {"entrances", "lifts", "accessibleSeating", "accessibleToilets"},
    "visual": {"visualAssist"},
    "auditory": {"auditoryAssist"},
    "calm": {"calmRoom", "calmPathHint"},
}


@pytest.mark.parametrize("need,fields", sorted(SUPPORT_FIELDS.items()))
def test_find_support_services_filters_by_need(need: str, fields: set[str]) -> None:
    result = find_support_services(VALID, need=need)
    assert result["need"] == need
    assert set(result["facilities"]) == fields
    assert "isVerified" in result


def test_find_support_services_standard_includes_everything() -> None:
    result = find_support_services(VALID, need="standard")
    assert set(result["facilities"]) >= {"entrances", "calmRoom", "visualAssist"}


def test_find_support_services_invalid_need_falls_back_to_standard() -> None:
    result = find_support_services(VALID, need="teleportation")
    assert result["need"] == "standard"
    assert "note" in result
    assert "error" not in result


def test_find_support_services_carries_is_verified_flag() -> None:
    assert find_support_services(VALID)["isVerified"] is True
    assert find_support_services("dallas")["isVerified"] is False


# --- query_realtime_status ---

def test_query_realtime_status_deterministic_for_fixed_stadium_and_hour() -> None:
    first = query_realtime_status("seattle", hour=9)
    second = query_realtime_status("seattle", hour=9)
    assert first == second


def test_query_realtime_status_shape_and_enums() -> None:
    status = query_realtime_status("seattle", hour=9)
    assert status["simulated"] is True
    entrance_names = {e["entranceName"] for e in database.find_stadium_by_id("seattle")["support"]["entrances"]}
    for entry in status["entrance_traffic"]:
        assert entry["entrance"] in entrance_names
        assert entry["congestion"] in TRAFFIC_STATES
    assert status["calmest_entrance"] in entrance_names


def test_query_realtime_status_calmest_entrance_is_accessible() -> None:
    accessible = {
        e["entranceName"]
        for e in database.find_stadium_by_id("seattle")["support"]["entrances"]
        if e["isAccessible"]
    }
    for hour in range(24):
        assert query_realtime_status("seattle", hour=hour)["calmest_entrance"] in accessible


def test_query_realtime_status_invalid_hour_falls_back_to_current_hour() -> None:
    status = query_realtime_status("seattle", hour="not-a-number")  # type: ignore
    assert "error" not in status
    assert 0 <= status["hour_utc"] <= 23


def test_query_realtime_status_lift_outage_keyed_to_entrance_name() -> None:
    # Deterministic seed known to produce an outage.
    status = query_realtime_status("mexico-city", hour=1)
    outage = status["lift_outage"]
    assert outage is not None
    entrance_names = {
        e["entranceName"] for e in database.find_stadium_by_id("mexico-city")["support"]["entrances"]
    }
    assert outage["entrance"] in entrance_names
    assert status["calmest_entrance"] != outage["entrance"]


def test_query_realtime_status_without_accessible_entrances_degrades_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_stadium = {
        "id": "no-access-stadium",
        "support": {
            "entrances": [
                {"entranceName": "Gate 1", "isAccessible": False, "guidance": ""},
                {"entranceName": "Gate 2", "isAccessible": False, "guidance": ""},
            ],
        },
    }
    monkeypatch.setattr(actions.database, "find_stadium_by_id", lambda sid: fake_stadium)
    for hour in range(24):
        status = actions.query_realtime_status("no-access-stadium", hour=hour)
        assert status["lift_outage"] is None
        assert status["calmest_entrance"] is None


# --- compile_arrival_guide ---

def test_compile_arrival_guide_structured_steps() -> None:
    plan = compile_arrival_guide(VALID, ["physical", "calm"], hour=14)
    step_actions = [step["action"] for step in plan["steps"]]
    assert step_actions[:3] == ["enter_stadium", "time_buffer", "amenities_en_route"]
    assert step_actions.count("need_support") == 2
    assert "lift_outage_warning" not in step_actions
    assert plan["needs"] == ["physical", "calm"]
    assert plan["simulated"] is True

    gate_step = plan["steps"][0]
    accessible = {
        e["entranceName"]
        for e in database.find_stadium_by_id(VALID)["support"]["entrances"]
        if e["isAccessible"]
    }
    assert gate_step["entrance"] in accessible

    arrive_step = plan["steps"][1]
    assert arrive_step["minutes_before_kickoff"] >= 60

    services_step = plan["steps"][2]
    assert {"drinkingWater", "medicalAid", "babyCare"} <= set(services_step)


def test_compile_arrival_guide_includes_lift_outage_warning_step() -> None:
    plan = compile_arrival_guide("mexico-city", ["physical"], hour=1)
    step_actions = [step["action"] for step in plan["steps"]]
    assert "lift_outage_warning" in step_actions
    warning = plan["steps"][step_actions.index("lift_outage_warning")]
    assert warning["entrance"]
    assert "maintenance" in warning["note"]


def test_compile_arrival_guide_invalid_needs_fall_back_with_note() -> None:
    plan = compile_arrival_guide(VALID, ["flying", "physical"], hour=13)
    assert plan["needs"] == ["physical"]
    assert "note" in plan

    plan = compile_arrival_guide(VALID, ["flying"], hour=13)
    assert plan["needs"] == ["standard"]
    assert "note" in plan


def test_compile_arrival_guide_deduplicates_repeated_needs() -> None:
    plan = compile_arrival_guide(VALID, ["physical", "physical"], hour=13)
    assert plan["needs"] == ["physical"]
    assert "note" not in plan


def test_compile_arrival_guide_accepts_needs_as_string_and_defaults() -> None:
    assert compile_arrival_guide(VALID, "physical", hour=13)["needs"] == ["physical"]  # type: ignore
    assert compile_arrival_guide(VALID, hour=13)["needs"] == ["standard"]


def test_compile_arrival_guide_echoes_language() -> None:
    assert compile_arrival_guide(VALID, ["standard"], language="es", hour=13)["language"] == "es"


# --- dispatch_action ---

def test_dispatch_action_returns_valid_json_string() -> None:
    raw = dispatch_action("fetch_stadium_info", {"stadium_id": "dallas"})
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed == fetch_stadium_info("dallas")


def test_dispatch_action_unknown_tool_returns_error_json() -> None:
    parsed = json.loads(dispatch_action("launch_rockets", {"stadium_id": "dallas"}))
    assert "error" in parsed
    assert "launch_rockets" in parsed["error"]


def test_dispatch_action_unknown_stadium_returns_error_json() -> None:
    parsed = json.loads(dispatch_action("fetch_stadium_info", {"stadium_id": UNKNOWN}))
    assert "error" in parsed


def test_dispatch_action_missing_or_malformed_args_never_raise() -> None:
    assert "error" in json.loads(dispatch_action("fetch_stadium_info", {}))
    assert "error" in json.loads(dispatch_action("fetch_stadium_info", None))
    assert "error" in json.loads(dispatch_action("fetch_stadium_info", {"stadium_id": 42}))


def test_dispatch_action_drops_unexpected_args() -> None:
    parsed = json.loads(
        dispatch_action("fetch_stadium_info", {"stadium_id": "dallas", "bogus": True}),
    )
    assert "error" not in parsed


def test_dispatch_action_validates_need_enum_via_fallback() -> None:
    parsed = json.loads(
        dispatch_action(
            "find_support_services", {"stadium_id": "dallas", "need": "warp"},
        ),
    )
    assert parsed["need"] == "standard"
    assert "note" in parsed


def test_dispatch_action_matches_direct_call_for_pinned_hour() -> None:
    raw = dispatch_action("query_realtime_status", {"stadium_id": "seattle", "hour": 9})
    assert json.loads(raw) == query_realtime_status("seattle", hour=9)


def test_dispatch_action_internal_failure_returns_error_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(stadium_id: str) -> None:
        raise RuntimeError("simulated internal failure")

    monkeypatch.setitem(actions._ACTION_REGISTRY, "fetch_stadium_info", (_boom, ("stadium_id",)))
    parsed = json.loads(dispatch_action("fetch_stadium_info", {"stadium_id": "dallas"}))
    assert "error" in parsed
    assert "fetch_stadium_info" in parsed["error"]


def test_find_entrance_missing() -> None:
    from backend.actions import _find_entrance
    res = _find_entrance([{"entranceName": "Gate A"}], "Gate B")
    assert res is None


def test_resolve_hour_invalid_type() -> None:
    from backend.actions import _resolve_hour
    res = _resolve_hour([])
    assert isinstance(res, int)
    assert 0 <= res < 24
