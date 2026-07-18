"""Tests verifying stadium static database loading and search querying."""

from backend import database

SUPPORT_KEYS = {
    "entrances",
    "accessibleSeating",
    "calmRoom",
    "auditoryAssist",
    "visualAssist",
    "lifts",
    "accessibleToilets",
    "calmPathHint",
    "isVerified",
}

AMENITIES_KEYS = {"drinkingWater", "medicalAid", "babyCare", "prayerSpace"}


def test_sixteen_stadiums() -> None:
    assert len(database.list_all_stadiums()) == 16


def test_stadium_ids_unique() -> None:
    ids = [s["id"] for s in database.list_all_stadiums()]
    assert len(ids) == len(set(ids))


def test_tournament_schedule_references_valid_stadiums() -> None:
    db = database.load_stadium_data()
    ids = {s["id"] for s in db["stadiums"]}
    assert db["tournament"]["openingMatch"]["stadiumId"] in ids
    assert db["tournament"]["final"]["stadiumId"] in ids


def test_every_stadium_has_complete_support_keys() -> None:
    for s in database.list_all_stadiums():
        assert set(s["support"]) == SUPPORT_KEYS, s["id"]


def test_every_stadium_has_complete_amenities_keys() -> None:
    for s in database.list_all_stadiums():
        assert set(s["amenities"]) == AMENITIES_KEYS, s["id"]


def test_every_stadium_has_accessible_entrance() -> None:
    for s in database.list_all_stadiums():
        entrances = s["support"]["entrances"]
        assert any(e["isAccessible"] for e in entrances), s["id"]


def test_find_stadium_unknown_returns_none() -> None:
    assert database.find_stadium_by_id("invalid-id") is None


def test_search_stadiums_case_insensitively() -> None:
    metlife = database.query_stadiums("metlife")
    assert any(s["id"] == "new-york-new-jersey" for s in metlife)

    mexico = database.query_stadiums("MEXICO")
    assert any(s["id"] == "mexico-city" for s in mexico)


def test_search_stadiums_empty_query_returns_nothing() -> None:
    assert database.query_stadiums("") == []
    assert database.query_stadiums("   ") == []


def test_search_stadiums_no_match_returns_empty() -> None:
    assert database.query_stadiums("invalid-id") == []
