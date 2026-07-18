"""Access layer for the static tournament stadiums dataset (assets/stadiums.json).

Loads, validates, and exposes lookups and queries over stadium configuration details.
The underlying data is loaded on module initialization.
"""

import json
from pathlib import Path
from typing import Any

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "stadiums.json"

# Load the static payload once on module import
with _ASSETS_DIR.open(encoding="utf-8") as _f:
    _STADIUMS_PAYLOAD: dict[str, Any] = json.load(_f)

Stadium = dict[str, Any]

_STADIUMS_LIST: list[Stadium] = _STADIUMS_PAYLOAD.get("stadiums", [])
_STADIUMS_BY_ID: dict[str, Stadium] = {s["id"]: s for s in _STADIUMS_LIST if "id" in s}


def load_stadium_data() -> dict[str, Any]:
    """Retrieve the full stadium database payload."""
    return _STADIUMS_PAYLOAD


def list_all_stadiums() -> list[Stadium]:
    """Retrieve all stadium items in the database."""
    return _STADIUMS_LIST


def find_stadium_by_id(stadium_id: str) -> Stadium | None:
    """Retrieve a single stadium matching stadium_id, or None if not found."""
    return _STADIUMS_BY_ID.get(stadium_id)


def _match_stadium(stadium: Stadium, term: str, attributes: tuple[str, ...]) -> bool:
    """Check if a stadium attribute contains the search term case-insensitively."""
    for attr in attributes:
        val = stadium.get(attr)
        if val and term in str(val).lower():
            return True
    return False


def query_stadiums(search_term: str) -> list[Stadium]:
    """Search stadiums case-insensitively across name, publicName, fifaTitle, city, country."""
    term = search_term.strip().lower()
    if not term:
        return []

    attributes = ("stadiumName", "publicName", "fifaTitle", "city", "country")
    return [s for s in list_all_stadiums() if _match_stadium(s, term, attributes)]
