"""Core backend action routines querying the tournament database.

Provides concrete operations for details retrieval, facilities lookup, live status
simulation, and arrival guide compilation. These routines are shared between the
AI and offline fallback path.
"""

import json
import random
from datetime import UTC, datetime
from typing import Any

from backend import database

JSONPayload = dict[str, Any]

PERMITTED_NEEDS: frozenset[str] = frozenset(
    {"physical", "visual", "auditory", "calm", "standard"},
)
TRAFFIC_STATES: tuple[str, ...] = ("low", "moderate", "high")

_SUPPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "physical": ("entrances", "lifts", "accessibleSeating", "accessibleToilets"),
    "visual": ("visualAssist",),
    "auditory": ("auditoryAssist",),
    "calm": ("calmRoom", "calmPathHint"),
    "standard": (
        "entrances",
        "accessibleSeating",
        "calmRoom",
        "auditoryAssist",
        "visualAssist",
        "lifts",
        "accessibleToilets",
        "calmPathHint",
    ),
}

_LIFT_OUTAGE_CHANCE = 0.15
_BUFFER_MINUTES = {"low": 60, "moderate": 75, "high": 90}

_OFF_PEAK_CURVE = (75, 20, 5)
_GROWING_CURVE = (45, 40, 15)
_BUSY_CURVE = (20, 45, 35)
_PEAK_CURVE = (10, 30, 60)
_DECREASING_CURVE = (35, 40, 25)

_TRAFFIC_WEIGHTS_BY_HOUR: tuple[tuple[int, int, int], ...] = (
    _OFF_PEAK_CURVE, _OFF_PEAK_CURVE, _OFF_PEAK_CURVE,
    _OFF_PEAK_CURVE, _OFF_PEAK_CURVE, _OFF_PEAK_CURVE,
    _GROWING_CURVE, _GROWING_CURVE, _GROWING_CURVE, _GROWING_CURVE, _GROWING_CURVE,
    _BUSY_CURVE, _BUSY_CURVE, _BUSY_CURVE, _BUSY_CURVE,
    _PEAK_CURVE, _PEAK_CURVE, _PEAK_CURVE, _PEAK_CURVE, _PEAK_CURVE, _PEAK_CURVE,
    _DECREASING_CURVE, _DECREASING_CURVE, _DECREASING_CURVE,
)


def _stadium_error(stadium_id: object) -> JSONPayload:
    """Compose a user-friendly JSON error block for invalid stadium queries."""
    stadium_list = ", ".join(s["id"] for s in database.list_all_stadiums()[:3])
    return {
        "error": (
            f"Unknown stadium identifier {stadium_id!r}. "
            f"Please supply one of the 16 host stadium IDs, e.g.: {stadium_list}."
        ),
    }


def _duplicate_entrances(stadium: database.Stadium) -> list[JSONPayload]:
    """Copy entrance properties to prevent mutations on cache databases."""
    return [dict(ent) for ent in stadium["support"]["entrances"]]


def _coerce_need(need: object) -> tuple[str, str | None]:
    """Ensure raw need strings are coerced to a valid needs parameter."""
    if isinstance(need, str) and need.strip().lower() in PERMITTED_NEEDS:
        return need.strip().lower(), None
    note = (
        f"Unrecognized preference {need!r}; showing standard assistance details. "
        f"Valid items: {', '.join(sorted(PERMITTED_NEEDS))}."
    )
    return "standard", note


def _normalize_needs_input(needs: object) -> list[Any]:
    """Coerce raw needs input to a list of elements."""
    if isinstance(needs, str):
        return [needs]
    if isinstance(needs, (list, tuple)):
        return list(needs)
    return []


def _process_needs_list(raw_needs: list[Any]) -> tuple[list[str], list[Any]]:
    """Verify raw need strings against allowed support tokens."""
    verified: list[str] = []
    ignored: list[Any] = []
    for raw in raw_needs:
        need = raw.strip().lower() if isinstance(raw, str) else raw
        if need in PERMITTED_NEEDS:
            if need not in verified:
                verified.append(need)
        else:
            ignored.append(raw)
    return verified, ignored


def _coerce_multiple_needs(needs: object) -> tuple[list[str], str | None]:
    """Clean a collection of raw needs into a list of verified support enums."""
    raw_needs = _normalize_needs_input(needs)
    if not raw_needs:
        return ["standard"], None
    verified, ignored = _process_needs_list(raw_needs)
    note = None
    if ignored:
        note = (
            f"Ignored unknown parameters {ignored!r}. "
            f"Valid options: {', '.join(sorted(PERMITTED_NEEDS))}."
        )
    if not verified:
        verified = ["standard"]
    return verified, note


def fetch_stadium_info(stadium_id: str) -> JSONPayload:
    """Retrieve primary attributes, entrances, capacities, and match metadata for a stadium."""
    stadium = database.find_stadium_by_id(stadium_id)
    if stadium is None:
        return _stadium_error(stadium_id)
    metadata = database.load_stadium_data()["tournament"]
    schedule: JSONPayload = {
        "access_ticket_categories": list(
            metadata["accessibility_tickets"]["categories"],
        ),
        "companion_status": metadata["accessibility_tickets"]["companion_status"],
    }
    if metadata["openingMatch"]["stadiumId"] == stadium_id:
        schedule["hosts_opening_match"] = metadata["openingMatch"]["date"]
    if metadata["final"]["stadiumId"] == stadium_id:
        schedule["hosts_final"] = metadata["final"]["date"]
    return {
        "id": stadium["id"],
        "stadiumName": stadium["stadiumName"],
        "publicName": stadium["publicName"],
        "fifaTitle": stadium["fifaTitle"],
        "city": stadium["city"],
        "country": stadium["country"],
        "spectatorCapacity": stadium["spectatorCapacity"],
        "capacity_annotation": "approximate tournament capacity",
        "entrances": _duplicate_entrances(stadium),
        "schedule": schedule,
    }


def find_support_services(stadium_id: str, need: str = "standard") -> JSONPayload:
    """Look up assistance and support services, filtered by a specific category."""
    stadium = database.find_stadium_by_id(stadium_id)
    if stadium is None:
        return _stadium_error(stadium_id)
    need, note = _coerce_need(need)
    support = stadium["support"]
    facilities: JSONPayload = {
        field: _duplicate_entrances(stadium) if field == "entrances" else support[field]
        for field in _SUPPORT_FIELDS[need]
    }
    result: JSONPayload = {
        "stadium_id": stadium["id"],
        "stadium_name": stadium["stadiumName"],
        "need": need,
        "facilities": facilities,
        "isVerified": support["isVerified"],
    }
    if note:
        result["note"] = note
    return result


def _select_calmest_entrance(
    entrance_traffic: list[JSONPayload], lift_outage_entrance: str | None,
) -> str | None:
    """Find the entrance with the lowest congestion, routing away from lift outages."""
    ordered = sorted(
        (ent for ent in entrance_traffic if ent["isAccessible"]),
        key=lambda ent: (
            ent["entrance"] == lift_outage_entrance,
            TRAFFIC_STATES.index(ent["congestion"]),
        ),
    )
    return ordered[0]["entrance"] if ordered else None


def _resolve_hour(hour: object) -> int:
    """Resolve raw hour input into a valid hour index between 0 and 23."""
    if hour is None:
        return datetime.now(UTC).hour
    if isinstance(hour, (int, float, str, bytes)):
        try:
            return int(hour) % 24
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC).hour


def _simulate_lift_outage(entrances: list[JSONPayload], rng: random.Random) -> JSONPayload | None:
    """Simulate a lift outage event at a random accessible entrance."""
    if rng.random() >= _LIFT_OUTAGE_CHANCE:
        return None
    accessible = [e["entranceName"] for e in entrances if e["isAccessible"]]
    if not accessible:
        return None
    outage_entrance = rng.choice(accessible)
    return {
        "entrance": outage_entrance,
        "note": (
            f"Elevator near {outage_entrance} is currently undergoing maintenance; "
            "assistance is available at alternative accessible entrances."
        ),
    }


def query_realtime_status(stadium_id: str, hour: int | None = None) -> JSONPayload:
    """Simulate real-time entrance traffic and system facility outages for a stadium."""
    stadium = database.find_stadium_by_id(stadium_id)
    if stadium is None:
        return _stadium_error(stadium_id)
    hour_idx = _resolve_hour(hour)
    rng = random.Random(f"{stadium_id}-{hour_idx}")  # noqa: S311 # nosec
    weights = _TRAFFIC_WEIGHTS_BY_HOUR[hour_idx]
    entrances = stadium["support"]["entrances"]
    entrance_traffic = [
        {
            "entrance": ent["entranceName"],
            "isAccessible": ent["isAccessible"],
            "congestion": rng.choices(TRAFFIC_STATES, weights=weights, k=1)[0],
        }
        for ent in entrances
    ]
    lift_outage = _simulate_lift_outage(entrances, rng)
    outage_ent = lift_outage["entrance"] if lift_outage else None
    return {
        "simulated": True,
        "stadium_id": stadium["id"],
        "hour_utc": hour_idx,
        "entrance_traffic": entrance_traffic,
        "lift_outage": lift_outage,
        "calmest_entrance": _select_calmest_entrance(entrance_traffic, outage_ent),
    }


def _get_need_tips(need: str, support: JSONPayload) -> JSONPayload:
    """Extract support tips associated with a specific accessibility need category."""
    return {
        field: support[field]
        for field in _SUPPORT_FIELDS[need]
        if field != "entrances"
    }


def _find_entrance(entrances: list[JSONPayload], name: str) -> JSONPayload | None:
    """Locate an entrance record by name."""
    for e in entrances:
        if e["entranceName"] == name:
            return e
    return None


def _build_arrival_steps(
    support: JSONPayload,
    amenities: JSONPayload,
    entrance_name: str,
    congestion: str,
    valid_needs: list[str],
    lift_outage: JSONPayload | None,
) -> list[JSONPayload]:
    """Assemble step-by-step navigation timeline details."""
    entrance = _find_entrance(support["entrances"], entrance_name)
    has_specific_need = (valid_needs != ["standard"])

    steps: list[JSONPayload] = [
        {
            "action": "enter_stadium",
            "entrance": entrance_name,
            "entrance_guidance": entrance["guidance"] if entrance else "",
            "congestion": congestion,
            "reason": "least congested accessible entrance right now (simulated live data)",
        },
        {
            "action": "time_buffer",
            "minutes_before_kickoff": (
                _BUFFER_MINUTES[congestion] + (15 if has_specific_need else 0)
            ),
        },
        {
            "action": "amenities_en_route",
            "drinkingWater": amenities["drinkingWater"],
            "medicalAid": amenities["medicalAid"],
            "babyCare": amenities["babyCare"],
        },
    ]
    for need in valid_needs:
        steps.append({
            "action": "need_support",
            "need": need,
            "tips": _get_need_tips(need, support),
        })

    if lift_outage is not None:
        steps.append(
            {
                "action": "lift_outage_warning",
                "entrance": lift_outage["entrance"],
                "note": lift_outage["note"],
            },
        )
    for i, step in enumerate(steps, start=1):
        step["step"] = i
    return steps


def compile_arrival_guide(
    stadium_id: str,
    needs: list[str] | None = None,
    language: str = "en",
    hour: int | None = None,
) -> JSONPayload:
    """Compile a structured step-by-step navigation timeline for a stadium visitor."""
    stadium = database.find_stadium_by_id(stadium_id)
    if stadium is None:
        return _stadium_error(stadium_id)
    valid_needs, note = _coerce_multiple_needs(needs)
    status = query_realtime_status(stadium_id, hour=hour)
    support = stadium["support"]
    amenities = stadium["amenities"]

    entrance_name = status["calmest_entrance"]
    congestion = next(
        (
            item["congestion"]
            for item in status["entrance_traffic"]
            if item["entrance"] == entrance_name
        ),
        "low",
    )

    steps = _build_arrival_steps(
        support,
        amenities,
        entrance_name,
        congestion,
        valid_needs,
        status["lift_outage"],
    )

    result: JSONPayload = {
        "stadium_id": stadium["id"],
        "stadium_name": stadium["stadiumName"],
        "language": language,
        "needs": valid_needs,
        "simulated": True,
        "isVerified": support["isVerified"],
        "steps": steps,
    }
    if note:
        result["note"] = note
    return result


_ACTION_REGISTRY: dict[str, tuple[Any, tuple[str, ...]]] = {
    "fetch_stadium_info": (fetch_stadium_info, ("stadium_id",)),
    "find_support_services": (find_support_services, ("stadium_id", "need")),
    "query_realtime_status": (query_realtime_status, ("stadium_id", "hour")),
    "compile_arrival_guide": (compile_arrival_guide, ("stadium_id", "needs", "language", "hour")),
}


def _prepare_action_kwargs(
    args: dict[str, Any] | None, accepted: tuple[str, ...],
) -> dict[str, Any]:
    """Filter arguments keeping only those accepted by the target action."""
    if not isinstance(args, dict):
        return {}
    return {k: v for k, v in args.items() if k in accepted}


def _is_invalid_stadium(stadium_id: object) -> bool:
    """Verify if a stadium identifier is malformed or not found in database."""
    return not isinstance(stadium_id, str) or database.find_stadium_by_id(stadium_id) is None


def dispatch_action(name: str, args: dict[str, Any] | None) -> str:
    """Execute a specified action by name and serialize its output as JSON."""
    if not isinstance(name, str) or name not in _ACTION_REGISTRY:
        payload: JSONPayload = {
            "error": (
                f"Unknown action {name!r}. "
                f"Available options: {', '.join(sorted(_ACTION_REGISTRY))}."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)
    func, accepted = _ACTION_REGISTRY[name]
    kwargs = _prepare_action_kwargs(args, accepted)
    if _is_invalid_stadium(kwargs.get("stadium_id")):
        return json.dumps(_stadium_error(kwargs.get("stadium_id")), ensure_ascii=False)
    try:
        result = func(**kwargs)
    except Exception as exc:  # noqa: BLE001 — defensive catch to prevent unhandled tool crashes
        result = {"error": f"Action {name!r} failed with: {exc}"}
    return json.dumps(result, ensure_ascii=False)
