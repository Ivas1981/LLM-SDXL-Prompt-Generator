from __future__ import annotations

import re
from typing import Any

VALID_TIMES = (
    "morning",
    "noon",
    "afternoon",
    "golden hour",
    "dusk",
    "evening",
    "night",
    "dawn",
)

DAY_TIMES = {"morning", "noon", "afternoon", "golden hour", "dawn"}
NIGHT_TIMES = {"dusk", "evening", "night"}

DAY_MARKERS = ("noon", "midday", "sunlight", "daylight", "bright sun", "sunshine", "blue sky")
NIGHT_MARKERS = ("moonlight", "starlight", "candlelight", "darkness", "night sky")
EVENING_MARKERS = ("dusk", "evening", "twilight", "sunset")

INCOMPATIBLE_WEATHER_FOR_DAY = ("blizzard", "heavy snow", "freezing fog")
INCOMPATIBLE_WEATHER_FOR_NIGHT = ("bright midday sun", "harsh noon light")

# Locations that are fully enclosed/indoors - no outdoor weather
INDOOR_LOCATIONS = (
    "bedroom", "bathroom", "kitchen", "office", "library", "studio", "living room",
    "classroom", "hospital", "mall", "shop", "store", "restaurant", "cafe", "bar",
    "club", "theater", "cinema", "museum", "gallery", "gym", "locker room", "shower",
    "hallway", "corridor", "elevator", "lobby", "reception", "waiting room",
    "warehouse", "factory", "workshop", "garage", "basement", "attic", "cellar",
    "indoor", "interior", "room", "building", "apartment", "house", "home",
)

# Locations that are underground or fully enclosed - no sunlight
UNDERGROUND_LOCATIONS = (
    "metro", "subway", "underground", "cave", "cavern", "tunnel", "bunker",
    "sewer", "catacomb", "mine", "dungeon", "basement", "cellar", "subterranean",
    "underground station", "subway station", "metro station",
)

# Outdoor locations that can have any weather
OUTDOOR_LOCATIONS = (
    "street", "road", "highway", "park", "forest", "beach", "desert", "mountain",
    "field", "meadow", "lake", "river", "ocean", "sea", "cliff", "canyon",
    "valley", "hill", "plain", "prairie", "savanna", "jungle", "swamp", "marsh",
    "outdoor", "exterior", "outside", "open air", "rooftop", "terrace", "balcony",
    "garden", "yard", "courtyard", "plaza", "square", "bridge", "dock", "pier",
    "harbor", "marina", "airport", "runway", "helipad", "construction site",
    "ruins", "cemetery", "graveyard", "battlefield", "stadium", "arena",
)

LOCATION_TIME_LEAK_PATTERNS = re.compile(
    r"\b(at night|at dusk|at dawn|at noon|at midday|at sunset|at sunrise|in the morning|in the evening|at midnight|at twilight|nighttime|daytime|at golden hour|during sunset|during sunrise|during golden hour)\b",
    re.IGNORECASE,
)

LOCATION_WEATHER_LEAK_PATTERNS = re.compile(
    r"\b(in the rain|under heavy snow|in fog|in the blizzard|raining outside|storm outside|rainy|snowy|foggy|sunny|overcast|in snow|under snow|during rain|during snow|in a blizzard|in heavy snow|in freezing fog)\b",
    re.IGNORECASE,
)

SUNLIGHT_MARKERS = ("sunlight", "sunshine", "bright sun", "direct sun", "harsh sun", "midday sun", "golden hour sun", "sun rays", "sunbeam", "solar", "daylight", "blue sky", "clear sky")
OUTDOOR_WEATHER_MARKERS = ("rain", "snow", "hail", "storm", "thunder", "lightning", "downpour", "drizzle", "shower", "blizzard", "fog", "mist", "overcast", "cloudy", "wind", "breeze", "gale")


def _is_indoor(location: str) -> bool:
    loc_lower = location.lower()
    return any(re.search(rf"\b{re.escape(indoor)}\b", loc_lower) for indoor in INDOOR_LOCATIONS)


def _is_underground(location: str) -> bool:
    loc_lower = location.lower()
    return any(re.search(rf"\b{re.escape(underground)}\b", loc_lower) for underground in UNDERGROUND_LOCATIONS)


def _is_outdoor(location: str) -> bool:
    loc_lower = location.lower()
    return any(re.search(rf"\b{re.escape(outdoor)}\b", loc_lower) for outdoor in OUTDOOR_LOCATIONS)


def validate_environment(env: dict[str, Any]) -> str | None:
    """Return None if environment is consistent, or a human-readable error string."""
    if not isinstance(env, dict):
        return "environment must be a JSON object"

    time_of_day = str(env.get("time_of_day", "")).strip().lower()
    if time_of_day not in VALID_TIMES:
        return f"time_of_day must be exactly one of {list(VALID_TIMES)}"

    weather = str(env.get("weather", "")).strip().lower()
    location = str(env.get("location", "")).strip()

    if LOCATION_TIME_LEAK_PATTERNS.search(location):
        return "location must not contain time-of-day phrases; put time in time_of_day field"

    if LOCATION_WEATHER_LEAK_PATTERNS.search(location):
        return "location must not contain weather phrases; put weather in weather field"

    # Time-of-day vs weather compatibility
    if time_of_day in DAY_TIMES:
        for marker in NIGHT_MARKERS:
            if marker in weather:
                return f"weather '{weather}' is incompatible with daytime '{time_of_day}'"
        for marker in EVENING_MARKERS:
            if marker in weather:
                return f"weather '{weather}' is incompatible with daytime '{time_of_day}'"

    if time_of_day in NIGHT_TIMES:
        for marker in DAY_MARKERS:
            if marker in weather:
                return f"weather '{weather}' is incompatible with night '{time_of_day}'"
        for marker in INCOMPATIBLE_WEATHER_FOR_NIGHT:
            if marker in weather:
                return f"weather '{weather}' is incompatible with night '{time_of_day}'"

    if time_of_day in {"noon", "afternoon", "morning", "dawn"}:
        for marker in INCOMPATIBLE_WEATHER_FOR_DAY:
            if marker in weather:
                return f"weather '{weather}' is unlikely at {time_of_day}"

    # Location-based weather/lighting constraints
    if _is_indoor(location) or _is_underground(location):
        # No outdoor weather indoors/underground
        for marker in OUTDOOR_WEATHER_MARKERS:
            if marker in weather:
                loc_type = "underground" if _is_underground(location) else "indoor"
                return f"weather '{weather}' is incompatible with {loc_type} location '{location}'"

    if _is_underground(location):
        # No sunlight underground
        for marker in SUNLIGHT_MARKERS:
            if marker in weather:
                return f"weather '{weather}' (sunlight) is incompatible with underground location '{location}'"

    # Indoor locations shouldn't have strong sunlight unless near windows
    if _is_indoor(location) and not _is_underground(location):
        # Allow some window light markers but not direct harsh sun
        harsh_sun = ("harsh sun", "direct sun", "midday sun", "bright sun")
        for marker in harsh_sun:
            if marker in weather:
                return f"weather '{weather}' (direct harsh sunlight) is unlikely for indoor location '{location}'"

    return None