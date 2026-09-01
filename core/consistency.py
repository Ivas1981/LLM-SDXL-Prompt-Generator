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

DAY_TIMES = {"morning", "noon", "afternoon", "golden hour"}
NIGHT_TIMES = {"dusk", "evening", "night"}

DAY_MARKERS = ("noon", "midday", "sunlight", "daylight", "bright sun", "sunshine", "blue sky")
NIGHT_MARKERS = ("moonlight", "starlight", "candlelight", "darkness", "night sky")
EVENING_MARKERS = ("dusk", "evening", "twilight", "sunset")

INCOMPATIBLE_WEATHER_FOR_DAY = ("blizzard", "heavy snow", "freezing fog")
INCOMPATIBLE_WEATHER_FOR_NIGHT = ("bright midday sun", "harsh noon light")

LOCATION_TIME_LEAK_PATTERNS = re.compile(
    r"\b(at night|at dusk|at dawn|at noon|at midday|at sunset|at sunrise|in the morning|in the evening|at midnight|at twilight|nighttime|daytime|at golden hour|during sunset|during sunrise|during golden hour)\b",
    re.IGNORECASE,
)

LOCATION_WEATHER_LEAK_PATTERNS = re.compile(
    r"\b(in the rain|under heavy snow|in fog|in the blizzard|raining outside|storm outside|rainy|snowy|foggy|sunny|overcast|in snow|under snow|during rain|during snow|in a blizzard|in heavy snow|in freezing fog)\b",
    re.IGNORECASE,
)


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

    if time_of_day in {"noon", "afternoon", "morning"}:
        for marker in INCOMPATIBLE_WEATHER_FOR_DAY:
            if marker in weather:
                return f"weather '{weather}' is unlikely at {time_of_day}"

    return None