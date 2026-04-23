import time
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

_MS = ZoneInfo("Europe/Moscow")

HOLIDAYS = frozenset([
    datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3),
    datetime(2026, 1, 4), datetime(2026, 1, 5), datetime(2026, 1, 6),
    datetime(2026, 1, 7), datetime(2026, 1, 8),
    datetime(2026, 2, 23),
    datetime(2026, 3, 8),
    datetime(2026, 5, 1), datetime(2026, 5, 9),
    datetime(2026, 5, 11),
    datetime(2026, 6, 12),
    datetime(2026, 11, 4),
])


def now_ms() -> datetime:
    ts = time.time()
    return datetime.fromtimestamp(ts, _MS)


def is_in_session_range(dt: datetime) -> bool:
    """Check if datetime falls within Moscow Trading Session."""
    hour = dt.hour
    minute = dt.minute
    if dt.weekday() >= 5:  # Weekend
        return False
    if dt.date() in HOLIDAYS:  # Holiday
        return False
    if hour < 10 or hour >= 19:
        return False
    if hour == 18 and minute >= 50:
        return False
    return True


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def is_holiday(dt: datetime) -> bool:
    return dt.date() in HOLIDAYS