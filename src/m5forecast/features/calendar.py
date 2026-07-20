"""Calendar features: seasonality, holidays, and event *windows*.

No shifting needed here — the calendar is known arbitrarily far into the
future (day-of-week for next month is not a forecast). This is exactly
TFT's "known future inputs" category (Phase 3), and it is why calendar
features are the safest features in the project.

EDA finding driving the design (Phase 6): events have SHAPES — Super Bowl
spikes the day BEFORE, Thanksgiving two days before, Christmas craters on
the day. A binary is_event flag cannot see build-ups, so we ship signed
distance features (days_to_next_event, days_since_event), computed once
on the 1,941-day calendar and broadcast to all series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EVENT_WINDOW_CAP = 30  # beyond a month from any event, distance is just "far"


def _event_distances(dates: pd.DataFrame) -> pd.DataFrame:
    """Per unique day: days to the next event and since the previous one (capped)."""
    cal = dates.sort_values("d").reset_index(drop=True)
    is_event = cal["event_name_1"].notna().to_numpy()
    n = len(cal)

    to_next = np.full(n, _EVENT_WINDOW_CAP, dtype="int16")
    nxt = _EVENT_WINDOW_CAP
    for i in range(n - 1, -1, -1):
        nxt = 0 if is_event[i] else min(nxt + 1, _EVENT_WINDOW_CAP)
        to_next[i] = nxt

    since = np.full(n, _EVENT_WINDOW_CAP, dtype="int16")
    prev = _EVENT_WINDOW_CAP
    for i in range(n):
        prev = 0 if is_event[i] else min(prev + 1, _EVENT_WINDOW_CAP)
        since[i] = prev

    cal["days_to_event"] = to_next
    cal["days_since_event"] = since
    return cal[["d", "days_to_event", "days_since_event"]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Date parts, weekend flag, Christmas flag, event distances. snap already exists (Phase 5)."""
    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["dom"] = d.day.astype("int8")
    df["week"] = d.isocalendar().week.astype("int8")
    df["month"] = d.month.astype("int8")
    df["year"] = (d.year - 2011).astype("int8")  # small int, model-friendly
    df["is_christmas"] = ((d.month == 12) & (d.day == 25)).astype("int8")

    dist = _event_distances(df[["d", "event_name_1"]].drop_duplicates("d"))
    df = df.merge(dist, on="d", how="left")
    df["is_event"] = (df["days_to_event"] == 0).astype("int8")
    return df
