"""Mode detection: weekday SIGNAL, weekend LEARNING."""

from datetime import datetime
from zoneinfo import ZoneInfo

from idxbot.__main__ import detect_mode

JKT = ZoneInfo("Asia/Jakarta")


def test_weekday_signal():
    # Wednesday
    dt = datetime(2026, 9, 2, 10, 0, tzinfo=JKT)
    assert detect_mode(dt) == "SIGNAL"


def test_saturday_learning():
    dt = datetime(2026, 9, 5, 10, 0, tzinfo=JKT)
    assert detect_mode(dt) == "LEARNING"


def test_sunday_learning():
    dt = datetime(2026, 9, 6, 10, 0, tzinfo=JKT)
    assert detect_mode(dt) == "LEARNING"
