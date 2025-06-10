"""Simulation Time Class using datetime for implementation.
This class provides methods to manage and manipulate simulation time,
including incrementing time, checking the current time's properties,
and determining if the current time is a weekday."""
from datetime import datetime, timedelta
from typing import Optional


class SimTime:
    """Simulation Time Class using datetime for implementation."""

    def __init__(self, start_datetime: Optional[datetime] = None) -> None:
        self._dt = start_datetime or datetime(year=1, month=1, day=1, hour=0, minute=0)
        self.tick = 0.0

    @property
    def minute_of_day(self):
        return self._dt.hour * 60 + self._dt.minute

    @property
    def hour_of_day(self):
        return self._dt.hour

    @property
    def day_of_week(self):
        # Monday is 0, Sunday is 6 (to match Python's datetime)
        return self._dt.weekday()

    @property
    def day_of_year(self):
        return self._dt.timetuple().tm_yday

    @property
    def year(self):
        return self._dt.year

    @property
    def datetime(self) -> datetime:
        """Return the current simulation time as a datetime object."""
        return self._dt

    def increment(self, minutes: int = 60) -> None:
        """Increment the simulation time by a number of minutes (default 60)."""
        self._dt += timedelta(minutes=minutes)

    def is_weekday(self) -> bool:
        """Return True if it is a weekday (Monday=0, Sunday=6)."""
        return self._dt.weekday() < 5
