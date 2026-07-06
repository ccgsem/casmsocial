"""Simulation time utilities."""

from __future__ import annotations

from datetime import datetime, timedelta


class SimTime:
    """Convenience wrapper around :class:`datetime.datetime`.

    The class keeps simulation time normalized to minute resolution, exposes
    helpers for frequently used calendar properties, and provides a flexible
    ``increment`` method that defaults to one hour steps but can consume any
    combination of days, hours, minutes, or a ``timedelta``.
    """

    DEFAULT_START = datetime(year=2025, month=1, day=1)

    def __init__(self, start_datetime: datetime | None = None) -> None:
        start = start_datetime or self.DEFAULT_START
        self._start = start.replace(second=0, microsecond=0)
        self._dt = self._start
        self.tick = 0.0  # total minutes elapsed

    @property
    def datetime(self) -> datetime:
        """Current simulation timestamp."""
        return self._dt

    @property
    def start_datetime(self) -> datetime:
        """Datetime the simulation was initialized with."""
        return self._start

    @property
    def minute_of_day(self) -> int:
        """Minutes counted from 00:00 of the current day."""
        return self._dt.hour * 60 + self._dt.minute

    @property
    def hour_of_day(self) -> int:
        return self._dt.hour

    @property
    def day_of_week(self) -> int:
        """Python weekday where Monday is 0 and Sunday is 6."""
        return self._dt.weekday()

    @property
    def day_of_year(self) -> int:
        return self._dt.timetuple().tm_yday

    @property
    def year(self) -> int:
        return self._dt.year

    @property
    def date(self):
        return self._dt.date()

    @property
    def time(self):
        return self._dt.time().replace(second=0, microsecond=0)

    @property
    def elapsed(self) -> timedelta:
        """Time elapsed since the simulation start."""
        return self._dt - self._start

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed.total_seconds() / 60.0

    @property
    def is_weekday(self) -> bool:
        return self.day_of_week < 5

    @property
    def is_weekend(self) -> bool:
        return not self.is_weekday

    def reset(self) -> None:
        """Reset the simulation clock back to the original start."""
        self._dt = self._start
        self.tick = 0.0

    def set_datetime(self, new_datetime: datetime) -> None:
        """Force the simulation to a specific datetime."""
        self._dt = new_datetime.replace(second=0, microsecond=0)

    def increment(
        self,
        minutes: float | None = None,
        *,
        hours: float | None = None,
        days: float | None = None,
        delta: timedelta | None = None,
    ) -> datetime:
        """Advance the simulation time.

        Args:
            minutes: Number of minutes to advance. When omitted, defaults to 60.
            hours: Additional hours to advance.
            days: Additional days to advance.
            delta: Optional timedelta that overrides the other arguments.

        Returns:
            datetime: The updated simulation timestamp.
        """
        advance = delta
        if advance is None:
            total_minutes = 0.0
            if minutes is not None:
                total_minutes += minutes
            if hours is not None:
                total_minutes += hours * 60
            if days is not None:
                total_minutes += days * 24 * 60
            if total_minutes == 0:
                total_minutes = 60.0
            advance = timedelta(minutes=total_minutes)

        self._dt += advance
        self.tick += advance.total_seconds() / 60.0
        return self._dt

    def __repr__(self) -> str:
        return f"SimTime(datetime={self._dt.isoformat(timespec='minutes')})"
