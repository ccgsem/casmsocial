""""""

from datetime import datetime, timedelta


def get_closest_monday(current_date):
    """
    Get the closest Monday to the given date.
    If the current date is a Monday, return that date.
    If the current date is before the next Monday, return the next Monday.
    If the current date is after the last Monday, return the last Monday.
    Args:
        current_date (date): The date to check.
    Returns:
        date: The closest Monday to the given date.
    """
    today = current_date
    today_weekday = today.weekday()

    # Monday is 0, Sunday is 6
    days_until_next_monday = (7 - today_weekday) % 7

    # Calculate the date of the next monday
    next_monday = today + timedelta(days=days_until_next_monday)

    # Calculate the date of the previous monday
    days_since_last_monday = today_weekday
    last_monday = today - timedelta(days=days_since_last_monday)

    # Determine which Monday is closer
    if (next_monday - today) < (today - last_monday):
        return next_monday
    else:
        return last_monday


def get_midnight(dt: datetime) -> datetime:
    """
    Get the midnight time for a given datetime.

    Args:
        dt (datetime): The datetime to convert to midnight.

    Returns:
        datetime: A new datetime object with the time set to midnight.
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)
