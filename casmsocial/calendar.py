
class Calendar(object):
    """Calendar Class"""
    def __init__(self) -> None:
        self.__minute_of_day = 0
        self.__hour_of_day = 0
        self.__day_of_week = 0
        self.__day_of_year = 1
        self.__year = 1
        self.tick = 0.0

    @property
    def minute_of_day(self):
        return self.__minute_of_day
    
    @property
    def hour_of_day(self):
        return self.__hour_of_day
    
    @property
    def day_of_week(self):
        return self.__day_of_week
    
    @property
    def day_of_year(self):
        return self.__day_of_year
    
    @property
    def year(self):
        return self.__year
    
    def increment(self) -> None:
        """Increment the calendar by one hour."""
        self.__hour_of_day += 1
        self.__minute_of_day += 60
        if self.__hour_of_day > 23:
            self.__hour_of_day = 0
            self.__minute_of_day = 0
            self.__day_of_week += 1
            if self.__day_of_week > 6:
                self.__day_of_week = 0
            self.__day_of_week += 1
            if self.__day_of_week > 365:
                self.__day_of_year = 1
                self.__year += 1

    def is_weekday(self) -> bool:
        """Return True if it is a weekday."""
        return not (self.__day_of_week == 0 or self.__day_of_week == 6)

