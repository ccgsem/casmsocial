
class Calendar(object):
    def __init__(self):
        self.minute_of_day = 0
        self.hour_of_day = 0
        self.day_of_week = 0
        self.day_of_year = 1
        self.year = 1
        self.tick = 0.0

    def increment(self):
        self.hour_of_day += 1
        self.minute_of_day += 60
        if self.hour_of_day > 23:
            self.hour_of_day = 0
            self.minute_of_day = 0
            self.day_of_week += 1
            if self.day_of_week > 6:
                self.day_of_week = 0
            self.day_of_year += 1
            if self.day_of_year > 365:
                self.day_of_year = 1
                self.year += 1

    def is_weekday(self):
        return not (self.day_of_week == 0 or self.day_of_week == 6)

