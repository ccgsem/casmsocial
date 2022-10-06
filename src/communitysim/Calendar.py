
class Calendar:
    def __init__(self, steps_per_day: int):
        self.steps_per_day = steps_per_day
        self.step = 0
        self.day = 0
        self.week = 0
        self.month = 0
        self.year = 0
        self.isNewDay = False
        self.isNewWeek = False
        self.isNewMonth = False
        self.isNewYear = False

    def calendarStep(self):
        self.isNewDay = False
        self.isNewWeek = False
        self.isNewMonth = False
        self.isNewYear = False

        self.step += 1
        if self.step % self.steps_per_day == 0:
            self.day += 1
            self.isNewDay = True
            if self.day % 7 == 0:
                self.week += 1
                self.isNewWeek = True
            if self.day % 30 == 0:
                self.month += 1
                self.isNewMonth = True
            if self.day % 365 == 0:
                self.year += 1
                self.isNewYear = True

    def day(self) -> int:
        return self.day

    def week(self) -> int:
        return self.week

    def month(self) -> int:
        return self.month

    def year(self) -> int:
        return self.year

    def dateString(self) -> str:
        partOfDay = self.step % self.steps_per_day

        return f"Day {self.day}.{partOfDay}"