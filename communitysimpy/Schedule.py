""" Schedule class used by Agents """
from typing import Tuple

class Schedule:
    def __init__(self, repeat: int, activities: [int]):
        self.repeat = repeat
        self.activities = activities
        assert(len(self.activities) == self.repeat)

    def activityAt(self, tick: int) -> int:
        bounded_tick = tick % self.repeat
        return self.activities[bounded_tick]

    def data(self) -> Tuple:
        """Get the data for the schedule in a tuple.

        Returns:
            The schedule data as a tuple. 
        """
        return (self.repeat, self.activities)

def restoreSchedule(schedule_data: Tuple) -> Schedule:
    """Create a Schedule object from the data created in the data() function.

    Returns:
        A new Schedule object.
    """
    return Schedule(schedule_data[0], schedule_data[1])