""" Schedule class used by Agents """
from typing import Tuple

"""
    This class represents a cyclical schedule.
    'activities' is a list of the place that an agent will go to at each step.
    e.g. If there are two elements in 'activities', an agent will go to these places:
        - step 0 -> activities[0]
        - step 1 -> activities[1]
        - step 2 -> activities[2]
        ...
"""
class Schedule(object):
    def __init__(self, activities):
        self.repeat = len(activities)
        self.activities = activities

    def activityAt(self, tick: int) -> int:
        bounded_tick = tick % self.repeat
        return self.activities[bounded_tick]

    def data(self) -> Tuple:
        """Get the data for the schedule in a tuple.

        Returns:
            The schedule data as a tuple. 
        """
        return (self.activities)

def restoreSchedule(schedule_data: Tuple) -> Schedule:
    """Create a Schedule object from the data created in the data() function.

    Returns:
        A new Schedule object.
    """
    return Schedule(schedule_data)