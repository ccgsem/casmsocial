from __future__ import annotations

from dataclasses import astuple, dataclass, field


@dataclass
class Act:
    """Act Class

    Names a place that a person will go to a particular time
    """

    person_id: int
    activity_id: int
    activity_sequence: int
    starttime_min: int
    endtime_min: int

    def contains(self, time: float) -> bool:
        """Return True if the time is within the start and end times of the activity."""
        return self.starttime_min <= time and time <= self.endtime_min


@dataclass
class Activities:
    """Activities Class

    A collection of activities for a person.
    """

    activities_id: int
    name: str
    acts: tuple[Act]

    def addAct(self, act: Act) -> None:
        """Add an activity to the activities list."""
        acts_list = list(self.acts)
        acts_list.append(act)
        self.acts = tuple(acts_list)

    def activityAt(self, time: float) -> Act:
        """Find the activity at a particular time."""
        next_act = None
        for act in self.acts:
            if act.contains(time):
                return act

        return next_act

    def reset(self) -> None:
        """Reset the activities list."""
        self.acts = ()

    def data(self) -> tuple:
        """Get the data for activities in a tuple.

        Returns:
            The activities data as a tuple.
        """
        return (self.activities_id, self.name, tuple([astuple(a) for a in self.acts]))

    @classmethod
    def restore(cls, data: tuple[tuple[int]]) -> Activities:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Activities object.
        """
        return cls(data[0], data[1], tuple([Act(*act) for act in list(data[2])]))


@dataclass
class Schedules:
    """Schedules Class

    A collection of schedules for a person.
    """

    __schedules: list[Activities] = field(default_factory=lambda: [])

    # def __init__(self, schedules: tuple[Activities]) -> None:
    #    self.__schedules = list(schedules)

    def __len__(self) -> int:
        return len(self.__schedules)

    def __getitem__(self, idx: int) -> Activities:
        return self.__schedules[idx]

    @property
    def schedules(self) -> list[Activities]:
        return self.__schedules

    def addActivities(self, activities: Activities) -> None:
        self.__schedules.append(activities)

    def data(self) -> tuple:
        """Get the data for an activities set in a tuple.

        Returns:
            The activities set data as a tuple.
        """
        return tuple([activities.data() for activities in self.__schedules])

    @classmethod
    def restore(cls, data: tuple[tuple[tuple[int]]]) -> Schedules:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Schedules object.
        """
        return cls(tuple([Activities.restore(activities) for activities in list(data)]))
