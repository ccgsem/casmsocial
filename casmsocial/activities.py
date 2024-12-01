from __future__ import annotations
from pydantic import BaseModel
import pyarrow.parquet as pq

#from .Place import Place


#class Act(BaseModel):
class Act(object):
    """Act Class
    
    Names a place that a person will go to a particular time
    """
    # person_id: int
    # activity_id: int
    # activity_sequence: int
    # starttime_min: int
    # endtime_min: int

    def __init__(
            self,
            person_id: int,
            activity_id: int,
            activity_sequence: int,
            starttime_min: int,
            endtime_min: int) -> None:
        #     super().__init__(
        #         person_id=person_id,
        #         activity_id=activity_id,
        #         activity_sequence=activity_sequence,
        #         starttime_min=starttime_min,
        #         endtime_min=endtime_min)
        self.person_id = person_id
        self.activity_id = activity_id
        self.activity_sequence = activity_sequence
        self.starttime_min = starttime_min
        self.endtime_min = endtime_min
        
    def contains(self, time: float) -> bool:
        """Return True if the time is within the start and end times of the activity."""
        return self.starttime_min <= time and time <= self.endtime_min
    
    def data(self) -> tuple:
        """Get the data for an activity in a tuple.

        Returns:
            The activity data as a tuple. 
        """
        return (
            self.person_id,
            self.activity_id,
            self.activity_sequence,
            self.starttime_min,
            self.endtime_min
            )
    
    @classmethod
    def restore(cls, data: tuple[int]) -> Act:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Act object.
        """
        #return cls(*list(data))
        return cls(
            person_id=data[0],
            activity_id=data[1],
            activity_sequence=data[2],
            starttime_min=data[3],
            endtime_min=data[-1]
            )


class Activities(object):
    """Activities Class

    A collection of activities for a person.
    """
    # __id: int
    # __acts: tuple[Act]

    def __init__(
            self,
            id: int,
            acts: tuple[Act] = ()
            ) -> None:
        self.__id = id
        self.__acts = list(acts)

    def addAct(self, act: Act) -> None:
        self.__acts.append(act)

    def activityAt(self, time: float) -> Act:
        """Find the activity at a particular time.
        """
        next_act = None
        for act in self.__acts:
            if act.contains(time):
                return act
            
        return next_act

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def acts(self) -> list[Act]:
        return self.__acts

    def data(self) -> tuple:
        """Get the data for activities in a tuple.

        Returns:
            The activities data as a tuple. 
        """
        return (
            self.__id,
            tuple([a.data() for a in self.__acts])
        )
   
    @classmethod
    def restore(cls, data: tuple[tuple[int]]) -> Activities:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Activities object.
        """
        return cls(
            data[0],
            tuple([Act.restore(act) for act in list(data[1])])
        )
    

class Schedules(object):
    """Schedules Class
    
    A collection of schedules for a person.
    """

    def __init__(self, schedules: tuple[Activities]) -> None:
        self.__schedules = list(schedules)

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
        return tuple(
            [activities.data() for activities in self.__schedules]
        )

    @classmethod
    def restore(cls, data: tuple[tuple[tuple[int]]]) -> Schedules:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Schedules object.
        """
        return cls(
            tuple(
                [Activities.restore(activities) for activities in list(data)]
                )
        )
    