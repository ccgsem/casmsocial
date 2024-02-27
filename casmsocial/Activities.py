from __future__ import annotations
from pydantic import BaseModel
import pyarrow.parquet as pq

#from .Place import Place

class Act(BaseModel):
    """Act Class
    
    Names a place that a person will go to a particular time
    """
    person_id: int
    activity_id: int
    activity_sequence: int
    starttime_min: int
    endtime_min: int

    def __init__(
            self,
            person_id: int,
            activity_id: int,
            activity_sequence: int,
            starttime_min: int,
            endtime_min: int) -> None:
        super().__init__(
            person_id=person_id,
            activity_id=activity_id,
            activity_sequence=activity_sequence,
            starttime_min=starttime_min,
            endtime_min=endtime_min)
        
    def contains(self, time: float) -> bool:
        """Return True if the time is within the start and end times of the activity."""
        return self.starttime_min <= time and time <= self.endtime_min


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
        return tuple(self.activities)
   
    @classmethod
    def restore(cls, data: tuple[Act]) -> Activities:
        """Create an  object from the data created in the data() function.

        Returns:
            A new Activities object.
        """
        return cls(data)
    
class ActivitiesSuperset(object):
    """Activities Set Class
    
    A set of collections of activities for a person.
    """

    def __init__(self, activities_superset: tuple[Activities]) -> None:
        self.__activities_superset = list(activities_superset)

    def __len__(self) -> int:
        return len(self.__activities_superset)
    
    def __getitem__(self, idx: int) -> Activities:
        return self.__activities_superset[idx]

    @property
    def activities_superset(self) -> list[Activities]:
        return self.__activities_superset

    def addActivities(self, activities: Activities) -> None:
        self.__activities_superset.append(activities)

    def data(self) -> tuple:
        """Get the data for an activities set in a tuple.

        Returns:
            The activities set data as a tuple. 
        """
        return self.__activities_superset

    @classmethod
    def restore(cls, data: tuple[Activities]) -> ActivitiesSuperset:
        """Create an  object from the data created in the data() function.

        Returns:
            A new ActivitiesSuperset object.
        """
        return cls(data)
    