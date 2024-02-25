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


class Activities(object):
    """Activities Class"""
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

    def findActAt(self, time: float) -> Act:
        pass

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def activities(self) -> list[Act]:
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
            A new Schedule object.
        """
        return cls(data)


class ActivityCreator(object):
    """Activity Creator Class"""

    def __init__(self, act_filename: str):
        self.act_filename = act_filename

    def run(self: object, map: dict) -> None:
        """Run method
        
        Loads the activity file and creates the activity map
        """
        # This should be the most eficient way to extract the data via pyarrow
        # See https://stackoverflow.com/questions/53157495/fastest-way-to-iterate-pyarrow-table/55633193#55633193
        table = pq.read_table(self.act_filename)

        pass
    