from pydantic import BaseModel
import pyarrow.parquet as pq

from .Place import Place

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


class Activity(BaseModel):
    """Activity Class
    
    Names a place that a person will go to a particular time
    """
    __id: int
    __place: Place
    __schedule_id: int
    __start_time: float
    __end_time: float
    __activity_type:  int

    def __init__(self, record: dict):

        self.__id = id
        self.__place = place
        self.__schedule_id = schedule_id
        self.__start_time = start_time
        self.__end_time = end_time
        self.__activity_type = activity_type


class Activities(BaseModel):
    """Activities Class"""
    __id: int
    __acts: list[Act]

    def __init__(self, id: int):
        self.__id = id
        self.__acts = []

    def addAct(self, act: Act) -> None:
        self.__acts.append(act)

    def findActAt(self, time: float) -> Act:
        pass

    def id(self) -> int:
        return self.__id



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
    