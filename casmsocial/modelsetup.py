from .person import Person, person_cache
from .place import Place
from .household import Household
from .school import School
from .work import Work
from .activities import Act, Activities, Schedules

from typing import Dict
from csv import DictReader
import pathlib
import pyarrow.parquet as pq

from repast4py.space import DiscretePoint as dpt


def pointInBounds(point, bounds):
    xInBounds = point.x >= bounds.xmin and point.x < (bounds.xmin + bounds.xextent)
    yInBounds = point.y >= bounds.ymin and point.y < (bounds.ymin + bounds.yextent)
    zInBounds = point.z == 0 or (point.z >= bounds.zmin and point.z < (bounds.zmin + bounds.zextent))

    return xInBounds and yInBounds and zInBounds


class ModelSetup:
    """This class is responsible for setting up the model. It reads in the"""

    @staticmethod
    def initPersons(
        personFile: pathlib.Path,
        placeMap: Dict,
        activitiesMap: Dict,
        thisRank: int,
        context,
        grid,
        rng) -> dict[int,int]:

        agentIdMap = {}

        # with open(personFile, 'r', newline='') as f:
        #     persons = DictReader(f)
        table = pq.read_table(personFile)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                row = [x.as_py() for x in row]  # convert arrow scalars to python
                p = dict(zip(table.column_names, row))

                personID = p['sp_id']
                hhId = p['sp_hh_id']
                rank = placeMap[hhId].rank

                if rank != thisRank:
                    continue

                startingLocation = placeMap[hhId].location

                places = [
                    p['sp_hh_id'],
                    p['sp_work_id'],
                    p['sp_school_id']
                ]

                schedule = activitiesMap[personID]
                print(f'personID={personID}, schedule={schedule}')
                activities = Activities(personID, tuple(schedule))
                schedules = Schedules(())
                schedules.addActivities(activities)

                startingRisk = rng.random()
                
                person = Person(
                    personID,
                    rank,
                    schedules,
                    places,
                    startingLocation,
                    startingRisk)
                person_cache[person.uid] = person
                agentIdMap[personID] = person.uid
                context.add(person)
                grid.move(person, startingLocation)
        
        return agentIdMap

    @staticmethod
    def initPlacesFromFile(
        rank: int,
        placeType: str,
        placeFile: pathlib.Path,
        placeMap: dict[int, Place],
        localPlaces,
        grid) -> tuple([dict[int, Place], list[int]]):

        table = pq.read_table(placeFile)

        # with open(placeFile, 'r', newline='') as f:
        #     places = DictReader(f)
        #     for p in places:
        for batch in table.to_batches():
            for row in zip(*batch.columns):
                row = [x.as_py() for x in row]  # convert arrow scalars to python
                p = dict(zip(table.column_names, row))
                
                placeId = p['sp_id']

                # `location` is currently referenced required but not used
                if 'x' not in p:
                    p['x'] = 0
                if 'y' not in p:
                    p['y'] = 0
                location = dpt(x=int(p['x']), y=int(p['y']), z=0)

                place = None
                match placeType:
                    case 'household':
                        place = Household(p)
                    case 'work':
                        place = Work(placeId, location)
                    case  'school':
                        place = School(placeId, location)
                    case _:
                        print(f'Error: Bad placetype during place initialization: {placeType}')
                        place = Place(placeId, location)

                placeMap[placeId] = place

                localBounds = grid.get_local_bounds()
                if pointInBounds(location, localBounds):
                    place.rank = rank
                    localPlaces.append(place)

        return placeMap, localPlaces

    @staticmethod
    def initPlaces(
        rank: int,
        householdFile: str,
        schoolFile: str,
        workFile: str,
        grid) -> tuple([dict[int, Place], list[int]]):

        placeMap = {}
        localPlaces = []

        placeMap, localPlaces = ModelSetup.initPlacesFromFile(rank, 'household', householdFile, placeMap, localPlaces, grid)
        placeMap, localPlaces = ModelSetup.initPlacesFromFile(rank, 'work', workFile, placeMap, localPlaces, grid)
        placeMap, localPlaces = ModelSetup.initPlacesFromFile(rank, 'school', schoolFile, placeMap, localPlaces, grid)

        return placeMap, localPlaces

    @staticmethod
    def initActivities(
            activitiesFile: pathlib.Path
        ) -> dict[int, list[int]]:
        # activitiesMap looks like:
        # personID -> Activities object
        act_map = {}

        # This should be the most eficient way to extract the data via pyarrow
        # See https://stackoverflow.com/questions/53157495/fastest-way-to-iterate-pyarrow-table/55633193#55633193
        table = pq.read_table(activitiesFile)
        
        for batch in table.to_batches():
            # for row in zip(*batch.columns):
            #     print(row)
            d = batch.to_pydict()
            for sp_persons_id, activity_id, activity_seq, start, end in zip(d['sp_persons_id'], d['activity_id'], d['activity_sequence'], d['starttime_min'], d['endtime_min']):
                if sp_persons_id not in act_map:
                    act_map[sp_persons_id] = [Act(sp_persons_id, activity_id, activity_seq, start, end)]
                else:
                    act_map[sp_persons_id].append(Act(sp_persons_id, activity_id, activity_seq, start, end))

        return act_map

    @staticmethod
    def initContacts(
        contactFile: pathlib.Path
        ) -> dict[int,dict[int,int]]:

        # contactMap looks like:
        # personID -> { hour_of_day -> [ otherPersonIDs ] }
        # dict
        contactMap = {}

        # with open(contactFile, 'r', newline='') as f:
        #     contacts = DictReader(f)
        table = pq.read_table(contactFile)

        for batch in table.to_batches():
            # for row in zip(*batch.columns):
            #     print(row)
            d = batch.to_pydict()
            for source, target, hour_of_the_day in zip(d['from_person'], d['to_person'], d['hour']):
                if source not in contactMap:
                    contactMap[source] = {}

                if hour_of_the_day not in contactMap[source]:
                    contactMap[source][hour_of_the_day] = []

                contactMap[source][hour_of_the_day].append(target)

        return contactMap