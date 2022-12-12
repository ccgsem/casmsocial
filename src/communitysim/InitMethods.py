from .Person import Person, person_cache
from .Place import Place
from .Household import Household
from .School import School
from .Work import Work
from .Schedule import Schedule

from typing import Dict
from csv import DictReader
import pathlib
import pyarrow.parquet as pq

from repast4py.space import DiscretePoint as dpt

def initPersons(personFile: str, placeMap: Dict, scheduleMap: Dict, thisRank: int, context, grid, rng):
    agentIdMap = {}

    with open(personFile, 'r', newline='') as f:
        persons = DictReader(f)
        for p in persons:
            personID = int(p['sp_id'])
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

            schedule = scheduleMap[personID]

            startingRisk = rng.random()
            
            person = Person(personID, rank, schedule, places, startingLocation, startingRisk)
            person_cache[person.uid] = person
            agentIdMap[personID] = person.uid
            context.add(person)
            grid.move(person, startingLocation)
    return agentIdMap

def pointInBounds(point, bounds):
    xInBounds = point.x >= bounds.xmin and point.x < (bounds.xmin + bounds.xextent)
    yInBounds = point.y >= bounds.ymin and point.y < (bounds.ymin + bounds.yextent)
    zInBounds = point.z == 0 or (point.z >= bounds.zmin and point.z < (bounds.zmin + bounds.zextent))

    return xInBounds and yInBounds and zInBounds

def initPlacesFromFile(rank: int, placeType: str, placeFile: str, placeMap, localPlaces, grid):
    with open(placeFile, 'r', newline='') as f:
        places = DictReader(f)
        for p in places:
            placeId = p['sp_id']
            location = dpt(x=int(p['x']), y=int(p['y']), z=0)
            place = None
            if placeType == 'household':
                place = Household(p)
            elif placeType == 'work':
                place = Work(placeId, location)
            elif placeType == 'school':
                place = School(placeId, location)
            else:
                print(f'Error: Bad placetype during place initialization: {placeType}')
                place = Place(placeId, location)

            placeMap[placeId] = place

            localBounds = grid.get_local_bounds()
            if pointInBounds(location, localBounds):
                place.rank = rank
                localPlaces.append(place)

    return placeMap, localPlaces

def initPlaces(rank: int, householdFile: str, schoolFile: str, workFile: str, grid):
    placeMap = {}
    localPlaces = []

    placeMap, localPlaces = initPlacesFromFile(rank, 'household', householdFile, placeMap, localPlaces, grid)
    placeMap, localPlaces = initPlacesFromFile(rank, 'work', workFile, placeMap, localPlaces, grid)
    placeMap, localPlaces = initPlacesFromFile(rank, 'school', schoolFile, placeMap, localPlaces, grid)

    return placeMap, localPlaces

def initSchedules(scheduleFile: pathlib.Path):
    # scheduleMap looks like:
    # personID -> Schedule object
    scheduleMap = {}

    # This should be the most eficient way to extract the data via pyarrow
    # See 
    table = pq.read_table(scheduleFile)

    for batch in table.to_batches():
        # for row in zip(*batch.columns):
        #     print(row)
        d = batch.to_pydict()
        for sp_persons_id, activity_ids in zip(d['sp_persons_id'], d['activity_ids']):
            scheduleMap[sp_persons_id] = Schedule(
                [int(activity) for activity in activity_ids.split(':')]
            )

    return scheduleMap

def initContacts(contactFile: str):
    # contactMap looks like:
    # personID -> { step -> [ otherPersonIDs ] }
    contactMap = {}

    with open(contactFile, 'r', newline='') as f:
        contacts = DictReader(f)
        for contact in contacts:
            source = int(contact['from_person'])
            target = int(contact['to_person'])
            step = int(contact['step'])

            if source not in contactMap:
                contactMap[source] = {}

            if step not in contactMap[source]:
                contactMap[source][step] = []

            contactMap[source][step].append(target)

    return contactMap