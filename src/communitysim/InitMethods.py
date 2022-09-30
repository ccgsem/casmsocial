from Human import Human
from Household import Household
from School import School
from Work import Work

from repast4py.space import DiscretePoint as dpt

def initHumans(personFile: str, placeMap: Dict, scheduleMap: Dict, thisRank: int, context, grid):

    with open(personFile, 'r', newline='') as f:
        persons = DictReader(f)
        for p in persons:
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

            schedule = scheduleMap[personID]
            
            human = Human(personID, rank, schedule, places, startingLocation)
            human_cache[human.uid] = human
            context.add(human)
            grid.move(human, startingLocation)

def pointInBounds(point, bounds):
    xInBounds = point.x >= bounds.xmin and point.x < (bounds.xmin + bounds.xextent)
    yInBounds = point.y >= bounds.ymin and point.y < (bounds.ymin + bounds.yextent)
    zInBounds = point.z >= bounds.zmin and point.z < (bounds.zmin + bounds.zextent)

    return xInBounds and yInBounds and zInBounds

def initPlaces(rank: int, householdFile: str, schoolFile: str, workFile: str, grid):
    placeMap = {}
    localPlaces = []

    with open(placeFile, 'r', newline='') as f:
        places = DictReader(f)
        for p in places:
            placeId = p['sp_id']
            location = dpt(x=p['x'], y=p['y'], z=0)
            place = Place(placeId, location)

            placeMap[placeId] = place

            localBounds = grid.get_local_bounds()
            if pointInBounds(location, localBounds):
                localPlaces.append(place)

    return placeMap, localPlaces

def initSchedules(scheduleFile: str):
    # scheduleMap looks like:
    # personID -> Schedule object
    scheduleMap = {}
    with open(scheduleFile, 'r', newline='') as f:
        activities = DictReader(f)
        for a in activities:
            scheduleMap[a['sp_persons_id']] = Schedule(
                [int(activity) for activity in a['activity_ids'].split(':')]
            )

    return scheduleMap

def initContacts(contactFile: str):
    # contactMap looks like:
    # personID -> { step -> [ otherPersonIDs ] }
    contactMap = {}

    with open(contactFile, 'r', newline='') as f:
        contacts = DictReader(f)
        for contact in contacts:
            source = contact['from_person']
            target = contact['to_person']
            step = int(contact['step'])

            if source not in contactMap:
                contactMap[source] = {}

            if step not in contactMap[source]:
                contactMap[source][step] = []

            contactMap[source][step].append(target)

    return contactMap