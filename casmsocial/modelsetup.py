from casmsocial.person import (
    Person,
    person_cache
)
from casmsocial.place import (
    Place,
    Places
)
from casmsocial.activities import (
    Act,
    Activities,
    Schedules
)

from typing import (
    Dict,
    Tuple
)
import pathlib
import pyarrow.parquet as pq


def pointInBounds(point, bounds):
    xInBounds = point.x >= bounds.xmin and \
        point.x < (bounds.xmin + bounds.xextent)
    yInBounds = point.y >= bounds.ymin and \
        point.y < (bounds.ymin + bounds.yextent)
    zInBounds = point.z == 0 or \
        (point.z >= bounds.zmin and point.z < (bounds.zmin + bounds.zextent))

    return xInBounds and yInBounds and zInBounds


class ModelSetup:
    """This class is responsible for setting up the model. It reads in the"""

    @staticmethod
    def initPersons(
        personFile: pathlib.Path,
        placeMap: Dict,
        activitiesMap: Dict,
        person_places: list[str],
        thisRank: int,
        context,
        cspace,
        rng
    ) -> dict[int, int]:

        agentIdMap = {}

        table = pq.read_table(personFile)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                p = dict(zip(table.column_names, row))

                personID = p['sp_id']

                # TODO: add tests for this
                #  - places = [ p[x] for x in person_places ]
                #  - all places should be in placeMap
                #  - the first place is a household
                #  - how to handle the case where the person is not on this rank?
                #  - how to handle the case where the person is not in the activitiesMap?
                places = [ p[x] for x in person_places ]

                hhId = places[0]  # p['sp_hh_id']
                if (hhId not in placeMap):
                    print(f'Error: No place found for {p}')

                rank = placeMap[hhId].rank

                if rank != thisRank:
                    continue

                startingLocation = placeMap[hhId].location

                schedule = activitiesMap[personID]
                # print(f'personID={personID}, schedule={schedule}')
                activities = Activities(personID, tuple(schedule))
                schedules = Schedules(())
                schedules.addActivities(activities)

                # Person
                #  - places: list[int]
                #  - schedules: Schedules

                person = Person(
                    personID,
                    rank,
                    schedules,
                    places,
                    startingLocation,
                    p  # initDict for additional data
                )
                person_cache[person.uid] = person
                agentIdMap[personID] = person.uid
                context.add(person)
                cspace.move(person, startingLocation)
                # print(person.places)

        return agentIdMap

    @staticmethod
    def initPlacesFromFile(
        rank: int,
        placeTypeIndex: int,
        placeFile: pathlib.Path,
        placeMap: dict[int, Place],
        localPlaces,
        cspace
    ) -> Tuple[Dict[int, Place], list[int]]:

        # get the place type
        placeConfig = Places.get_place_config(placeTypeIndex)
        placeType = placeConfig.type
        placeDataType = placeConfig.dataType

        # load the places from the file
        table = pq.read_table(placeFile)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                place_record = dict(zip(table.column_names, row))

                placeId = place_record['sp_id']
                place = placeType(place_record, placeDataType)
                placeMap[placeId] = place

                localBounds = cspace.get_local_bounds()
                if pointInBounds(place.location, localBounds):
                    place.rank = rank
                    localPlaces.append(place)

        return placeMap, localPlaces

    @staticmethod
    def initPlaces(
        rank: int,
        place_files: list[pathlib.Path],
        cspace
    ) -> Tuple[Dict[int, Place], list[int]]:

        placeMap: map = {}
        localPlaces: list = []

        for placeTypeIndex, placeFile in enumerate(place_files):
            placeMap, localPlaces = \
                ModelSetup.initPlacesFromFile(
                    rank,
                    placeTypeIndex,
                    placeFile,
                    placeMap,
                    localPlaces,
                    cspace
                )

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
            for sp_persons_id, activity_id, activity_seq, start, end in \
                zip(
                    d['sp_persons_id'],
                    d['activity_id'],
                    d['activity_sequence'],
                    d['starttime_min'],
                    d['endtime_min']):

                if sp_persons_id not in act_map:
                    act_map[sp_persons_id] = \
                        [
                            Act(
                                sp_persons_id,
                                activity_id,
                                activity_seq,
                                start,
                                end
                            )
                        ]
                else:
                    act_map[sp_persons_id].append(
                        Act(
                            sp_persons_id,
                            activity_id,
                            activity_seq,
                            start,
                            end
                        )
                    )

        return act_map

    @staticmethod
    def initContacts(
        contactFile: pathlib.Path
    ) -> dict[int, dict[int, int]]:

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
            for source, target, hour_of_the_day in \
                zip(
                    d['from_person'],
                    d['to_person'],
                    d['hour']):

                if source not in contactMap:
                    contactMap[source] = {}

                if hour_of_the_day not in contactMap[source]:
                    contactMap[source][hour_of_the_day] = []

                contactMap[source][hour_of_the_day].append(target)

        return contactMap
