"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the SIModel
"""
import os
import pathlib
import time
from collections import namedtuple
from typing import ClassVar

import pyarrow.parquet as pq
import repast4py
from dotenv import find_dotenv, load_dotenv
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import schedule

from casmsocial.activities import Act, Activities, Schedules
from casmsocial.calendar import Calendar
from casmsocial.datautility import convert_to_int

# note: place types are set by derived Model classes
from casmsocial.factory import Models
from casmsocial.message import Message
from casmsocial.model import Model
from casmsocial.person import Person, PersonConfig, person_cache
from casmsocial.place import PlaceConfig, PlacesProjection


class SIModel(Model):
    """
    The SIModel class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    The SIModel class is a subclass of the Model class, which is an abstract
    base class that defines the interface for all models in the casmsocial.
    The SIModel class implements the start and step methods, which are called
    by the run function in the casmsocial module to start and run the model.

    The SIModel class adds the following functionality to the Model class:

    - The SIModel class initializes geographic places and agents.
    - The SIModel class updates the  environment for the current time step.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    # class variables

    # list of places configurations
    __placeConfigs: ClassVar[list[PlaceConfig]] = []

    # remote place configuration
    __remote_place_config: PlaceConfig = None

    # person configuration
    __person_config: PersonConfig = None

    # list of planned activities (column names in the person file for activities)
    __planned_activity_names: ClassVar[list[str]] = []

    # list of activities
    __activity_names: ClassVar[list[str]] = []

    # activites data type: namedtuple
    __activities_data_type: namedtuple = None

    # class methods
    @classmethod
    def register_place_config(cls, config: PlaceConfig) -> None:
        """Register a place configuration."""
        cls.__placeConfigs.append(config)

    @classmethod
    def get_place_configs(cls) -> list[PlaceConfig]:
        """Get the list of place configurations."""
        return cls.__placeConfigs

    @classmethod
    def get_place_config(cls, idx: int) -> PlaceConfig:
        """Get a PlacesConfig from the list of configs."""
        return cls.__placeConfigs[idx]

    @classmethod
    def get_place_config_idx(cls, name: str) -> int:
        """Get the index of a PlacesConfig in the list of configs."""
        for idx, config in enumerate(cls.__placeConfigs):
            if config.name == name:
                return idx
        return -1

    @classmethod
    def get_place_config_name(cls, idx: int) -> str:
        """Get the name of a PlacesConfig in the list of configs."""
        return cls.__placeConfigs[idx].name

    @classmethod
    def get_all_place_config_names(cls) -> list[str]:
        """Get the names of all PlacesConfig in the list of configs."""
        return [config.name for config in cls.__placeConfigs]

    @classmethod
    def register_remote_place_config(cls, config: PlaceConfig) -> None:
        """Register a remote place configuration."""
        cls.__remote_place_config = config

    @classmethod
    def get_remote_place_config(cls) -> PlaceConfig:
        """Get the remote place configuration."""
        return cls.__remote_place_config

    @classmethod
    def register_person_config(cls, config: PersonConfig) -> None:
        """Register a person configuration."""
        Person.registerPersonDataClass(config.dataType)
        cls.__person_config = config

    @classmethod
    def get_person_config(cls) -> PersonConfig:
        """Get the person configuration."""
        return cls.__person_config

    @classmethod
    def register_planned_activity_names(cls, planned_activity_names: list[str]) -> None:
        """Register planned activities."""
        cls.__planned_activity_names = planned_activity_names

    @classmethod
    def get_planned_activity_names(cls) -> list[str]:
        """Get the planned activities."""
        return cls.__planned_activity_names

    @classmethod
    def register_activity_names(cls, activity_names: list[str]) -> None:
        """Register alternate activities."""
        cls.__activity_names = activity_names

    @classmethod
    def get_activity_names(cls) -> list[str]:
        """Get all activities."""
        return cls.__activity_names

    @classmethod
    def get_activities_data_type(cls) -> namedtuple:
        """Get the activities data type."""
        if not cls.__activities_data_type:
            cls.__activities_data_type = namedtuple(
                'ActivitiesDataclass',
                cls.get_activity_names()
            )
        return cls.__activities_data_type

    # instance variables
    def __init__(
        self,
        comm: MPI.Intracomm,
        params: dict
    ):
        """ Constructor for the SIModel class

        Args:
            comm: the mpi communicator over which the model is distributed.
            params: the simulation input parameters
        """
        Model.set_model(self)

        print("Creating SIModel...")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.params = params

        # start timer
        self.start_time = time.time()

        # create the schedule
        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_event(0, self.initializePopulation)
        self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_repeating_event(1.1, 10, self.log_agents)
        self.runner.schedule_stop(self.params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        self.steps_per_day = int(self.params['steps.per.day'])
        self.cal = Calendar()

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(self.comm)

        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        # load $CASMSOCIAL_DATA_PATH from .env
        load_dotenv(find_dotenv())
        data_input_path = os.environ.get("CASMSOCIAL_DATA_PATH")

        data_input_path = pathlib.Path.cwd() if not data_input_path else pathlib.Path(data_input_path)

        self.data_input_path = data_input_path

    def initializePopulation(self) -> None:
        """
        Initialize population

        This method initializes the population by creating the places and agents
        from the input data files.

        The method performs the following steps:"""
        # register the place types (derived classes should set place types)

        # create SharedContext consisting of all of the places in this model
        self.places_proj = PlacesProjection("places_projection", self.comm)

        # initialize the places
        place_filenames = [
            self.data_input_path / filename for filename in self.params['places.files']
        ]

        # self.place_map, self.local_places = self.createPlaces(
        self.createPlaces(
            place_filenames
        )
        # print(f"size of place_map = {len(self.place_map)}")
        local_places = self.places_proj.get_local_places()
        print(f"rank {self.rank}: number of local places={len(local_places)}")

        # activitiesMap is a dict of personID->Schedule object
        activitiesMap = self.createActivities(
            self.data_input_path / self.params['activities.file']
        )

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons at each
        # place
        self.contact_map = {}
        if 'contact.file' in self.params:
            print("Loading contact file...")

            self.contact_map = self.createContacts(
                self.data_input_path / self.params['contact.file']
            )
        else:
            print("Error: contact file not specified.")

        print(F"rank {self.rank}: contacts size={len(self.contact_map)}")

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        # self.person_id_map = {}
        self.createPersons(
            self.data_input_path / self.params['persons.file'],
            activitiesMap,
            self.rng)

        # print(F"rank {self.rank}: number of person agents={len(self.context.agents())}")

        # agent_list = list(self.context.agents())
        # print(F"rank {self.rank}: number of person agents={len(agent_list)}")

        # saved = []
        # for p in self.context.agents():
        #     print(p)
        #     result = p.save()
        #     print(result)
        #     saved.append(result)
        #     if len(saved) > 0:
        #         break

        # restored = []
        # for i in saved:
        #     p = Person.restore(i)
        #     restored.append(p)
        #     print(p)

    def createPersons(
        self,
        personsFile: pathlib.Path,
        # placeMap: Dict,
        activitiesMap: dict,
        rng
    ) -> None:
        """ Create persons from the given file.

        Args:
            personsFile (pathlib.Path): The persons file.
            activitiesMap (dict): The activities map.
            rng: The random number generator.
        """
        # get the person type
        personType = self.get_person_config().type

        # get the activities data type: namedtuple to store places for activities
        activitiesDataType = self.get_activities_data_type()

        # get the planned_activity_names, which are the fields in the person file that
        # contain the place ids (e.g. 'sp_work_id', 'sp_school_id', etc.)
        planned_activity_names = self.get_planned_activity_names()

        # get the activity names (list should be at least as long as planned_activity_names)
        activity_names = self.get_activity_names()

        # get the alternate activity names (activities not in the planned activities)
        alternate_activities_names = activity_names[len(planned_activity_names):]

        # load the persons from the file
        table = pq.read_table(personsFile)

        print(table.column_names)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                p = dict(zip(table.column_names, row))

                personID = p['sp_id']

                # TODO: add tests for this
                #  - activities_data = [ p[x] for x in planned_activity_names ]
                #  - all places should be in placeMap
                #  - the first place is a household
                #  - how to handle the case where the person is not on this rank?
                #  - how to handle the case where the person is not in the activitiesMap?
                places = [ convert_to_int(p[x]) for x in planned_activity_names]

                for place in places:
                    if isinstance(place, str):
                        print(f"Error: Place {place} not found.")
                        return

                hhId = places[0]  # p['sp_hh_id']

                household = self.places_proj.lookup_place(hhId)
                if not household:
                    print(f"Error: No household found for {p}")
                    # continue

                rank = household.rank

                if rank != self.rank:
                    print(f"Error: Person {personID} tagged on rank={rank} is not on this rank.")
                    continue

                schedule = activitiesMap[personID]
                # print(f'personID={personID}, schedule={schedule}')
                activities = Activities(personID, 'weekday', tuple(schedule))
                schedules = Schedules()
                schedules.addActivities(activities)

                # Person
                #  - schedules: Schedules

                # add alternate places for alternate activities that are not
                # already in the person's schedule.  These alternate activities
                # are initially empty and may be determined during the simulation.
                for activity_name in alternate_activities_names:
                    activities = Activities(personID, activity_name, ())
                    schedules.addActivities(activities)

                # add alternate activities to the person's places
                alternate_places = [None] * len(alternate_activities_names)
                places = places + alternate_places

                # place is converted to a namedtuple for readability
                places = activitiesDataType(*places)

                person = personType(
                    personID,
                    rank,
                    schedules,
                    places,
                    p  # initDict for additional data
                )

                self.context.add(person)
                self.places_proj.add(person)
                self.places_proj.assign_agent_to_place(person, household)

    def createPlacesFromFile(
        self,
        placeTypeIndex: int,
        placesFile: pathlib.Path
    ) -> None:
        """
        Create places from the given file.

        Args:
            placeTypeIndex (int): The index of the place type.
            placesFile (pathlib.Path): The place file.
        """

        # get the place type
        placeConfig = self.get_place_config(placeTypeIndex)
        placeType = placeConfig.type
        placeDataType = placeConfig.dataType

        # load the places from the file
        table = pq.read_table(placesFile)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                place_record = dict(zip(table.column_names, row))
                if 'rank' not in place_record:
                    place_record['rank'] = 0
                place = placeType(place_record, placeDataType)
                self.places_proj.add_place(place)

    def createPlaces(
        self,
        places_files: list[pathlib.Path]
    ) -> None:
        """
        Create places from the given files.

        Args:
            places_files (list[pathlib.Path]): The list of place files.
        """
        for placeTypeIndex, placesFile in enumerate(places_files):
            self.createPlacesFromFile(
                placeTypeIndex,
                placesFile
            )

        # add a remote place
        remote_place = self.get_remote_place_config().type(
            {'sp_id': 0, 'rank': 0},
            self.get_remote_place_config().dataType
        )
        self.places_proj.add_place(remote_place)

    def createActivities(
        self,
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

    def createContacts(
        self,
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

    def movePersons(self) -> None:
        """Move all persons"""
        # to_move = []
        # next_place = Place()
        countOfBadMoves = 0

        for person in self.context.agents():
            result = person.move(self.cal, self.places_proj)
            if not result:
                countOfBadMoves += 1

        print(f"number of bad moves = {countOfBadMoves}")

    def step(self) -> None:
        """Step the model forward one time step."""
        # tick = self.runner.schedule.tick

        self.cal.increment()

        print(
            "Step on "
            f"day {self.cal.day_of_year}, "
            f"hour {self.cal.hour_of_day}, "
            f"minute {self.cal.minute_of_day}"
        )

        self.movePersons()

        self.context.synchronize(Person.restore)

        # 2025-02-26 jcline: this is a hack to get the person_id_map
        # self.get_local_ids()

        # 2025-02-26 jcline: this is no longer needed due to places_projection?
        # self.add_people_to_places()

        # self.make_contacts(tick)

        self.update_environment()

        # self.send_messages_between_agents()

        for person in self.context.agents():
            person.step()

        self.log_agents()

        # for person in self.context.agents():
        #     person.count_colocations(self.cspace)

        # self.data_set.log(tick)
        # clear the meet log counts for the next tick
        # self.meet_log.max_meets = \
        #     self.meet_log.min_meets = self.meet_log.total_meets = 0

    def reset(self) -> None:
        for place in self.local_places:
            place.reset()

    def get_local_ids(self) -> None:
        for person in self.context.agents():
            if person.id not in self.person_id_map:
                self.person_id_map[person.id] = person.uid

    def add_people_to_places(self) -> None:
        for person in self.context.agents():
            print(f"Adding person {person.id} to place {person.state.place_id}")
            # if person.state.place_id not in self.place_map:
            #     print(f"Person {person.id} has no place.")
            #     return
            # self.place_map[person.state.place_id].addPerson(person)

    def make_contacts(self, tick) -> None:

        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # print(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.state.place_id)
            if not contactIDs:
                # print(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.state.place_id}.")
                continue

            contacts = []
            for contactID in contactIDs:
                contacts.append(
                    self.context.agent(self.person_id_map[contactID])
                )
            person.make_contacts(contacts)

    def send_messages_between_agents(self) -> None:
        """Send messages between agents."""
        messages_to_send: list[Message] = []
        remote_person_ids = []

        # send the first round of messages
        # Step 1: Send and receive messages
        agents = self.context.agents(shuffle=True)

        for person in agents:

            import secrets
            recipient = secrets.choice(agents)

            if person.id != recipient.id:  # Avoid sending to self
                message = person.create_message(
                    recipient= recipient.id,
                    message=f"Hello from {person.uid}",
                    timestamp=(
                        "Step on "
                        f"day {self.cal.day_of_year}, "
                        f"hour {self.cal.hour_of_day}, "
                        f"minute {self.cal.minute_of_day}"
                    )
                )

            messages = person.send_messages()
            if len(messages) > 0:
                print(f"Person {person} has messages.")

                for message in messages:
                    recipient = message.recipient
                    # recipient_person = self.context.agent(self.person_id_map[recipient])
                    if recipient in self.person_id_map:  # message to local person
                        recipient_uid = self.person_id_map[recipient]
                        print(f"Message from {message.sender} to {person_cache[recipient_uid].state}:")
                        person_cache[recipient_uid].receive_message(message)
                    else:   # message to remote person
                        print(f"Message from {message.sender} to {recipient}:")

                        # get remote person ID
                        remote_person_ids.append(recipient)

                        # create message to send to other rank
                        message_to_send = message
                        message_to_send.recipients = [recipient]
                        messages_to_send.append(message_to_send)

            else:
                print(f"Person {person.id} has no messages.")
                continue

        # Exchange messages between processors
        all_messages = \
            self.exchange_messages(
                remote_person_ids,
                messages_to_send
            )

        # Step 2: Deliver messages from remote processors
        for message in all_messages:
            recipient = message.recipient
            if recipient in self.person_id_map:
                recipient_uid = self.person_id_map[recipient]
                print(
                    f"Remote message from {message.sender} to "
                    f"{person_cache[recipient_uid].state}:"
                )
                person_cache[recipient_uid].receive_message(message)
            else:
                print(f"Remote message from {message.sender} to {recipient} not delivered")

        # Step 3: Process messages
        for person in agents:
            person.process_messages()

    def get_remote_person_id_map(
            self,
            remote_person_ids: list[int]) -> dict[int, int]:
        """Get the remote person ID map."""
        remote_person_id_map = {}

        # 1. Send request for remote person IDs to other ranks
        send_buffers = [[] for _ in range(self.size)]
        for person_id in remote_person_ids:
            for rank in range(self.size):
                if rank != self.rank:
                    send_buffers[rank].append(person_id)

        # 2. Receive remote person IDs from other ranks and map and send UIDs
        received_buffers = self.comm.alltoall(send_buffers)
        send_buffers = [[] for _ in range(self.size)]  # reset send_buffers

        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for person_id in all_messages:
            if person_id in self.person_id_map:
                for rank in range(self.size):
                    if rank != self.rank:
                        person_uid = self.person_id_map[person_id]
                        msg = {'id': person_id, 'uid': person_uid}
                        send_buffers[rank].append(msg)

        # 3. Send remote person ID->UID map to other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for msg in all_messages:
            remote_person_id_map[msg['id']] = msg['uid']

        return remote_person_id_map

    def exchange_messages(
            self,
            remote_person_ids: list[int],
            messages_to_send: list[Message]
        ) -> list[Message]:
        """Exchange messages between processors using MPI."""
        remote_person_id_map = self.get_remote_person_id_map(remote_person_ids)

        # Step 1: Send messages with recipients mapped to other ranks
        send_buffers = [[] for _ in range(self.size)]
        for msg in messages_to_send:
            recipient_id = msg.recipient
            if recipient_id in remote_person_id_map:
                recipient_uid = remote_person_id_map[recipient_id]
                recipient_rank = recipient_uid.rank
            send_buffers[recipient_rank].append(msg)

        # Step 2: Receive messages from other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        return all_messages

    def update_environment(self) -> None:
        """Update the environment for the current time step."""
        pass

    def log_agents(self) -> None:
        """Log the agents at the current time step."""
        pass

    def at_end(self) -> None:
        """Actions to take at the end of the simulation."""
        pass

    def start(self) -> None:
        self.runner.execute()
        self.at_end()
        end_time = time.time()

        print(f"Simulation took {end_time - self.start_time} seconds.")


# Register SIModel
Models.add_model(
    SIModel.__module__ + '.' + SIModel.__name__,
    SIModel)


# utility functions
def update_activities_data(activities_data: namedtuple, **kwargs) -> namedtuple:
    """Update the activities data."""
    return activities_data._replace(**kwargs)
