"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the SIModel
"""
import os
import pathlib
import time
from collections import namedtuple
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
import pyarrow.parquet as pq
import repast4py
from dotenv import load_dotenv
from loguru import logger
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import schedule

from casmsocial.activities import Act, Activities, Schedules
from casmsocial.data_utilities import convert_to_int
from casmsocial.date_utilities import get_closest_monday, get_midnight
from casmsocial.environment import Environment
from casmsocial.factory import Models
from casmsocial.message import Message
from casmsocial.model import Model
from casmsocial.person import Person, PersonConfig, person_cache
from casmsocial.place import Place, PlaceConfig, PlacesProjection
from casmsocial.sim_time import SimTime


class MissingRequiredParameterError(Exception):
    def __init__(self, keys):
        keys_str = ", ".join(str(k) for k in keys) if isinstance(keys, (list, tuple)) else str(keys)
        super().__init__(f"Missing required parameter(s): {keys_str}")


class InvalidTimeStepError(Exception):
    def __init__(self, value):
        super().__init__(f"Invalid time step value: {value}. Time step must be an integer.")


class InvalidPlacesFilesError(Exception):
    def __init__(self, value):
        super().__init__(f"places.files must be a list of filenames, got: {value}")


class InvalidTableNameError(Exception):
    def __init__(self, table_name):
        super().__init__(f"Invalid table name: {table_name}")


# note: place types are set by derived Model classes


class SimEnvironment(Environment):
    """Sim (social interaction model) environment class.

    Creates a basic physical and social environment for the simulation.
    """

    def __init__(self, name: str):
        """Constructor for the SimEnvironment class.

        Args:
            name: The name of the environment.
        """
        super().__init__(name)

    def setup(self) -> None:
        """Set up the environment."""
        pass

    def teardown(self) -> None:
        """Tear down the environment."""
        pass

    def movePersons(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Move all persons"""
        # to_move = []
        # next_place = Place()
        countOfBadMoves = 0

        places_proj = context.get_projection("places_projection")
        for person in context.agents():
            result = person.move(cal, places_proj)
            if not result:
                countOfBadMoves += 1

        logger.debug(f"number of bad moves = {countOfBadMoves}")

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        # theModel = Model.get_model()
        # tick = self.runner.schedule.tick

        # move persons
        self.movePersons(context, cal)
        context.synchronize(Person.restore)

        # theModel.make_contacts(tick)

    def get_values_at_place(self, place: Place) -> namedtuple:
        """Get the values at the given coordinates."""
        return None


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

    # list of places configurations (deprecated)
    __placeConfigs: ClassVar[list[PlaceConfig]] = []

    # remote place configuration (deprecated)
    __remote_place_config: PlaceConfig = None

    # list of place_types names (replaces __placeConfigs)
    __place_type_names: ClassVar[list[str]] = []

    # person configuration
    __person_config: ClassVar[PersonConfig] = None

    # list of planned activities (column names in the person file for activities)
    __planned_activity_names: ClassVar[list[str]] = []

    # list of activities
    __activity_names: ClassVar[list[str]] = []

    # activites data type: namedtuple
    __activities_data_type: namedtuple = None

    # environment
    __environment: Environment = None

    # class methods

    # Register a place configuration. (deprecated)
    @classmethod
    def register_place_config(cls, config: PlaceConfig) -> None:
        """Register a place configuration."""
        cls.__placeConfigs.append(config)

    # Get the list of place configurations. (deprecated)
    @classmethod
    def get_place_configs(cls) -> list[PlaceConfig]:
        """Get the list of place configurations."""
        return cls.__placeConfigs

    # Get a specific place configuration by index. (deprecated)
    @classmethod
    def get_place_config(cls, idx: int) -> PlaceConfig:
        """Get a PlacesConfig from the list of configs."""
        return cls.__placeConfigs[idx]

    # Get a specific place configuration by name. (deprecated)
    @classmethod
    def get_place_config_idx(cls, name: str) -> int:
        """Get the index of a PlacesConfig in the list of configs."""
        for idx, config in enumerate(cls.__placeConfigs):
            if config.name == name:
                return idx
        return -1

    # Get the name of a specific place configuration by index. (deprecated)
    @classmethod
    def get_place_config_name(cls, idx: int) -> str:
        """Get the name of a PlacesConfig in the list of configs."""
        return cls.__placeConfigs[idx].name

    # Get the names of all place configurations. (deprecated)
    @classmethod
    def get_all_place_config_names(cls) -> list[str]:
        """Get the names of all PlacesConfig in the list of configs."""
        return [config.name for config in cls.__placeConfigs]

    # Register a remote place configuration. (deprecated)
    @classmethod
    def register_remote_place_config(cls, config: PlaceConfig) -> None:
        """Register a remote place configuration."""
        cls.__remote_place_config = config

    # Get the remote place configuration. (deprecated)
    @classmethod
    def get_remote_place_config(cls) -> PlaceConfig:
        """Get the remote place configuration."""
        return cls.__remote_place_config

    @classmethod
    def register_place_names(cls, place_names: list[str]) -> None:
        """Register place names.

        Args:
            place_names (list[str]): The list of place type names.
        """
        cls.__place_type_names = place_names

    @classmethod
    def get_places_names(cls) -> list[str]:
        """Get the names of all registered place types.
        Returns:
            list[str]: The list of place type names.
        """
        if not cls.__place_type_names:
            cls.__place_type_names = [config.name for config in cls.__placeConfigs]
        return cls.__place_type_names

    @classmethod
    def register_person_config(cls, config: PersonConfig) -> None:
        """Register a person configuration."""
        Person.registerPersonDataClass(config.dataType)
        Person.registerBehaviorEngine(config.behaviorEngine)
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
            cls.__activities_data_type = namedtuple("ActivitiesDataclass", cls.get_activity_names())
        return cls.__activities_data_type

    @classmethod
    def register_environment(cls, environment: Environment) -> None:
        """Register the environment."""
        cls.__environment = environment

    @classmethod
    def get_environment(cls) -> Environment:
        """Get the environment."""
        environment = cls.__environment
        if not environment:
            cls.__environment = SimEnvironment("sim_environment")
        return cls.__environment

    # instance variables
    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the SIModel class

        Args:
            comm: the mpi communicator over which the model is distributed.
            params: the simulation input parameters
        """
        Model.set_model(self)

        logger.info("Creating SIModel...")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.params = params

        # start timer
        self.start_time = time.time()

        self._validate_and_set_required_params()
        self._set_optional_params_with_defaults()
        self._remove_deprecated_params()
        self._compute_ticks()

        logger.info(f"Rank {self.rank} starting SIModel with params: {self.params}")

        # create the schedule
        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_event(0, self.initialize_population)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(self.params["ticks"])
        self.runner.schedule_end_event(self.at_end)

        # set the start datetime and timezone
        start_datetime = datetime.strptime(self.params["start.datetime"], "%Y-%m-%d %H:%M:%S")
        tz = ZoneInfo(self.params["timezone"])
        start_datetime = start_datetime.replace(tzinfo=tz)

        # initialize the simulation time
        self.cal = SimTime(start_datetime=start_datetime)

        # set the time step in minutes
        self.time_step_minutes = self.params["time.step.minutes"]

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(self.comm)

        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        load_dotenv()
        data_input_path = os.environ.get("CASMSOCIAL_DATA_PATH")
        data_input_path = pathlib.Path.cwd() if not data_input_path else pathlib.Path(data_input_path)
        self.data_input_path = data_input_path

        # create a DuckDB connection for in-memory operations
        self.conn = duckdb.connect(database=":memory:", read_only=False)
        self.queries = {
            "get_tables": "SHOW TABLES",
            "get_table_schema": "DESCRIBE {table_name}",
            "hh": "SELECT SELECT sp_id, 'Household' as place_type, latitude, longitude  FROM hh",
            "work": "SELECT SELECT sp_id, 'Workplace' as place_type, latitude, longitude FROM work",
            "sch": "SELECT SELECT sp_id, 'School' as place_type, latitude, longitude FROM sch",
            "create_places": """
                CREATE TABLE places AS
                SELECT * FROM hh
                UNION BY NAME
                SELECT * FROM work
                UNION BY NAME
                SELECT * FROM sch;
                """,
            "create_person_last_known_location": """
                CREATE TABLE person_last_known_location (
                    person_id INTEGER PRIMARY KEY, -- PRIMARY KEY implies UNIQUE
                    place_id INTEGER,
                    minute_last_updated INTEGER
                )
                """,
        }

        # create the DuckDB tables for current locations of all persons
        # This table will be used to store the last known location of each person
        self.conn.execute(self.queries["create_person_last_known_location"])

    def _validate_and_set_required_params(self):
        """Validate and set required parameters."""
        required_keys = ["persons.file", "activities.file"]
        for key in required_keys:
            if key not in self.params:
                logger.error(f"Missing required parameter: {key}")
                raise MissingRequiredParameterError(key)

    def _set_optional_params_with_defaults(self):
        """Set optional parameters with default values if not provided."""
        optional_keys = [
            "start.datetime",
            "duration.hours",
            "timezone",
            "time.step.minutes",
            "places.file",
            "places.files",
            "contacts.file",
        ]
        for key in optional_keys:
            if key not in self.params:
                logger.warning(f"Optional parameter {key} not found, using default value.")
                self.params[key] = None

        self._set_default_start_datetime()
        self._set_default_duration_hours()
        self._set_default_timezone()
        self._parse_time_step_minutes()

        if self.params["places.file"] is None and "places.files" not in self.params:
            logger.error("No places file specified. Please provide 'places.file' or 'places.files'.")
            raise MissingRequiredParameterError(["places.file", "places.files"])

    def _set_default_start_datetime(self):
        if self.params["start.datetime"] is None:
            self.params["start.datetime"] = get_midnight(get_closest_monday(datetime.now())).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    def _set_default_duration_hours(self):
        if self.params["duration.hours"] is None:
            self.params["duration.hours"] = 24

    def _set_default_timezone(self):
        if self.params["timezone"] is None:
            self.params["timezone"] = "America/New_York"

    def _parse_time_step_minutes(self):
        if self.params["time.step.minutes"] is None:
            self.params["time.step.minutes"] = 60
            return
        if isinstance(self.params["time.step.minutes"], str):
            try:
                self.params["time.step.minutes"] = int(self.params["time.step.minutes"])
            except ValueError as err:
                logger.error(f"Invalid time step value: {self.params['time.step.minutes']}")
                raise InvalidTimeStepError(self.params["time.step.minutes"]) from err
        if "time.step.minutes" in self.params and isinstance(self.params["time.step.minutes"], str):
            try:
                self.params["time.step.minutes"] = int(self.params["time.step.minutes"])
            except ValueError as err:
                logger.error(f"Invalid time step value: {self.params.get('time.step', None)}")
                raise InvalidTimeStepError(self.params.get("time.step.minutes", None)) from err
        if self.params["time.step.minutes"] <= 0:
            logger.error(
                f"Invalid time step value: {self.params['time.step.minutes']}. Time step must be a positive integer."
            )
            raise InvalidTimeStepError(self.params["time.step.minutes"])
        if 1440 % self.params["time.step.minutes"] != 0:
            logger.error(
                f"Invalid time step value: {self.params['time.step.minutes']}. "
                "Time step must be a divisor of 1440 (the number of minutes in a day)."
            )
            raise InvalidTimeStepError(self.params["time.step.minutes"])

    def _remove_deprecated_params(self):
        """Remove deprecated parameters from the params dictionary."""
        deprecated_keys = ["stop.at", "steps.per.day"]
        for key in deprecated_keys:
            if key in self.params:
                logger.warning(f"Deprecated parameter {key} found, please update your configuration.")
                del self.params[key]

    def _compute_ticks(self):
        if "time.step.minutes" not in self.params or "duration.hours" not in self.params:
            logger.error("Missing required parameters: time.step.minutes or duration.hours")
            raise MissingRequiredParameterError(["time.step.minutes", "duration.hours"])
        self.params["ticks"] = int(self.params["duration.hours"] * 60 / self.params["time.step.minutes"])

    def initialize_population(self) -> None:
        """
        Initialize population

        This method initializes the population by creating the places and agents
        from the input data files.

        The method performs the following steps:"""
        # register the place types (derived classes should set place types)

        # create SharedContext consisting of all of the places in this model
        self.places_proj = PlacesProjection("places_projection", self.comm)
        self.context.add_projection(self.places_proj)

        # initialize the places
        if "places.file" in self.params and self.params["places.file"] is not None:
            logger.debug("Loading places from single file...")
            self.create_places_from_file(0, self.data_input_path / self.params["places.file"])
        else:
            logger.debug("Loading places from multiple files...")
            if "places.files" not in self.params:
                logger.error("Error: places.files parameter not specified.")
                raise MissingRequiredParameterError("places.files")
                logger.error("Error: places.files must be a list of filenames.")
                raise InvalidPlacesFilesError(self.params["places.files"])
                self.params["places.files"] = [self.params["places.files"]]
            elif not isinstance(self.params["places.files"], list):
                logger.error("Error: places.files must be a list of filenames.")
                raise InvalidPlacesFilesError(self.params["places.files"])

            place_filenames = [self.data_input_path / filename for filename in self.params["places.files"]]
            self.create_places_from_files(place_filenames)

        local_places = self.places_proj.get_local_places()
        logger.info(f"rank {self.rank}: number of local places={len(local_places)}")

        # schedulesList is a list of dict of personID->Schedule object
        schedulesList = self.create_activities(self.data_input_path / self.params["activities.file"])
        if not schedulesList or len(schedulesList) == 0:
            logger.error("Error: activities file is empty or not found.")
            raise MissingRequiredParameterError("activities.file")
        logger.info(f"rank {self.rank}: weekday activitiesMap size={len(schedulesList[0])}")
        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons at each
        # place
        self.contact_map = {}
        if "contact.file" in self.params:
            logger.debug("Loading contact file...")

            self.contact_map = self.create_contacts(self.data_input_path / self.params["contact.file"])
        else:
            logger.error("Error: contact file not specified.")

        logger.debug(f"rank {self.rank}: contacts size={len(self.contact_map)}")

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        # self.person_id_map = {}
        self.create_persons(self.data_input_path / self.params["persons.file"], schedulesList, self.rng)

        # initialize the table for person last known locations
        # person_last_known_location_df is a polars DataFrame with columns:
        person_last_known_location_df = pl.DataFrame(
            [person.last_known_place for person in self.context.agents(agent_type=0)]
        )
        # write the person locations to the DuckDB table
        self.conn.execute(
            """
            INSERT INTO person_last_known_location SELECT * FROM person_last_known_location_df
            """
        )
        logger.info(
            "person_last_known_location_df:\n"
            f"number of rows: {person_last_known_location_df.shape[0]}\n"
            f"{person_last_known_location_df.head}"
        )
        result = self.conn.execute(self.queries["get_tables"]).fetchall()
        logger.info(f"rank {self.rank}: DuckDB tables after initialization: {result}")

    def create_persons(
        self,
        personsFile: pathlib.Path,
        schedulesList: list[dict[int, list[int]]],
        rng,
    ) -> None:
        """Create persons from the given file.

        Args:
            personsFile (pathlib.Path): The persons file.
            schedulesList (list[dict]): The list of activities maps.
            rng: The random number generator.
        """
        # get the person type and data type
        personType = self.get_person_config().person_type

        # get the activities map, which is a dict of personID->Activities object
        # Currently, we assume that there is only one schedule in the list,
        # which is the weekday schedule. If there are multiple schedules, we
        # will need to handle them differently.
        activitiesMap = schedulesList[0] if schedulesList else {}

        # get the activities data type: namedtuple to store places for activities
        activitiesDataType = self.get_activities_data_type()

        # get the planned_activity_names, which are the fields in the person file that
        # contain the place ids (e.g. 'sp_work_id', 'sp_school_id', etc.)
        planned_activity_names = self.get_planned_activity_names()

        # get the activity names (list should be at least as long as planned_activity_names)
        activity_names = self.get_activity_names()

        # get the alternate activity names (activities not in the planned activities)
        alternate_activities_names = activity_names[len(planned_activity_names) :]

        # load the persons from the file
        table = pq.read_table(personsFile)

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                p = dict(zip(table.column_names, row))

                personID = p["sp_id"]

                # TODO: add tests for this
                #  - activities_data = [ p[x] for x in planned_activity_names ]
                #  - all places should be in placeMap
                #  - the first place is a household
                #  - how to handle the case where the person is not on this rank?
                #  - how to handle the case where the person is not in the activitiesMap?
                places = [convert_to_int(p[x]) for x in planned_activity_names]

                for place in places:
                    if isinstance(place, str):
                        logger.error(f"Error: Place {place} not found.")
                        return

                hhId = places[0]  # p['sp_hh_id']

                household = self.places_proj.lookup_place(hhId)
                if not household:
                    logger.error(f"Error: No household found for {p}")
                    continue

                rank = household.rank

                if rank != self.rank:
                    logger.error(f"Error: Person {personID} tagged on rank={rank} is not on this rank.")
                    continue

                # Person
                #  - activitiesMap: schedulesList[0] is a dict of personID->Activities object

                # if personID not in activitiesMap:
                if personID not in activitiesMap:
                    logger.error(f"Error: No activities found for person {personID}.")
                    continue

                # get the schedule for the person
                schedule = activitiesMap[personID]
                activities = Activities(personID, "weekday", tuple(schedule))
                schedules = Schedules()
                schedules.addActivities(activities)

                # create an empty list for weekend activities (if not already present)
                weekend_activities = Activities(personID, "weekend", ())
                schedules.addActivities(weekend_activities)

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
                    p,  # initDict for additional data
                )

                self.context.add(person)
                self.places_proj.add(person)
                self.places_proj.assign_agent_to_place(person, household)

    def create_places_from_file(self, placeTypeIndex: int, placesFile: pathlib.Path) -> None:
        """
        Create places from the given file.

        Args:
            placeTypeIndex (int): The index of the place type.
            placesFile (pathlib.Path): The place file.
        """

        # get the place type
        placeConfig = self.get_place_config(placeTypeIndex)
        placeType = placeConfig.place_type
        placeDataType = placeConfig.dataType

        table_names = ["hh", "work", "sch"]
        if placeConfig.name == "Places":
            table_names = [placeConfig.name.lower()]

        table_name = table_names[placeTypeIndex]
        print(
            f"Creating places of type {placeConfig.name} from {placesFile}:"
            f" {placeType.__name__} with data type {placeDataType.__name__}"
            f" and table name {table_name}"
        )
        table = pq.read_table(placesFile)
        # Register the PyArrow table as a DuckDB view
        self.conn.register("my_temp_view", table)

        # Use that view in your CREATE TABLE AS query
        query = f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM my_temp_view'  # noqa: S608
        self.conn.execute(query)

        # Optionally fetch the result (if needed)
        # result = self.conn.table(table_name).arrow()

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                place_record = dict(zip(table.column_names, row))
                if "rank" not in place_record:
                    place_record["rank"] = 0
                place = placeType(place_record, placeDataType)
                self.places_proj.add_place(place)

    def create_places_from_files(self, places_files: list[pathlib.Path]) -> None:
        """
        Create places from the given files.

        Args:
            places_files (list[pathlib.Path]): The list of place files.
        """
        for placeTypeIndex, placesFile in enumerate(places_files):
            self.create_places_from_file(placeTypeIndex, placesFile)

        # create the places table
        logger.info("Creating places table...")
        self.conn.execute(self.queries["create_places"])
        # verify the places table was created
        # if not self.conn.table("places").exists():
        # logger.error("Error: places table was not created.")
        # raise InvalidTableNameError("places")
        # show the tables to verify creation
        logger.info("Showing tables after creating places...")
        result = self.conn.execute("SHOW TABLES").fetchall()  # Show tables to verify creation
        logger.info(f"Created tables: {result}")

        places_df = self.conn.query("SELECT * FROM places").pl()
        logger.info(f"Number of places created: {len(places_df)}")

        # add a remote place
        remote_place = self.get_remote_place_config().place_type(
            {"sp_id": 0, "rank": 0}, self.get_remote_place_config().dataType
        )
        self.places_proj.add_place(remote_place)

    def create_activities(self, activitiesFile: pathlib.Path) -> list[dict[int, list[int]]]:
        """Create activities from the given file.
        This method reads the activities file and creates a mapping of person IDs
        to their activities. Activities are grouped in schedules for each person.
        The default schedule is for weekdays, but weekend activities can be added later.

        Args:
            activitiesFile (pathlib.Path): The activities file.
        Returns:
            list[dict[int, list[int]]]: A list containing a single dictionary
            mapping person IDs to their activities for a weekday.
        """
        # activitiesMap looks like:
        # personID -> Activities object
        act_map = {}

        # This should be the most eficient way to extract the data via pyarrow
        # See https://stackoverflow.com/questions/53157495/fastest-way-to-iterate-pyarrow-table/55633193#55633193
        table = pq.read_table(activitiesFile)

        for batch in table.to_batches():
            d = batch.to_pydict()
            for sp_persons_id, activity_id, activity_seq, start, end in zip(
                d["sp_persons_id"], d["activity_id"], d["activity_sequence"], d["starttime_min"], d["endtime_min"]
            ):
                if sp_persons_id not in act_map:
                    act_map[sp_persons_id] = [Act(sp_persons_id, activity_id, activity_seq, start, end)]
                else:
                    act_map[sp_persons_id].append(Act(sp_persons_id, activity_id, activity_seq, start, end))

        return [act_map]

    def create_contacts(self, contactFile: pathlib.Path) -> dict[int, dict[int, int]]:
        # contactMap looks like:
        # personID -> { hour_of_day -> [ otherPersonIDs ] }
        # dict
        contactMap = {}

        # with open(contactFile, 'r', newline='') as f:
        #     contacts = DictReader(f)
        table = pq.read_table(contactFile)

        for batch in table.to_batches():
            d = batch.to_pydict()
            for source, target, hour_of_the_day in zip(d["from_person"], d["to_person"], d["hour"]):
                if source not in contactMap:
                    contactMap[source] = {}

                if hour_of_the_day not in contactMap[source]:
                    contactMap[source][hour_of_the_day] = []

                contactMap[source][hour_of_the_day].append(target)

        return contactMap

    def step(self) -> None:
        """Step the model forward one time step."""

        self.cal.increment(self.time_step_minutes)

        # log the current step
        logger.info(
            "Step on "
            f"day {self.cal.day_of_year}, "
            f"hour {self.cal.hour_of_day}, "
            f"minute {self.cal.minute_of_day}"
        )

        # 2025-02-26 jcline: this is a hack to get the person_id_map
        # self.get_local_ids()

        self.get_environment().step(self.context, self.cal)

        # sequence of actions
        # 1. sense physical environment
        # 2. sense social environment
        # 3. update state
        # 4. update beliefs
        # 5. communicate
        # 6. make decisions
        # 7. act on decisions

        # self.send_messages_between_agents()

        for person in self.context.agents():
            person.step(self.context, self.cal)

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
            logger.debug(f"Adding person {person.id} to place {person.state.place_id}")
            # if person.state.place_id not in self.place_map:
            #     logger.error(f"Person {person.id} has no place.")
            #     return
            # self.place_map[person.state.place_id].addPerson(person)

    def make_contacts(self, tick) -> None:
        for person in self.context.agents():
            personsContactMap = self.contact_map.get(person.id)
            if not personsContactMap:  # if person has no network
                # logger.debug(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(person.state.place_id)
            if not contactIDs:
                # logger.debug(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.state.place_id}.")
                continue

            contacts = []
            for contactID in contactIDs:
                contacts.append(self.context.agent(self.person_id_map[contactID]))
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
                    recipient=recipient.id,
                    message=f"Hello from {person.uid}",
                    timestamp=(
                        "Step on "
                        f"day {self.cal.day_of_year}, "
                        f"hour {self.cal.hour_of_day}, "
                        f"minute {self.cal.minute_of_day}"
                    ),
                )

            messages = person.send_messages()
            if len(messages) > 0:
                logger.debug(f"Person {person} has messages.")

                for message in messages:
                    recipient = message.recipient
                    # recipient_person = self.context.agent(self.person_id_map[recipient])
                    if recipient in self.person_id_map:  # message to local person
                        recipient_uid = self.person_id_map[recipient]
                        logger.debug(f"Message from {message.sender} to {person_cache[recipient_uid].state}:")
                        person_cache[recipient_uid].receive_message(message)
                    else:  # message to remote person
                        logger.debug(f"Message from {message.sender} to {recipient}:")

                        # get remote person ID
                        remote_person_ids.append(recipient)

                        # create message to send to other rank
                        message_to_send = message
                        message_to_send.recipients = [recipient]
                        messages_to_send.append(message_to_send)

            else:
                logger.debug(f"Person {person.id} has no messages.")
                continue

        # Exchange messages between processors
        all_messages = self.exchange_messages(remote_person_ids, messages_to_send)

        # Step 2: Deliver messages from remote processors
        for message in all_messages:
            recipient = message.recipient
            if recipient in self.person_id_map:
                recipient_uid = self.person_id_map[recipient]
                logger.debug(f"Remote message from {message.sender} to " f"{person_cache[recipient_uid].state}:")
                person_cache[recipient_uid].receive_message(message)
            else:
                logger.debug(f"Remote message from {message.sender} to {recipient} not delivered")

        # Step 3: Process messages
        for person in agents:
            person.process_messages()

    def get_remote_person_id_map(self, remote_person_ids: list[int]) -> dict[int, int]:
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
                        msg = {"id": person_id, "uid": person_uid}
                        send_buffers[rank].append(msg)

        # 3. Send remote person ID->UID map to other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [msg for buffer in received_buffers for msg in buffer]
        for msg in all_messages:
            remote_person_id_map[msg["id"]] = msg["uid"]

        return remote_person_id_map

    def exchange_messages(self, remote_person_ids: list[int], messages_to_send: list[Message]) -> list[Message]:
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

        logger.info(f"Simulation took {end_time - self.start_time} seconds.")


# Register SIModel
Models.add_model(SIModel.__module__ + "." + SIModel.__name__, SIModel)


# utility functions
def update_activities_data(activities_data: namedtuple, **kwargs) -> namedtuple:
    """Update the activities data."""
    return activities_data._replace(**kwargs)
