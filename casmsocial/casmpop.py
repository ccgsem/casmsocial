"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the CasmPop
"""

import os
import pathlib
import time
from collections import namedtuple
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import repast4py
import repast4py.random
from dotenv import load_dotenv
from loguru import logger
from mpi4py import MPI
from numpy.random import Generator
from repast4py import context as ctx
from repast4py import schedule

from casmsocial.activities import Act, Activities, Schedules
from casmsocial.data_utilities import (
    check_if_table_exists,
    convert_to_int,
    quote_table_identifier,
)
from casmsocial.date_utilities import get_closest_monday, get_midnight
from casmsocial.ducklake_utils import get_ducklake_connection
from casmsocial.environment import Environment
from casmsocial.factory import Models
from casmsocial.message import Message
from casmsocial.model import Model
from casmsocial.person import Person, person_cache
from casmsocial.place import Place, PlacesProjectionV2
from casmsocial.sim_time import SimTime


class MissingRequiredParameterError(Exception):
    def __init__(self, keys):
        if isinstance(keys, (list, tuple)):
            keys_str = ", ".join(str(k) for k in keys)
        else:
            keys_str = str(keys)
        super().__init__(f"Missing required parameter(s): {keys_str}")


class MissingRequiredTableError(Exception):
    def __init__(self, keys):
        if isinstance(keys, (list, tuple)):
            keys_str = ", ".join(str(k) for k in keys)
        else:
            keys_str = str(keys)
        super().__init__(f"Missing required table(s): {keys_str}")


class InvalidTimeStepError(Exception):
    def __init__(self, value):
        super().__init__(
            f"Invalid time step value: {value}. Time step must be "
            "an integer."
        )


class InvalidTableNameError(Exception):
    def __init__(self, table_name):
        super().__init__(f"Invalid table name: {table_name}")


class MissingDataPathError(Exception):
    def __init__(self, data_path):
        super().__init__(f"Missing or invalid data path: {data_path}")


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


class CasmPop(Model):
    """
    The CasmPop class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    The CasmPop class is a subclass of the Model class, which is an abstract
    base class that defines the interface for all models in the casmsocial.
    The CasmPop class implements the start and step methods, which are called
    by the run function in the casmsocial module to start and run the model.

    The CasmPop class adds the following functionality to the Model class:

    - The CasmPop class initializes geographic places and agents.
    - The CasmPop class updates the  environment for the current time step.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    # class variables
    __personClass: type[Person] = Person
    __placeClass: type[Place] = Place

    # list of planned activities (column names in person file for
    # activities)
    __planned_activity_names: ClassVar[list[str]] = []

    # list of activities
    __activity_names: ClassVar[list[str]] = []

    # activites data type: namedtuple
    __activities_data_type: namedtuple = None

    # environment
    __environment: Environment = None

    # class methods
    @classmethod
    def get_default_parameters(cls) -> dict:
        """Get the default parameters for the CasmPop model."""
        return {
            "model.name": cls.__module__ + "." + cls.__name__,
            "random.seed": 42,
            "places.table": None,
            "persons.table": None,
            "activities.table": None,
            "contacts.table": None,
            "start.datetime": None,
            "duration.hours": None,
            "timezone": None,
            "time.step.minutes": None,
        }

    @classmethod
    def get_default_performance_parameters(cls) -> dict:
        """Get the default performance parameters for the CasmPop model.

        These parameters are used to configure the parallel processing
        settings for places and agents. The default settings are based on
        performance testing and may be adjusted based on the specific model
        and hardware configuration.
        """
        return {
            "parallel.places.enabled": True,
            "parallel.places.min_threshold": 50,
            "parallel.places.max_workers": None,  # Use CPU count
            "parallel.places.auto_update": False,
            "parallel.agents.enabled": False,
            "parallel.agents.min_threshold": 1000000,
        }

    @classmethod
    def getPersonClass(cls) -> type[Person]:
        """Get the person class."""
        return cls.__personClass

    @classmethod
    def setPersonClass(
        cls, person_class: type[Person], person_data
    ) -> None:
        """Set the person class."""
        person_class.setPersonDataClass(person_data)
        cls.__personClass = person_class

    @classmethod
    def getPlaceClass(cls) -> type[Place]:
        """Get the place class."""
        return cls.__placeClass

    @classmethod
    def setPlaceClass(
        cls, place_class: type[Place], place_data
    ) -> None:
        """Set the place class."""
        place_class.setPlaceDataClass(place_data)
        cls.__placeClass = place_class

    @classmethod
    def register_planned_activity_names(
        cls, planned_activity_names: list[str]
    ) -> None:
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
                "ActivitiesDataclass", cls.get_activity_names()
            )
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
        """Constructor for the CasmPop class

        Args:
            comm: the mpi communicator over which the model is distributed.
            params: the simulation input parameters
        """
        Model.set_model(self)

        logger.info("Creating CasmPop...")
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.params = params

        # start timer
        self.start_time = time.time()

        self._validate_and_set_required_params()
        self._set_optional_params_with_defaults()
        self._compute_ticks()
        self._configure_parallel_processing()

        logger.info(
            f"Rank {self.rank} starting CasmPop with params: "
            f"{self.params}"
        )

        # create the schedule
        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_event(0, self.build_context)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(self.params["ticks"])
        self.runner.schedule_end_event(self.at_end)

        # set the start datetime and timezone
        start_datetime = datetime.strptime(
            self.params["start.datetime"], "%Y-%m-%d %H:%M:%S"
        )
        tz = ZoneInfo(self.params["timezone"])
        start_datetime = start_datetime.replace(tzinfo=tz)

        # initialize the simulation time
        self.cal = SimTime(start_datetime=start_datetime)

        # set the time step in minutes
        self.time_step_minutes = self.params["time.step.minutes"]

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(self.comm)

        # set the data resources (e.g. data paths, DuckLake connection, etc.)
        self._set_data_resources()

        self.queries = {}

        self.contact_map = {}

        # **note**
        #
        # If you are using the repast4py.parameters module, you can just
        # include a 'random.seed' key in your YAML or JSON configuration file.
        # The framework will automatically call init() for you during parameter
        # initialization.

    def _set_data_resources(self) -> None:
        # the data input path should be defined by $CASMSOCIAL_DATA_PATH
        load_dotenv()  # load environment variables from .env file if it exists

        # check if the data path is set and valid
        data_path = os.environ.get("CASMSOCIAL_DATA_PATH")
        if not data_path or not pathlib.Path(data_path).exists():
            raise MissingDataPathError(data_path)
        self.data_path = pathlib.Path(data_path)

        ducklake_path = os.environ.get("CASMSOCIAL_DUCKLAKE_PATH")
        if ducklake_path:
            self.conn = get_ducklake_connection(pathlib.Path(ducklake_path))
        else:
            raise MissingDataPathError("CASMSOCIAL_DUCKLAKE_PATH")

    def _validate_and_set_required_params(self):
        """Validate and set required parameters."""
        required_keys = ["places.table", "persons.table", "activities.table"]
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
            "contacts.table",
        ]
        for key in optional_keys:
            if key not in self.params:
                logger.warning(
                    f"Optional parameter {key} not found, using "
                    f"default value."
                )
                self.params[key] = None

        self._set_default_start_datetime()
        self._set_default_duration_hours()
        self._set_default_timezone()
        self._parse_time_step_minutes()

    def _set_default_start_datetime(self):
        if self.params["start.datetime"] is None:
            midnight = get_midnight(get_closest_monday(datetime.now()))
            self.params["start.datetime"] = midnight.strftime(
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
                self.params["time.step.minutes"] = int(
                    self.params["time.step.minutes"]
                )
            except ValueError as err:
                logger.error(
                    f"Invalid time step value: "
                    f"{self.params['time.step.minutes']}"
                )
                raise InvalidTimeStepError(
                    self.params["time.step.minutes"]
                ) from err
        if (
            "time.step.minutes" in self.params
            and isinstance(self.params["time.step.minutes"], str)
        ):
            try:
                self.params["time.step.minutes"] = int(
                    self.params["time.step.minutes"]
                )
            except ValueError as err:
                logger.error(
                    f"Invalid time step value: "
                    f"{self.params.get('time.step', None)}"
                )
                raise InvalidTimeStepError(
                    self.params.get("time.step.minutes", None)
                ) from err
        if self.params["time.step.minutes"] <= 0:
            logger.error(
                f"Invalid time step value: "
                f"{self.params['time.step.minutes']}. "
                f"Time step must be a positive integer."
            )
            raise InvalidTimeStepError(
                self.params["time.step.minutes"]
            )
        if 1440 % self.params["time.step.minutes"] != 0:
            logger.error(
                f"Invalid time step value: "
                f"{self.params['time.step.minutes']}. Time step must be "
                f"a divisor of 1440 (minutes in a day)."
            )
            raise InvalidTimeStepError(
                self.params["time.step.minutes"]
            )

    def _compute_ticks(self):
        if (
            "time.step.minutes" not in self.params
            or "duration.hours" not in self.params
        ):
            logger.error(
                "Missing required parameters: time.step.minutes or "
                "duration.hours"
            )
            raise MissingRequiredParameterError(
                ["time.step.minutes", "duration.hours"]
            )
        self.params["ticks"] = int(
            self.params["duration.hours"]
            * 60
            / self.params["time.step.minutes"]
        )

    def _configure_parallel_processing(self):
        """Configure parallel processing settings."""
        # Default parallel processing settings
        if "parallel.places.enabled" not in self.params:
            self.params["parallel.places.enabled"] = True
        if "parallel.places.min_threshold" not in self.params:
            self.params["parallel.places.min_threshold"] = 50
        if "parallel.places.max_workers" not in self.params:
            self.params["parallel.places.max_workers"] = None

        # Disable automatic place updates during simulation steps
        if "parallel.places.auto_update" not in self.params:
            self.params["parallel.places.auto_update"] = False

        # Agent processing: parallel disabled due to performance degradation
        # Thread overhead exceeded benefits for lightweight operations
        if "parallel.agents.enabled" not in self.params:
            self.params["parallel.agents.enabled"] = False
        if "parallel.agents.min_threshold" not in self.params:
            self.params["parallel.agents.min_threshold"] = 1000000

    def build_context(self) -> None:
        """
        Initialize population.

        This method initializes the population by creating the places and
        agents from the input data files.
        """

        # register the agent types (derived classes should set agent types)

        # create SharedContext consisting of all places in this model
        # Use enhanced projection with configurable parallel processing
        parallel_enabled = self.params.get(
            "parallel.places.enabled", True
        )
        parallel_min_threshold = self.params.get(
            "parallel.places.min_threshold", 50
        )
        parallel_max_workers = self.params.get(
            "parallel.places.max_workers", None
        )

        self.places_proj = PlacesProjectionV2(
            "places_projection",
            self.comm,
            enable_parallel_updates=parallel_enabled,
            parallel_min_threshold=parallel_min_threshold,
            parallel_max_workers=parallel_max_workers,
        )
        self.context.add_projection(self.places_proj)

        # create the input tables
        self.create_input_tables()

        # initialize the places
        # (note: already checked if "places.file" is in params)
        self.create_places()

        local_places = self.places_proj.get_local_places()
        logger.info(
            f"rank {self.rank}: number of local "
            f"places={len(local_places)}"
        )
        # add geometry to the places table
        # self.conn.execute(self.queries["add_geometries"])

        # contact_map is a dict of personID->{placeID->[personID]}
        # i.e. it is a map of personIDs to a list of contacted persons
        # at each place
        if "contacts.table" in self.params and self.params.get(
            "contacts.table"
        ):
            logger.info(
                f"Loading contact file "
                f"{self.params['contacts.table']}..."
            )

            self.contact_map = self.create_contacts()
        else:
            logger.warning("Warning: contacts table not specified.")

        logger.debug(
            f"rank {self.rank}: contacts size={len(self.contact_map)}"
        )

        self.rng = repast4py.random.default_rng

        # agent_id_map is a map of personID->repast4py.Agent.uid
        # self.person_id_map = {}
        self.create_persons(self.rng)

        result = self.conn.execute(
            self.queries["get_tables"]
        ).fetchall()
        logger.info(
            f"rank {self.rank}: DuckDB tables after "
            f"initialization: {result}"
        )

    def create_input_tables(self) -> None:
        """Load tables from the database."""

        #  create the places table as a view from the ducklake table
        places_table = self.params.get("places.table")
        if not check_if_table_exists(self.conn, places_table):
            raise MissingRequiredTableError(places_table)
        logger.info(f"creating <places> view from <{places_table}>...")
        self.conn.execute(
            "CREATE OR REPLACE VIEW places AS "
            "SELECT * FROM "
            f"{quote_table_identifier(places_table)}"  # noqa: S608
        )

        # create the persons table
        imputation = (
            self.params.get("Imputation", None)
            if "Imputation" in self.params
            else None
        )
        persons_table = self.params.get("persons.table")
        if not check_if_table_exists(self.conn, persons_table):
            raise MissingRequiredTableError(persons_table)
        logger.info(
            f"creating <persons> view from <{persons_table}>..."
        )
        if imputation is not None:
            logger.info(
                f"Using imputation {imputation} for <persons> "
                f"table <{persons_table}>..."
            )
            persons_identifier = quote_table_identifier(
                persons_table
            )
            print(f"persons_identifier: {persons_identifier}")
            self.conn.execute(
                f"""CREATE OR REPLACE VIEW persons AS
                SELECT * FROM {persons_table}
                WHERE Imputation = {imputation};
                """  # noqa: S608
            )
            print("Created persons view with imputation filter.")
        else:
            logger.info(
                f"Using <persons> table {persons_table}..."
            )
            persons_identifier = quote_table_identifier(
                persons_table
            )
            self.conn.execute(
                f"CREATE OR REPLACE VIEW persons AS "
                f"SELECT * FROM {persons_identifier}"  # noqa: S608
            )

        # create the activities table
        activities_table = self.params.get("activities.table")
        if not check_if_table_exists(self.conn, activities_table):
            raise MissingRequiredTableError(activities_table)
        if imputation is not None:
            logger.info(
                f"Using imputation {imputation} for activities "
                f"table {activities_table}..."
            )
            activities_identifier = quote_table_identifier(
                activities_table
            )
            self.conn.execute(
                f"""
                CREATE OR REPLACE VIEW activities AS
                SELECT * FROM {activities_identifier}
                WHERE Imputation = {imputation};
                """  # noqa: S608
            )
        else:
            logger.info(
                f"Using activities table {activities_table}..."
            )
            activities_identifier = quote_table_identifier(
                activities_table
            )
            self.conn.execute(
                f"CREATE OR REPLACE VIEW activities AS "
                f"SELECT * FROM {activities_identifier}"  # noqa: S608
            )

        self.queries = {
            "get_tables": "SHOW TABLES",
            "add_geometries": """
                -- 1. Load the spatial extension
                -- This is necessary to use ST_Point and other geospatial
                -- functions.
                INSTALL spatial;
                LOAD spatial;
                -- 2. Add the 'location' column of type GEOMETRY
                -- GEOMETRY is a generic spatial type that can store points,
                --lines, polygons, etc.
                ALTER TABLE places ADD COLUMN location GEOMETRY;
                -- 3. Populate the 'location' column
                -- ST_Point expects (X, Y) which translates to (longitude,
                -- latitude) for geographic points.
                UPDATE places
                -- Ensure that longitude and latitude are in the correct order
                -- for ST_Point
                SET location = ST_Point(longitude, latitude);
                """,
        }

        # create the contacts table if it exists
        if "contacts.table" in self.params and self.params.get(
            "contacts.table"
        ):
            contacts_table = self.params.get("contacts.table")
            if not check_if_table_exists(self.conn, contacts_table):
                logger.error(
                    f"Error: contacts table {contacts_table} "
                    "does not exist in the database."
                )
                raise MissingRequiredTableError(
                    contacts_table
                )
            if imputation is not None:
                logger.info(
                    f"Using imputation {imputation} for "
                    f"contacts table {contacts_table}..."
                )
                contacts_identifier = quote_table_identifier(
                    contacts_table
                )
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE VIEW contacts AS
                    SELECT * FROM {contacts_identifier}
                    WHERE Imputation = {imputation};
                    """  # noqa: S608
                )
            else:
                logger.info(
                    f"Using contacts table {contacts_table}..."
                )
                contacts_identifier = quote_table_identifier(
                    contacts_table
                )
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW contacts AS "
                    f"SELECT * FROM {contacts_identifier}"  # noqa: S608
                )

    def create_persons(
        self,
        rng: Generator,
    ) -> None:
        """Create persons from the given file.

        Args:
            rng: The random number generator.
        """
        # get the person type
        personType = self.getPersonClass()

        # Create the activities
        #  - schedulesList is a list of dict of personID->Schedule
        schedulesList = self.create_activities()
        if not schedulesList or len(schedulesList) == 0:
            logger.error("Error: no activities found.")
            raise MissingRequiredParameterError("activities.file")
        logger.info(
            f"rank {self.rank}: weekday activitiesMap "
            f"size={len(schedulesList[0])}"
        )

        # get the activities map, which is a dict of
        # personID->Activities object. Currently, we assume that there is
        # only one schedule in the list, which is the weekday schedule.
        # If there are multiple schedules, we will need to handle them
        # differently.
        activitiesMap = schedulesList[0] if schedulesList else {}

        # get the activities data type: namedtuple to store places
        # for activities
        activitiesDataType = self.get_activities_data_type()

        # get the planned_activity_names, which are the fields in
        # the person file that contain the place ids
        # (e.g. 'sp_work_id', 'sp_school_id', etc.)
        planned_activity_names = self.get_planned_activity_names()

        # get the activity names
        # (list should be as long as planned_activity_names)
        activity_names = self.get_activity_names()

        # get alternate activity names
        # (activities not in planned activities)
        alternate_activities_names = activity_names[
            len(planned_activity_names):
        ]

        # load the persons from the file
        # table = pq.read_table(personsFile)
        table = self.conn.execute(
            "SELECT * FROM persons"
        ).fetch_arrow_table()

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                p = dict(zip(table.column_names, row))

                personID = p["sp_id"]

                # TODO: add tests for this
                #  - activities_data = [p[x] for x in
                #    planned_activity_names]
                #  - all places should be in placeMap
                #  - the first place is a household
                #  - handle person not on this rank?
                #  - handle person not in activitiesMap?
                places = [
                    convert_to_int(p[x])
                    for x in planned_activity_names
                ]

                for place in places:
                    if isinstance(place, str):
                        logger.error(f"Error: Place {place} not found.")
                        return

                hhId = places[0]  # p['sp_hh_id']

                household = self.places_proj.lookup_place(hhId)
                if not household:
                    logger.error(
                        f"Error: No household found for {p}"
                    )
                    continue

                rank = household.rank

                if rank != self.rank:
                    logger.error(
                        f"Error: Person {personID} tagged on "
                        f"rank={rank} is not on this rank."
                    )
                    continue

                # Person
                #  - activitiesMap: schedulesList[0] is a dict of
                #    personID->Activities object

                # if personID not in activitiesMap:
                if personID not in activitiesMap:
                    logger.error(
                        f"Error: No activities found for "
                        f"person {personID}."
                    )
                    continue

                # get the schedule for the person
                schedule = activitiesMap[personID]
                activities = Activities(personID, "weekday", tuple(schedule))
                schedules = Schedules()
                schedules.addActivities(activities)

                # create an empty list for weekend activities (if not already
                # present)
                weekend_activities = Activities(personID, "weekend", ())
                schedules.addActivities(weekend_activities)

                # add alternate places for alternate activities
                # not already in the person's schedule. These
                # alternate activities are initially empty and may
                # be determined during the simulation.
                for activity_name in alternate_activities_names:
                    activities = Activities(
                        personID, activity_name, ()
                    )
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

    def create_places(self) -> None:
        """Create places in the project."""

        logger.info("Creating places...")

        # Get the place type and data type
        placeType = self.getPlaceClass()
        placeDataType = placeType.getPlaceDataClass()

        # Create the places table
        table = self.conn.execute(
            "SELECT * FROM places"
        ).fetch_arrow_table()

        for batch in table.to_batches():
            for row in zip(*batch.columns):
                # convert arrow scalars to python
                row = [x.as_py() for x in row]
                place_record = dict(zip(table.column_names, row))
                if "rank" not in place_record:
                    place_record["rank"] = 0
                place = placeType(place_record, placeDataType)
                self.places_proj.add_place(place)

    def create_places_from_file(
        self, placeTypeIndex: int, placesFile: pathlib.Path
    ) -> None:
        """
        Create places from the given file.

        Args:
            placeTypeIndex (int): The index of the place type.
            placesFile (pathlib.Path): The place file.
        """

        # get the place type
        logger.info(f"Creating places from {placesFile}...")

        placeType = self.get_agent_type_configs()[
            Place.TYPE
        ].agent_type
        placeDataType = self.get_agent_type_configs()[
            Place.TYPE
        ].agent_data_type
        table_name = "places"
        print(
            f"Creating places from {placesFile}: "
            f"with data type {placeDataType.__name__} "
            f"and table name {table_name}"
        )
        # table = pq.read_table(placesFile)
        table = self.conn.execute(
            "SELECT * FROM places"
        ).fetch_arrow_table()

        # Register the PyArrow table as a DuckDB view
        self.conn.register("my_temp_view", table)

        # Use that view in your CREATE TABLE AS query
        # query = f"""
        #     CREATE OR REPLACE TABLE "{table_name}" AS
        #     SELECT * FROM my_temp_view'
        # """
        # self.conn.execute(query)

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

    def create_activities(self) -> list[dict[int, list[int]]]:
        """Create activities from the given file.

        This method reads the activities file and creates a mapping of
        person IDs to their activities. Activities are grouped in
        schedules for each person. The default schedule is for weekdays,
        but weekend activities can be added later.

        Returns:
            list[dict[int, list[int]]]: A list containing a single
            dictionary mapping person IDs to their activities for a
            weekday.
        """
        # activitiesMap looks like:
        # personID -> Activities object
        act_map = {}

        # This should be the most efficient way to extract the data via
        # pyarrow. See
        # https://stackoverflow.com/questions/53157495/...
        table = self.conn.execute(
            "SELECT * FROM activities"
        ).fetch_arrow_table()

        for batch in table.to_batches():
            d = batch.to_pydict()
            for (
                sp_persons_id,
                activity_id,
                activity_seq,
                start,
                end,
                act_place_id,
            ) in zip(
                d["sp_persons_id"],
                d["activity_id"],
                d["activity_sequence"],
                d["starttime_min"],
                d["endtime_min"],
                d["sp_act_id"],
            ):
                if sp_persons_id not in act_map:
                    act_map[sp_persons_id] = [
                        Act(
                            sp_persons_id,
                            activity_id,
                            activity_seq,
                            start,
                            end,
                            act_place_id,
                        )
                    ]
                else:
                    act_map[sp_persons_id].append(
                        Act(
                            sp_persons_id,
                            activity_id,
                            activity_seq,
                            start,
                            end,
                            act_place_id,
                        )
                    )

        return [act_map]

    def create_contacts(self) -> dict[int, dict[int, int]]:
        # contactMap looks like:
        # personID -> { hour_of_day -> [ otherPersonIDs ] }
        contactMap = {}

        # with open(contactFile, 'r', newline='') as f:
        #     contacts = DictReader(f)
        # table = pq.read_table(contactFile)
        table = self.conn.execute(
            "SELECT * FROM contacts"
        ).fetch_arrow_table()

        for batch in table.to_batches():
            d = batch.to_pydict()
            for source, target, hour_of_the_day in zip(
                d["from_person"], d["to_person"], d["hour"]
            ):
                if source not in contactMap:
                    contactMap[source] = {}

                if hour_of_the_day not in contactMap[source]:
                    contactMap[source][hour_of_the_day] = []

                contactMap[source][hour_of_the_day].append(
                    target
                )

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

        # Automatic place updates are disabled - they caused
        # performance degradation

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

        # Process person agents (TYPE=0) with optimized sequential
        # processing. Parallel processing caused performance degradation
        # due to thread overhead exceeding benefits for lightweight
        # agent operations
        person_agents = list(
            self.context.agents(agent_type=0)
        )  # Only person agents

        if len(person_agents) > 0:
            agent_start_time = time.time()

            # Optimized sequential processing with minimal overhead
            for person in person_agents:
                person.step(self.context, self.cal)

            agent_processing_time = time.time() - agent_start_time

            # Log performance for large datasets
            if self.rank == 0 and len(person_agents) >= 1000:
                agents_per_second = (
                    len(person_agents) / agent_processing_time
                    if agent_processing_time > 0
                    else 0
                )
                logger.info(
                    f"Person agent processing: "
                    f"{len(person_agents):,} agents, "
                    f"{agent_processing_time:.2f}s, "
                    f"rate: {agents_per_second:,.0f} agents/sec"
                )

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
            logger.debug(
                f"Adding person {person.id} to place "
                f"{person.state.place_id}"
            )
            # if person.state.place_id not in self.place_map:
            #     logger.error(f"Person {person.id} has no place.")
            #     return
            # self.place_map[person.state.place_id].addPerson(person)

    def make_contacts(self, tick) -> None:
        for person in self.context.agents():
            personsContactMap = self.contact_map.get(
                person.id
            )
            if not personsContactMap:  # if person has no network
                # logger.debug(f"Person {person.id} has no network.")
                continue

            contactIDs = personsContactMap.get(
                person.state.place_id
            )
            if not contactIDs:
                # logger.debug(
                #     f"Person {person.id} has no contacts at "
                #     f"place {person.state.place_id}.")
                continue

            contacts = []
            for contactID in contactIDs:
                uid = self.person_id_map[contactID]
                contacts.append(self.context.agent(uid))
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
                    # recipient_person = self.context.agent(
                    #     self.person_id_map[recipient]
                    # )
                    if recipient in self.person_id_map:
                        # message to local person
                        recipient_uid = self.person_id_map[
                            recipient
                        ]
                        logger.debug(
                            f"Message from {message.sender} to "
                            f"{person_cache[recipient_uid].state}:"
                        )
                        person_cache[
                            recipient_uid
                        ].receive_message(message)
                    else:  # message to remote person
                        logger.debug(
                            f"Message from {message.sender} to "
                            f"{recipient}:"
                        )

                        # get remote person ID
                        remote_person_ids.append(recipient)

                        # create message to send to other rank
                        message_to_send = message
                        message_to_send.recipients = [
                            recipient
                        ]
                        messages_to_send.append(
                            message_to_send
                        )

            else:
                logger.debug(f"Person {person.id} has no messages.")
                continue

        # Exchange messages between processors
        all_messages = self.exchange_messages(
            remote_person_ids, messages_to_send
        )

        # Step 2: Deliver messages from remote processors
        for message in all_messages:
            recipient = message.recipient
            if recipient in self.person_id_map:
                recipient_uid = self.person_id_map[recipient]
                logger.debug(
                    f"Remote message from {message.sender} to "
                    f"{person_cache[recipient_uid].state}:"
                )
                person_cache[recipient_uid].receive_message(
                    message
                )
            else:
                logger.debug(
                    f"Remote message from {message.sender} to "
                    f"{recipient} not delivered"
                )

        # Step 3: Process messages
        for person in agents:
            person.process_messages()

    def get_remote_person_id_map(
        self, remote_person_ids: list[int]
    ) -> dict[int, int]:
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

        all_messages = [
            msg for buffer in received_buffers for msg in buffer
        ]
        for person_id in all_messages:
            if person_id in self.person_id_map:
                for rank in range(self.size):
                    if rank != self.rank:
                        person_uid = self.person_id_map[
                            person_id
                        ]
                        msg = {
                            "id": person_id,
                            "uid": person_uid,
                        }
                        send_buffers[rank].append(msg)

        # 3. Send remote person ID->UID map to other ranks
        received_buffers = self.comm.alltoall(send_buffers)
        all_messages = [
            msg for buffer in received_buffers for msg in buffer
        ]
        for msg in all_messages:
            remote_person_id_map[msg["id"]] = msg["uid"]

        return remote_person_id_map

    def exchange_messages(
        self,
        remote_person_ids: list[int],
        messages_to_send: list[Message],
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

    def log_agents(self) -> None:
        """Log the agents at the current time step."""
        pass

    def get_parallel_performance_stats(self) -> dict:
        """Get performance statistics from parallel place updates."""
        if hasattr(self.places_proj, "get_parallel_performance_stats"):
            return self.places_proj.get_parallel_performance_stats()
        return {}

    def at_end(self) -> None:
        """Actions to take at the end of the simulation."""
        # Log parallel processing performance if enabled
        perf_stats = self.get_parallel_performance_stats()
        if perf_stats and self.rank == 0:
            logger.info(f"Parallel processing performance stats: {perf_stats}")
        pass

    def start(self) -> None:
        self.runner.execute()
        self.at_end()
        end_time = time.time()

        logger.info(f"Simulation took {end_time - self.start_time} seconds.")


# Register CasmPop
Models.add_model(CasmPop.__module__ + "." + CasmPop.__name__, CasmPop)


# utility functions
def update_activities_data(
    activities_data: namedtuple, **kwargs
) -> namedtuple:
    """Update the activities data."""
    return activities_data._replace(**kwargs)
