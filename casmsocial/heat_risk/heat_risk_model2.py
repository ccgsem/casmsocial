"""
Author: Jon Cline
Created: 02 Dec 2024

Defining the heat risk model for the CASMSOCIAL/PRSIM project
"""

# ruff: noqa: SIM108
import math
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import repast4py.context as ctx
from loguru import logger
from mpi4py import MPI
from pyarrow.dataset import HivePartitioning

from casmsocial.activities import Act
from casmsocial.casmpop import (
    CasmPop,
    MissingRequiredParameterError,
    MissingRequiredTableError,
    SimEnvironment,
    update_activities_data,
)
from casmsocial.data_utilities import (
    check_if_table_exists,
    get_attribute_names_from_data,
    quote_table_identifier,
)
from casmsocial.factory import Models
from casmsocial.model import Model
from casmsocial.person import BehaviorEngine, Person, PersonData
from casmsocial.place import Place, PlaceData
from casmsocial.sim_time import SimTime


# utility functions for heat-related computations
def if_place_has_cooling_center(place) -> bool:
    """Check if a place has a cooling center.

    Arguments:
        place: Place: The place to check.

    Returns:
        bool: True if the place has a cooling center, False otherwise.
    """
    has_cc = getattr(place.data, "cooling_center", False)
    return has_cc and place.data.cooling_center > 0.0


def if_place_has_air_conditioning(place) -> bool:
    """Check if a place has air conditioning.

    Arguments:
        place: Place: The place to check.

    Returns:
        bool: True if the place has air conditioning, False otherwise.
    """
    return (
        "AIR" in get_attribute_names_from_data(place.data)
        and place.data.AIR
    )


def filter_heat_indices(
    heat_indices: list[float], threshold: float
) -> list[float]:
    """Filter out all heat indices above the threshold."""
    exceeded = True
    return [
        t for t in heat_indices if (exceeded := exceeded and t > threshold)
    ]


def filter_hourly_excess_heat(
    hourly_excess_heat: list[float], threshold: float
) -> list[float]:
    """Filter out all hourly excess heat values above the threshold."""
    exceeded = True
    return [
        t for t in hourly_excess_heat
        if (exceeded := exceeded and t > threshold)
    ]


def create_closest_cooling_centers(
    conn: duckdb.DuckDBPyConnection, experiment_id: int
) -> None:
    """ """

    query = f"""
    CREATE OR REPLACE VIEW closest_cooling_center AS
    WITH experiment AS (
        SELECT
            cooling_center_ids,
            open_min,
            close_min,
            distance_threshold
        FROM experiments
        WHERE experiment_id = {experiment_id}
    ),
    cooling_centers AS (
        SELECT
            p.sp_id,
            p.latitude,
            p.longitude,
            z[2] AS open_min,
            z[3] AS close_min,
            e.distance_threshold
        FROM experiment e
        CROSS JOIN UNNEST(
            list_zip(
            e.cooling_center_ids,
            e.open_min,
            e.close_min
            )
            ) AS t(z)
        JOIN places p
          ON p.sp_id = z[1]
    ),
    distances AS (
        SELECT
            p.sp_id,
            p.location,
            p.AIR,
            p.cooling_center,
            cc.sp_id AS cooling_center_id,
            cc.open_min,
            cc.close_min,
            cc.distance_threshold,

            -- Great-circle distance in meters using DuckDB spatial
            ST_Distance_Sphere(
                ST_Point(p.latitude,  p.longitude),
                ST_Point(cc.latitude, cc.longitude)
            ) AS distance_to_cooling_center_m

        FROM places p
        CROSS JOIN cooling_centers cc
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY sp_id
                ORDER BY distance_to_cooling_center_m
            ) AS rn
        FROM distances
    )
    SELECT
        sp_id,
        location,
        AIR,
        cooling_center,
        cooling_center_id AS closest_cooling_center_id,
        open_min,
        close_min,
        distance_threshold,
        distance_to_cooling_center_m AS distance_to_closest_cooling_center_m
    FROM ranked
    WHERE rn = 1;
    """

    conn.execute(query)
    # df = conn.execute(
    #    "SELECT * FROM closest_cooling_center LIMIT 5"
    # ).df()

    # print(f"DF OF CLOSEST COOLING CENTER {df}")
    cur = conn.execute("SELECT * FROM closest_cooling_center LIMIT 5")
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

    print(f"CC COLS: {cols}")
    for row in rows:
        print(f"CC ROW: {row}")


class MissingEnvironmentFile(Exception):
    def __init__(self, value):
        """Constructor for the MissingEnvironmentFile exception."""
        super().__init__(f"Missing environment file: {value}")
        self.value = value


# 1. define the environment
class HeatRiskEnvironment(SimEnvironment):
    """HeatRiskEnvironment class"""

    def __init__(self, name: str):
        """Constructor for the HeatRiskEnvironment class."""
        super().__init__(name)

        logger.info("Initializing HeatRiskEnvironment...")

        # 1. check if the required parameters are present
        # and if the files exist
        logger.info("Checking required parameters for HeatRiskEnvironment...")
        required_keys = [
            "environment.table",
            "distance_threshold_cooling_center",
            "cooling_center_operating_hours",
            "heat_threshold_cooling_center",
            "heat_threshold_health_effect",
            "cooling_centers_experiment.table",
            "experiment_id",
            "run_log_file",
        ]
        if not all(key in Model.get_model().params for key in required_keys):
            logger.error(
                f"Error: Missing required parameters in model: {required_keys}"
            )
            raise MissingRequiredParameterError(required_keys)

        # 2. establish the connection to the database
        # this will be used to query the microweather data
        # and to update the microweather snapshot
        self.conn = Model.get_model().conn
        if self.conn is None:
            logger.error(
                "Database connection is not set. Cannot update microweather snapshot."
            )
            return

        # 3. initialize the environment: microweather data
        #    - this is the data that will be used to compute the
        #      heat index and heat-related values for each place
        environment_table = Model.get_model().params[
            "environment.table"
        ]
        if not check_if_table_exists(self.conn, environment_table):
            logger.error(
                f"Error: Environment table {environment_table} "
                f"not found in database."
            )
            raise MissingRequiredTableError(environment_table)
        logger.info(
            f"Loading environment data from {environment_table} "
            f"table in database..."
        )
        microweather_table = self.conn.execute(
            f"SELECT * FROM {quote_table_identifier(environment_table)}"
        ).fetch_arrow_table()

        # self.microweather_arrow_file_path = Model.get_model().data_path / Model.get_model().params["environment.file"]
        # if not self.microweather_arrow_file_path.exists():
        #     logger.error(f"Error: Environment file {self.microweather_arrow_file_path} not found.")
        #     raise MissingEnvironmentFile(self.microweather_arrow_file_path)
        # logger.info(f"Loading environment data from {self.microweather_arrow_file_path}...")
        # microweather_dataset = ds.dataset(self.microweather_arrow_file_path, format="parquet", partitioning="hive")
        #microweather_table = microweather_dataset.to_table()
        self.microweather_df = pl.from_arrow(microweather_table)

        # convert the time column to datetime
        self.microweather_df = self.microweather_df.with_columns(
            [
                pl.col("time").str.strptime(
                    pl.Datetime, format="%Y-%m-%dT%H:%M:%S"
                )
            ]
        )
        logger.info(
            f"Loaded microweather data with "
            f"{self.microweather_df.shape[0]} rows"
        )
        logger.info(
            f"Microweather data schema is "
            f"{self.microweather_df.schema}"
        )

        # 5. load 1-day and 2-day hourly lagged heat index data
        lagged_weather_table = Model.get_model().params["lagged_weather.table"]
        if not check_if_table_exists(self.conn, lagged_weather_table):
            logger.error(
                f"Error: Lagged weather table {lagged_weather_table} "
                f"not found in database."
            )
            raise MissingRequiredTableError(lagged_weather_table)

        # if the table exists, create a view for it so we can easily query it
        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW lagged_weather AS
            SELECT * FROM {quote_table_identifier(lagged_weather_table)};
            """
        )

        lagged_weather_df = self.conn.execute(f"SELECT * FROM lagged_weather").pl()
        logger.info(
            f"Loaded lagged weather data with "
            f"{lagged_weather_df.shape[0]} rows"
        )

        # set default heat threshold (deprecated)
        self.__heat_threshold_cooling_center = Model.get_model().params.get(
            "heat_threshold_cooling_center", 32.2222
        )  # default heat threshold in Celsius 32.2222 (90.0 Fahrenheit)

        self.__heat_threshold_health_effect = Model.get_model().params.get(
            "heat_threshold_health_effect", 32.2222
        )

        self.__degrees_f_minutes_threshold_cooling_center = (
            Model.get_model().params.get(
                "degrees_f_minutes_threshold_cooling_center", 90
            )
        )

        self.__degrees_f_minutes_threshold_health_effect = (
            Model.get_model().params.get(
                "degrees_f_minutes_threshold_health_effect", 100
            )
        )

        self.__distance_threshold_cooling_center = (
            Model.get_model().params.get(
                "distance_threshold_cooling_center", 2500
            )
        )
        self.__cooling_center_operating_hours = (
            Model.get_model().params.get(
                "cooling_center_operating_hours", False
            )
        )

        # values from the microweather data
        self.fields = [
            "T_xy",  # temperature in Celsius
            "heat_index",  # heat index in Celsius
            "dew_point",  # dew point in Celsius
            "wbgt",  # wet bulb globe temperature in Celsius
        ]
        self.environment_tuple = namedtuple("HeatRiskEnvironment", self.fields)

    @property
    def distance_threshold_cooling_center(self) -> float:
        return self.__distance_threshold_cooling_center

    @property
    def cooling_center_operating_hours(self) -> float:
        return self.__cooling_center_operating_hours

    @property
    def heat_threshold_cooling_center(self) -> float:
        return self.__heat_threshold_cooling_center

    @property
    def heat_threshold_health_effect(self) -> float:
        return self.__heat_threshold_health_effect

    @property
    def degrees_f_minutes_threshold_cooling_center(self) -> float:
        return self.__degrees_f_minutes_threshold_cooling_center

    @property
    def degrees_f_minutes_threshold_health_effect(self) -> float:
        return self.__degrees_f_minutes_threshold_health_effect

    def setup(self) -> None:
        """Setup the environment."""
        logger.info("Setting up the HeatRiskEnvironment")

        context = Model.get_model().context
        cal = Model.get_model().cal

        self.update_snapshot(context, cal)

    def update_snapshot(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the microweather snapshot.
        This method updates the microweather snapshot in the database
        with the current microweather data for the given time.
        Arguments:
            context: ctx.SharedContext: The shared context.
            cal: SimTime: The calendar.
        """
        # check if the database connection is set
        if self.conn is None:
            self.conn = Model.get_model().conn
            if self.conn is None:
                logger.error("Database connection is not set. Cannot update microweather snapshot.")
                return

        # get the current time from the calendar: needed to filter the microweather data
        current_time = cal.datetime

        # check if the microweather data is available for the current time
        logger.info(f"Updating microweather snapshot at {current_time}")
        weather_snapshot = self.microweather_df.filter(
            pl.col("time") == current_time.replace(tzinfo=None)
        )
        if weather_snapshot.is_empty() or weather_snapshot.shape[0] == 0:
            logger.error(
                f"No microweather data found for time {current_time}."
            )
            return
        logger.info(
            f"Microweather data for {current_time}: "
            f"{weather_snapshot.shape[0]} rows"
        )
        logger.info(f"\n{weather_snapshot.head(5)}")

        cur = self.conn.execute("SELECT * FROM weather_snapshot LIMIT 5")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        print(f"WEATHER SNAPSHOT cols: {cols}")
        for row in rows:
            print(f"WEATHER SNAPSHOT row: {row}")

        # convert the weather snapshot to a DuckDB table (update the weather table)
        self.conn.execute(
            """
            CREATE OR REPLACE VIEW weather AS
            SELECT
                w.time,
                w.place_id,
                w.T_xy,
                w.heat_index,
                w.dew_point,
                w.wbgt,
                cc.AIR,
                cc.cooling_center,
                cc.closest_cooling_center_id,
                cc.distance_to_closest_cooling_center_m,
                cc.open_min,
                cc.close_min,
                cc.distance_threshold
            FROM weather_snapshot AS w
            JOIN closest_cooling_center AS cc ON w.place_id = cc.sp_id; -- Join condition
            """
        )
        weather_with_cc = self.conn.execute("SELECT * FROM weather").fetchall()
        if not weather_with_cc:
            logger.error("No weather data found after updating the snapshot.")
            return
        cur = self.conn.execute("SELECT * FROM weather LIMIT 5")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        print(f"WEATHER cols: {cols}")
        for row in rows:
            print(f"WEATHER row: {row}")

        # log the updated weather data
        logger.info(f"Updated microweather snapshot with {len(weather_with_cc)} rows.")
        places_proj = context.get_projection("places_projection")
        for row in weather_with_cc:
            place_id = row[1]  # second column is place_id
            place = places_proj.lookup_place(place_id)
            if place is None:
                logger.error(f"Place with ID {place_id} not found in places projection.")
                continue
            # update the place data with the microweather data
            place.data.T_xy = row[2]  # T_xy
            place.data.heat_index = row[3]  # heat_index
            place.data.dew_point = row[4]  # dew_point
            place.data.wbgt = row[5]  # wbgt
            place.data.AIR = row[6]  # AIR
            place.data.cooling_center = row[7]  # cooling_center
            place.data.closest_cooling_center_id = row[8]
            place.data.distance_to_closest_cooling_center_m = (
                row[9]
            )  # distance to closest cooling center in meters
            place.data.open_min = row[10]
            place.data.close_min = row[11]
            place.data.distance_threshold = row[12]

        logger.info("Microweather snapshot updated successfully.")

    def teardown(self) -> None:
        """Teardown the environment."""
        pass

    def step(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Update the environment."""
        super().step(context, cal)  # updates the social environment

        # now update the physical environment
        self.update_snapshot(context, cal)

    def get_lagged_heat_index(self, hour: int):  # -> list[tuple[float, float]]:
        """Get the one and two day lagged heat index in celcius."""

        result = self.conn.execute(
            """
            SELECT HOUR, one_day_lag_heat_index_c, two_day_lag_heat_index_c
            FROM lagged_weather
            WHERE HOUR = ?
            LIMIT ?;
            """,
            (hour, 1),
        ).fetchall()

        lagged_heat_index = [(one_day_lag, two_day_lag) for _, one_day_lag, two_day_lag in result]

        for row in lagged_heat_index:
            one_day_lag = row[0]  # second column is 1 day lag
            two_day_lag = row[1]  # third column is 2 day lag
        return one_day_lag, two_day_lag

    def get_closest_cooling_center(self, place_id: int, n: int = 3) -> list[tuple[int, float]]:
        """Get the closest cooling station.

        Arguments:
            place_id: int: The ID of the place to find the closest cooling station for.
            n: int: The number of closest places to return.

        Returns:
            list[tuple[int, float]]: The closest cooling stations.
        """
        result = self.conn.execute(
            """
            SELECT cc.sp_id, cc.location, cc.cooling_center, cc.closest_cooling_center_id, distance_to_closest_cooling_center_m, cc.open_min, cc.close_min, cc.distance_threshold
            FROM places AS p
            JOIN closest_cooling_center AS cc ON p.sp_id = cc.sp_id
            WHERE p.sp_id = ?
            ORDER BY distance_to_closest_cooling_center_m ASC
            LIMIT ?;
            """,
            (place_id, n),
        ).fetchall()
        candidates = [
            (place, distance, open_min, close_min, distance_threshold)
            for place, _, _, _, distance, open_min, close_min, distance_threshold in result
        ]
        # candidates = [(place, distance) for place, _, _, distance in result]
        print(f"CANDIDATES {candidates}")
        return candidates


# 2. Define a PlaceData Class
@dataclass
class PlaceDataWithClimate(PlaceData):
    """Place with heat index data."""

    AIR: bool = False
    cooling_center: float = 0.0  # bool = False
    T_xy: float = float("nan")  # temperature in Celsius
    heat_index: float = float("nan")  # heat index in Celsius
    dew_point: float = float("nan")  # dew point in Celsius
    wbgt: float = float("nan")  # wet bulb globe temperature in Celsius
    closest_cooling_center_id: int = None  # ID of the closest cooling center
    distance_to_closest_cooling_center_m: float = float("nan")  # distance to the closest cooling center in meters
    open_min: int = None  # time the cooling center opens in minutes
    close_min: int = None  # time the cooling center closes in minutes
    distance_threshold: int = None  # distance willing to travel to cooling center


# 3. Define a PersonData Class with heat risk data
@dataclass
class PersonDataWithHeatRisk(PersonData):
    """Data for a Person."""

    # previous_activities_idx: int = None
    outside_worker: bool = False
    outdoor_heat_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    outdoor_temp_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    outdoor_wbgt_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    outdoor_dew_point_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    heat_indices: deque = field(default_factory=lambda: deque([float("nan")]))
    hourly_excess_heat_cooling_center: deque = field(default_factory=lambda: deque([float("nan")]))
    hourly_excess_heat_health_effect: deque = field(default_factory=lambda: deque([float("nan")]))
    hours_with_heh_cooling_center: int = 0
    hours_with_heh_health_effect: int = 0
    max_hours_with_heh_cooling_center: int = 0
    max_hours_with_heh_health_effect: int = 0
    prob_heat_event: float = 0.0
    experienced_heat_event: bool = False
    moved_to_cooling_center: bool = False  # current state
    ever_moved_to_cooling_center: bool = False  # history flag
    cooling_center_visits: int = 0
    heat_event_place_id: int = None
    cooling_center_id: int = None
    outdoors: bool = False
    has_ac_access: bool = False
    f_minutes_cumulative_cooling_center: float = 0.0  # total exposure so far
    f_minutes_history_cooling_center: deque[float] = field(
        default_factory=lambda: deque(
            maxlen=None
        )  # keep **all** ticks; change maxlen if you only need a sliding window
    )
    f_minutes_cumulative_health_effect: float = 0.0  # total exposure so far
    f_minutes_history_health_effect: deque[float] = field(
        default_factory=lambda: deque(
            maxlen=None
        )  # keep **all** ticks; change maxlen if you only need a sliding window
    )
    heat_intensity: deque[float] = field(
        default_factory=lambda: deque([float("nan")])  # will grow one value per hour
    )
    dehh_cumulative_cooling_center: float = 0.0  # Daily Excess Hourly Heat (°F·h) for the “cooling center” threshold
    dehh_cumulative_health_effect: float = 0.0  # Daily Excess Hourly Heat for the “health effect” threshold
    # a deque that will hold ONE dict per completed hour
    hourly_log: deque[dict] = field(default_factory=lambda: deque())


# 4. Define a HeatRiskBehaviorEngine Class
class HeatRiskBehaviorEngine(BehaviorEngine):
    """HeatRiskBehaviorEngine class

    This class defines the behavior engine for a person in the heat risk model.

    After the model has a probability of a heat event, the person decides to seek cooling.
    - 1) prob
    - 2) knowing the probability of
    """

    def __init__(self, person: Person):
        """Constructor for the HeatRiskBehaviorEngine class."""
        super().__init__(person)

    def _finalise_hourly_record(self, person: Person, hour_of_day: int) -> None:
        """
        Build a dict that contains all the fields required for the
        *hourly* person log, push it onto the agent's `hourly_log` deque,
        and reset the per hour temporary trackers.
        """
        # ----- 1) HEH values (they are stored in the *hourly* deques) -----
        heh_cc = person.state.hourly_excess_heat_cooling_center[0]  # °F min, most recent hour
        heh_he = person.state.hourly_excess_heat_health_effect[0]  # °F min, most recent hour

        # ----- 2) Heat index statistics for the hour --------------------
        # The four most recent values belong to the hour that just ended.
        recent_heat_idxs = list(person.state.heat_indices)[:4]  # newest first
        # Guard against NaNs (they appear at the very beginning of the run)
        clean_vals = [v for v in recent_heat_idxs if not math.isnan(v)]

        # average (if we have any real values)
        avg_heat = float(sum(clean_vals) / len(clean_vals)) if clean_vals else float("nan")
        # max intensity (same as what you already store in `heat_intensity`)
        max_intensity = max(clean_vals) if clean_vals else float("nan")

        # The intensity for the hour was already computed by
        # `_maybe_update_heat_intensity` and lives in the deque
        hour_intensity = person.state.heat_intensity[0] if person.state.heat_intensity else float("nan")

        # Build the record
        record = {
            "hour_of_day": hour_of_day,
            "agent_id": person.id,
            "hourlyExcessHeatCoolingCenter": heh_cc,
            "hourlyExcessHeatHealthEffect": heh_he,
            "averageHeatIndex": avg_heat,
            "maxHeatIndex": max_intensity,  # should be the same as heatIntensity but can check
            "heatIntensity": hour_intensity,
        }

        # Store the record
        person.state.hourly_log.appendleft(record)

        # Reset the temporary per hour flags for the next hour
        person.state.hourly_moved_to_cooling_center = False
        person.state.hourly_outdoors = False
        person.state.hourly_has_ac_access = False
        # (no need to reset `hourly_intensity_max` now because we don't use it)

    # ------------------------------------------------------------------
    # Helper classes for decide function
    # ------------------------------------------------------------------
    # 1. Guard clauses: early exit checks
    # ------------------------------------------------------------------
    def _guard_missing_place(self, place: Place) -> bool:
        """Return True if place exists and has temperature data,
        False otherwise (caller should `return` immediately)."""
        if not place:
            logger.error(
                f"Person {self.agent.id} does not have a place"
            )
            return False
        if place.data.T_xy == float("nan"):
            logger.error(
                f"Person {self.agent.id} has no temperature data at "
                f"place {place.id}"
            )
            return False
        return True

    def _handle_already_experienced_event(self, place: Place) -> bool:
        """If the agent already suffered a heat event, push a “cooled” index
        and return True so the caller can exit early."""
        if self.agent.state.experienced_heat_event:
            logger.debug(f"Person {self.agent.id} has already experienced a heat event")
            # ---- compute the cooled heat index (same formula used later) ----
            outdoor_c = (place.data.heat_index - 32) * 5 / 9
            local_c = 23 + 0.1 * outdoor_c
            local_f = local_c * 1.8 + 32

            p = self.agent
            p.state.heat_indices.appendleft(local_f)
            p.state.outdoor_heat_indices.appendleft(place.data.heat_index)
            p.state.outdoor_wbgt_indices.appendleft(place.data.wbgt)
            p.state.outdoor_dew_point_indices.appendleft(place.data.dew_point)
            p.state.outdoor_temp_indices.appendleft(place.data.T_xy)
            return True
        return False

    def _next_place_has_ac(self, time: int, context) -> bool:
        """Check if next scheduled activity has AC.
        Helper used by `_maybe_return_from_cooling_center`.
        Checks *next* activity (at `time`) and returns True
        if that place has air conditioning."""
        schedule_names = [s.name for s in self.agent.schedules.schedules]
        weekday_idx = schedule_names.index("weekday")
        act = self.agent.schedules[weekday_idx].activityAt(time)
        next_place_id = act.place_id
        places_proj = context.get_projection("places_projection")
        next_place = places_proj.lookup_place(next_place_id)
        # places_proj = context.get_projection("places_projection")
        # next_place = context.get_projection("places_projection").lookup_place(act.place_id)
        return bool(next_place and next_place.data.AIR)

    def _push_cooled_index(self, place: Place) -> None:
        """Push the same “cooled” heat index that we use when a heat event
        has already been experienced."""
        outdoor_c = (place.data.heat_index - 32) * 5 / 9
        local_c = 23 + 0.1 * outdoor_c
        local_f = local_c * 1.8 + 32

        p = self.agent
        p.state.heat_indices.appendleft(local_f)
        p.state.outdoor_heat_indices.appendleft(place.data.heat_index)
        p.state.outdoor_wbgt_indices.appendleft(place.data.wbgt)
        p.state.outdoor_dew_point_indices.appendleft(place.data.dew_point)
        p.state.outdoor_temp_indices.appendleft(place.data.T_xy)

        fmin_this_tick_cc, fmin_this_tick_he = self.compute_f_minutes(
            temp_f=local_f,
            env=Model.get_model().get_environment(),
            step_minutes=15,
        )

        # Update the per agent containers
        p.state.f_minutes_cumulative_cooling_center += fmin_this_tick_cc
        p.state.f_minutes_history_cooling_center.append(fmin_this_tick_cc)
        p.state.f_minutes_cumulative_health_effect += fmin_this_tick_he
        p.state.f_minutes_history_health_effect.append(fmin_this_tick_he)

    def _maybe_return_from_cooling_center(
        self, place: Place, time: int, env: HeatRiskEnvironment, context
    ) -> bool:
        """Return True if method handled the situation.
        Agent stays in cooling centre or returns to regular schedule
        and caller should `return` immediately.
        """
        if not self.agent.state.moved_to_cooling_center:
            return False

        logger.debug(
            f"Person {self.agent.id} is currently in a cooling centre"
        )
        closes = place.data.close_min

        # Exit condition: centre closed **or** next scheduled place has AC
        if (
            (env.cooling_center_operating_hours and time >= closes)
            or self._next_place_has_ac(time, context)
        ):
            self.return_to_regular_schedule()
            return False
        else:
            # stay in centre - push a “cooled” index
            self._push_cooled_index(place)
            return True
        # return True

    # ------------------------------------------------------------------
    # 2. Heat index computation
    # ------------------------------------------------------------------
    def _compute_local_heat_index(
        self,
        place: Place,
        env: HeatRiskEnvironment,
        cal: SimTime,
    ) -> float:
        """Return perceived heat index (°F) for current tick.
        Logic mirrors three branches:
        * outdoor worker → raw outdoor value
        * indoor with AC → 23°C + 0.1·outdoor
        * indoor without AC → lagged heat index formula
        """
        # Adjust heat index based on person's situation:
        # 1. Indoors with AC: apply Nguyen et al. (2014),
        #    Quinn et al. (2017)
        #    indoor_heat_index_c = 23 + 0.1 * outdoor_heat_index_c
        # 2. Indoors without AC: apply Quinn et al. (2014)
        #    indoor_heat_index_c = 27.00 +
        #    0.24⋅outdoor_heat_index_c(t) +
        #    0.076⋅outdoor_heat_index_c(t-1) -
        #    0.016⋅outdoor_heat_index_c(t-2)
        # 3. At work as outside worker: apply outdoor_heat_index

        outdoor_f = place.data.heat_index  # already °C
        outdoor_c = (outdoor_f - 32) * 5 / 9  # convert to °C for math

        # 1. Outside worker (already flagged earlier)
        if self.agent.state.outdoors:
            return place.data.heat_index  # keep original °F value

        # 2. Inside **and** has AC → simple AC correction
        if self.agent.state.has_ac_access:
            indoor_c = 23 + 0.1 * outdoor_c
            return indoor_c * 1.8 + 32

        # 3. Inside **without** AC → lagged weather model
        hour = cal.hour_of_day
        one_day_lag, two_day_lag = env.get_lagged_heat_index(hour)
        indoor_c = 27.0 + 0.24 * outdoor_c + 0.076 * one_day_lag - 0.016 * two_day_lag
        return indoor_c * 1.8 + 32

    # ------------------------------------------------------------------
    # 2c. Hourly Excess Heat (HEH) bookkeeping
    # ------------------------------------------------------------------
    def _maybe_update_heh_counters(
        self, cal: SimTime, env: HeatRiskEnvironment
    ) -> None:
        """Update per hour HEH counters at minute 45.
        See Tang et al. (2021) for effects of different heat
        exposure patterns on hospitalizations.
        """
        if cal.minute_of_day % 60 != 45:
            return

        hrs_cc, hrs_he = self.get_hours_with_heh(
            env.heat_threshold_cooling_center,
            env.heat_threshold_health_effect,
        )
        self.agent.state.hours_with_heh_cooling_center = hrs_cc
        self.agent.state.hours_with_heh_health_effect = hrs_he

    def _maybe_update_heat_intensity(self, cal: SimTime) -> None:
        """At end of hour (minute 45) compute max perceived
        heat index for just finished hour and store it.
        """
        # Act once per hour when HEH routine fires
        # minute 45 == last tick of hour (15 min steps)
        if cal.minute_of_day % 60 != 45:
            return

        person = self.agent

        # 1) Grab last 4 perceived heat index values (current hour)
        recent_vals = list(person.state.heat_indices)[:4]
        # 2) Convert NaNs to -inf so they don't break max()
        recent_vals = [
            v if not math.isnan(v) else float("-inf")
            for v in recent_vals
        ]
        # 3) Compute the maximum (hourly intensity)
        hour_intensity = (
            max(recent_vals) if recent_vals else float("nan")
        )

        # 4) Store in per agent deque
        person.state.heat_intensity.appendleft(hour_intensity)

    def compute_f_minutes(
        self,
        temp_f: float,
        env: HeatRiskEnvironment,
        step_minutes: int = 15,
    ) -> float:
        # Pull threshold from model parameters (default 100°F)
        threshold_cc = (
            env.degrees_f_minutes_threshold_cooling_center
        )
        threshold_he = (
            env.degrees_f_minutes_threshold_health_effect
        )

        fmin_cc = 0
        fmin_he = 0

        if temp_f <= threshold_cc:
            fmin_cc = 0.0
        else:
            fmin_cc = (temp_f - threshold_cc) * step_minutes

        if temp_f <= threshold_he:
            fmin_he = 0
        else:
            fmin_he = (temp_f - threshold_he) * step_minutes

        return fmin_cc, fmin_he

    def compute_hourly_excess_heat(self, threshold: float):
        """Compute the Hourly Excess Heat (HEH)"""
        # HEH is sum of 5-min deviations above 100F within an hour
        # As defined in Seong et al., 2023
        # 15-min increments multiplied by 4
        person = self.agent
        heat_indices = person.state.heat_indices

        # Get heat indices for the past hour
        hourly_heat_indices = list(heat_indices)[:4]
        compare_to_threshold = [
            val - threshold if val > threshold else 0
            for val in hourly_heat_indices
        ]
        # Sum values between zero values
        segments = []
        current_sum = 0
        # If heat index > threshold, subtract from each index
        # Sum values within past hour between consecutive positives
        for val in compare_to_threshold:
            if val == 0:
                if current_sum > 0:  # If there's an accumulated sum, store it
                    segments.append(current_sum)
                    current_sum = 0  # Reset the sum for the next segment
            else:
                current_sum += val  # Accumulate the sum

        # Add the last segment if there's any remaining sum
        if current_sum > 0:
            segments.append(current_sum)

        # If no segments identified, set to [0]
        if not segments:
            segments.append(0)

        # Highest consecutive value x3 is HEH for hour
        # Multiply by 3: original study used 5-min intervals,
        # ours uses 15-min intervals
        heh = max(segments) * 3

        # Store HEH
        return heh

    def _heh_min_to_hour(self, heh_min: float) -> float:
        """Convert HEH from °F minutes to °F hours."""
        return heh_min / 60.0  # 60 min per hour

    def _update_daily_dehh(
        self, cal: SimTime, env: HeatRiskEnvironment
    ) -> None:
        """Called once per hour (minute 45).
        Adds just completed hour's HEH (converted to °F hour)
        to running daily DEHH total.
        """
        # Sanity check: only run at hour end
        if cal.minute_of_day % 60 != 45:
            return

        person = self.agent

        # 1) Fetch latest HEH values (appended in
        #    get_hours_with_heh)
        latest_heh_cc = (
            person.state.hourly_excess_heat_cooling_center[0]
        )  # °F min
        latest_heh_he = (
            person.state.hourly_excess_heat_health_effect[0]
        )  # °F min

        # 2) Convert to °F hour
        heh_cc_hour = latest_heh_cc / 60.0
        heh_he_hour = latest_heh_he / 60.0

        # 3) Accumulate
        person.state.dehh_cumulative_cooling_center += heh_cc_hour
        person.state.dehh_cumulative_health_effect += heh_he_hour

    def get_hours_with_heh(
        self: float,
        threshold_cooling_center: float,
        threshold_health_effect: float,
    ) -> float:
        # Update HEH for current hour
        person = self.agent
        heh_cooling_center = self.compute_hourly_excess_heat(threshold_cooling_center)
        heh_health_effect = self.compute_hourly_excess_heat(threshold_health_effect)

        person.state.hourly_excess_heat_cooling_center.appendleft(heh_cooling_center)
        person.state.hourly_excess_heat_health_effect.appendleft(heh_health_effect)

        # Hours with HEH above cooling center threshold
        all_heh_cc = person.state.hourly_excess_heat_cooling_center
        positive_heh_cc = filter_hourly_excess_heat(all_heh_cc, 0)
        hours_with_heh_cooling_center = len(positive_heh_cc)

        # Hours with HEH dangerous for health effects
        all_heh_health = person.state.hourly_excess_heat_health_effect
        positive_heh_health = filter_hourly_excess_heat(
            all_heh_health, 0
        )
        hours_with_heh_health_effect = len(positive_heh_health)

        # Update max values
        person.state.max_hours_with_heh_cooling_center = max(
            person.state.max_hours_with_heh_cooling_center,
            hours_with_heh_cooling_center,
        )
        person.state.max_hours_with_heh_health_effect = max(
            person.state.max_hours_with_heh_health_effect,
            hours_with_heh_health_effect,
        )

        return (
            hours_with_heh_cooling_center,
            hours_with_heh_health_effect,
        )

    def compute_heh_minutes(self):
        return

    def compute_prob_heat_event(self, threshold: float) -> float:
        """Compute the probability of a heat event."""
        heat_indices = self.agent.state.heat_indices

        # Filter out all heat indices above threshold
        heat_index = heat_indices[0]
        heat = filter_heat_indices(heat_indices, threshold)
        hours_above_threshold = len(heat)

        # Length of heat is number of hours above threshold
        prob_heat_event = (
            1
            - (
                1
                - ((heat_index - threshold) / 80.0) ** 2
            )
            ** (3 * hours_above_threshold)
        )
        # For deterministic model, will just use hours above certain heat index to determine if
        ## agent will seek a cooling center
        return prob_heat_event

    def decide_to_seek_cooling(
        self, place: Place, cal: SimTime
    ) -> bool:
        """Decide to seek cooling.
        Heat event probability already computed,
        person has not experienced heat event.
        """

        person = self.agent
        environment = Model.get_model().get_environment()
        time = cal.minute_of_day
        disable_incidents = Model.get_model().params.get(
            "disable_heat_health_incidents", False
        )

        # Consider seeking cooling if exposed to heat threshold
        # over certain hours. 15-min increments: 12 = 3 hours
        heh_hours_cooling_center = (
            person.state.hours_with_heh_cooling_center
        )
        heh_hours_health_effect = (
            person.state.hours_with_heh_health_effect
        )
        consider_seeking_cooling = heh_hours_cooling_center > 2

        have_health_effect = (
            heh_hours_health_effect > 2
            or heh_hours_cooling_center > 4
        )
        # Hard-disable health incidents
        if disable_incidents:
            have_health_effect = False
            person.state.experienced_heat_event = False
            person.state.heat_event_place_id = None

        if not consider_seeking_cooling:
            return False

        # 1) if hourly excess heat is too high, experience an adverse health effect
        if have_health_effect:
            person.state.experienced_heat_event = True
            person.state.heat_event_place_id = place.id
            return False

        # 2) If no health effect, find closest cooling center
        cooling_center_candidate_id = (
            place.data.closest_cooling_center_id
        )
        closest_cooling_center_distance = (
            place.data.distance_to_closest_cooling_center_m
        )
        cooling_center_opens = place.data.open_min
        cooling_center_closes = place.data.close_min
        cooling_center_distance_threshold = (
            place.data.distance_threshold
        )
        cooling_center_above_distance_threshold = (
            closest_cooling_center_distance
            > cooling_center_distance_threshold
        )

        # Check if cooling center is open
        is_open = True
        if environment.cooling_center_operating_hours:
            is_open = (
                time <= cooling_center_closes
                and time >= cooling_center_opens
            )

        # No cooling centers or too far or closed =
        # stay at current location
        if (
            cooling_center_candidate_id is None
            or cooling_center_above_distance_threshold
        ):
            cooling_center_candidate_id = place.id
            return False

        # Close cooling center but closed = stay at current
        if not is_open:
            cooling_center_candidate_id = place.id
            return False

        # 3) No health effect but excess heat high =
        # decide to seek cooling center
        seeking_cooling = consider_seeking_cooling
        if seeking_cooling:
            self.move_to_cooling_center(
                cooling_center_candidate_id,
                cal.hour_of_day,
                cooling_center_closes,
            )
            return True

        return False

    def move_to_cooling_center(
        self, place_id: int, current_hour: int, time_to_move: int
    ) -> None:
        """Move the person to a cooling center."""
        person = self.agent

        # find the next hour
        next_hour = current_hour + 1
        start_time = next_hour * 60

        # currently set end time to the end of the day
        end_time = time_to_move
        # end_time = 1440  # 24 hours

        # find the activity and schedule indices
        activity_names = CasmPop.get_activity_names()
        activity_id = activity_names.index("cooling_center")

        if person.state.moved_to_cooling_center:
            logger.error(
                f"Person {person.id} already moved to cooling center"
            )
            return

        person.state.moved_to_cooling_center = True
        person.state.ever_moved_to_cooling_center = True
        person.state.cooling_center_id = place_id

        schedule_names = [
            schedule.name for schedule in person.schedules.schedules
        ]
        schedule_idx = schedule_names.index("cooling_center")
        if self.agent.state.activities_idx == schedule_idx:
            logger.error(
                f"Person {person.id} already in cooling center schedule"
            )
            return

        # Modify person's activities to include cooling center
        activities_data = person.state.places
        person.state.places = update_activities_data(
            activities_data, cooling_center=place_id
        )

        act_go_to_cooling_center = Act(
            person.id, activity_id, 1.0, start_time, end_time, place_id
        )

        # Add activity to schedule
        person.schedules.schedules[schedule_idx].addAct(
            act_go_to_cooling_center
        )
        logger.debug(
            f"Person {person.id} updated schedule to move to "
            f"cooling center {place_id}"
        )

        previous_schedule = person.state.activities_idx
        person.state.activities_idx = schedule_idx

        for act in person.schedules.schedules[schedule_idx].acts:
            logger.debug(f"  {act}")
        logger.debug(
            f"Person {person.id} moved to schedule {schedule_idx} "
            f"from {previous_schedule}"
        )
        person.state.cooling_center_visits += 1

    def return_to_regular_schedule(self):
        person = self.agent

        schedule_names = [
            schedule.name for schedule in person.schedules.schedules
        ]
        weekday_idx = schedule_names.index("weekday")
        person.state.activities_idx = weekday_idx

        person.state.moved_to_cooling_center = False
        person.state.cooling_center_id = None

        logger.debug(
            f"Person {person.id} returned to regular schedule"
        )

    def decide(self, context: ctx.SharedContext, cal: SimTime) -> None:
        """Decide what to do."""
        # Simulation runs in 15 min steps, minute 45 is last tick
        # of hour. At minute 0 of next hour, finalise record.
        p = self.agent
        if cal.minute_of_day % 60 == 0:  # entered new hour
            # Hour just ended is `cal.hour_of_day - 1` (wrap at 0)
            finished_hour = (cal.hour_of_day - 1) % 24
            self._finalise_hourly_record(p, finished_hour)

        # Enforce "always false" every tick
        disable_heat_health_incidents = (
            Model.get_model().params.get(
                "disable_heat_health_incidents", False
            )
        )
        if disable_heat_health_incidents:
            self.agent.state.experienced_heat_event = False
            self.agent.state.heat_event_place_id = None

        # get the heat index at the person's location
        places_proj = context.get_projection("places_projection")
        place = places_proj.lookup_place(self.agent.currentPlaceID)

        # ---------- 1. Guard clauses ----------
        if not self._guard_missing_place(place):
            return
        if self._handle_already_experienced_event(place):
            return
        if self._maybe_return_from_cooling_center(
            place, cal.minute_of_day, Model.get_model().get_environment(), context
        ):
            return

        # ---------- 2. Determine outdoor/AC flags ----------
        activity_names = CasmPop.get_activity_names()
        work_id = activity_names.index("work")
        act = self.agent.schedules[
            self.agent.state.activities_idx
        ].activityAt(cal.minute_of_day)
        self.agent.state.outdoors = (
            int(act.activity_id) == work_id
            and self.agent.state.outside_worker
        )
        self.agent.state.has_ac_access = (
            bool(place.data.AIR)
            and not self.agent.state.outdoors
        )

        # ---------- 3. Compute perceived heat index ----------
        local_heat_f = self._compute_local_heat_index(
            place, Model.get_model().get_environment(), cal
        )

        p.state.heat_indices.appendleft(local_heat_f)
        p.state.outdoor_heat_indices.appendleft(
            place.data.heat_index
        )
        p.state.outdoor_wbgt_indices.appendleft(
            place.data.wbgt
        )
        p.state.outdoor_dew_point_indices.appendleft(
            place.data.dew_point
        )
        p.state.outdoor_temp_indices.appendleft(
            place.data.T_xy
        )

        # ---------- 4. Compute degrees f-minutes ----------
        fmin_this_tick_cc, fmin_this_tick_he = (
            self.compute_f_minutes(
                temp_f=local_heat_f,
                env=Model.get_model().get_environment(),
                step_minutes=15,
            )
        )

        # Update the per agent containers
        p.state.f_minutes_cumulative_cooling_center += fmin_this_tick_cc
        p.state.f_minutes_history_cooling_center.append(fmin_this_tick_cc)
        p.state.f_minutes_cumulative_health_effect += fmin_this_tick_he
        p.state.f_minutes_history_health_effect.append(fmin_this_tick_he)

        # ---------- 5. Update HEH counters once per hour ----------
        self._maybe_update_heh_counters(
            cal, Model.get_model().get_environment()
        )

        # ---------- 6. Update Heat Intensity once per hour ----------
        self._maybe_update_heat_intensity(cal)

        # ---------- 7. Update daily DEHH ----------
        self._update_daily_dehh(
            cal, Model.get_model().get_environment()
        )

        # 8. Cooling centre subengine
        if self.decide_to_seek_cooling(place, cal):
            p.state.heat_event_place_id = place.id


# 6a. Define agent log data
@dataclass
class PersonLogData:
    """Data for logging person agent information."""

    Imputation: int
    experiment_id: int
    minute_of_day: int
    hour_of_day: int
    rank: int  # rank of the agent in the MPI communicator
    agent_id: int
    x: float
    y: float
    place_id: int
    outdoorHeatIndex: float
    outdoorWBGT: float
    outdoorDewPoint: float
    outdoorTemperature: float
    heatIndex: float
    hourlyExcessHeatCoolingCenter: float
    hourlyExcessHeatHealthEffect: float
    hrsWithHourlyExcessHeatCoolingCenter: int
    hrsWithHourlyExcessHeatHealthEffect: int
    experiencedHeatEvent: bool
    movedToCoolingCenter: bool
    outdoors: bool
    hasACAccess: bool
    heatEventPlaceId: int
    coolingCenterId: int
    fMinutesCumulativeCoolingCenter: float  # total exposure up to this tick
    fMinutesCumulativeHealthEffect: float
    fMinutesThisTickCoolingCenter: float  # contribution of the *current* tick
    fMinutesThisTickHealthEffect: float
    heatIntensity: float = float("nan")  # intensity of the *current* hour
    dehhCoolingCenter: float = 0.0  # DEHH for the *current* day (°F·h)
    dehhHealthEffect: float = 0.0  # DEHH for health effect threshold

    # Hourly log fields
    hourOfDay: int = -1  # the hour this row belongs to
    placeIdAtHourEnd: int = -1  # agent's place at the end of the hour
    movedToCoolingCenterHour: bool = False
    outdoorsHour: bool = False
    hasACAccessHour: bool = False
    heatIntensityHour: float = float("nan")
    fMinutesThisTickCoolingCenterHour: float = 0.0
    fMinutesThisTickHealthEffectHour: float = 0.0

    # End of run log fields
    homePlaceId: int = -1
    everMovedToCoolingCenterRun: bool = False
    everOutdoorsRun: bool = False
    hasACAccessHome: bool = False
    maxHeatIntensityRun: float = float("-inf")
    dehhCoolingCenterRun: float = 0.0
    dehhHealthEffectRun: float = 0.0


class PersonRunLogData:
    """One row per agent written at the very end of the simulation."""

    agent_id: int
    hrsWithhourlyExcessHeatCoolingCenter: int
    hrsWithhourlyExcessHeatHealthEffect: int
    fMinutesCumulativeCoolingCenter: float
    fMinutesCumulativeHealthEffect: float
    averageHeatIndex: float  # mean of all *per tick* heat index values (°F)
    averageHeatIntensity: float  # mean of the hourly intensity values (°F)
    maxHeatIntensity: float  # highest intensity ever observed (°F)
    dehhCoolingCenter: float  # total daily excess hourly heat (°F·h) for cooling centre
    dehhHealthEffect: float  # same for health effect threshold


# 6a. Define run log data
@dataclass
class RunLogData:
    Imputation: int
    experiment_id: int
    countAgents: int
    totalHeatIndex: float
    totalHrsWithHEHCoolingCenter: int
    totalHrsWithHEHHealthEffect: int
    countMovedToCoolingCenter: int
    countExperiencedHealthEffect: int
    countVisitsToCoolingCenter: int
    totalFMinutesCoolingCenter: float = 0.0
    totalFMinutesHealthEffect: float = 0.0
    totalHeatIntensity: float = 0.0
    maxHeatIntensity: float = 0.0
    totalDEHHCoolingCenter: float = 0.0
    totalDEHHHealthEffect: float = 0.0


# 6b. Define the HeatRiskModel class
class HeatRiskModel2(CasmPop):
    """HeatRiskModel class"""

    @classmethod
    def get_default_parameters(cls) -> dict:
        """Get the default parameters for the ArtSocModel."""
        params = super().get_default_parameters()
        params.update(
            {
                "model.name": cls.__module__ + "." + cls.__name__,
                "write_agent_tick_log": True,
                "write_run_log": True,
                "write_agent_hourly_log": False,
                "write_end_run_person_log": False,
            }
        )
        return params

    def __init__(self, comm: MPI.Intracomm, params: dict):
        """Constructor for the HeatRiskModel class"""
        super().__init__(comm, params)

        # ----------------------------
        # Output toggles (defaults)
        # ----------------------------
        self.write_agent_tick_log = bool(self.params.get("write_agent_tick_log", True))
        self.write_run_log = bool(self.params.get("write_run_log", True))
        self.write_agent_hourly_log = bool(self.params.get("write_agent_hourly_log", False))
        self.write_end_run_person_log = bool(self.params.get("write_end_run_person_log", False))

        # ----------------------------
        # Paths created only if needed
        # ----------------------------
        self.agent_log_file = None
        self.agent_hourly_log_file = None
        self.run_log_file = None
        self.run_person_log_file = None

        # create the agent log file path
        self.write_agent_tick_log = bool(self.params.get("write_agent_tick_log", True))
        self.write_run_log = bool(self.params.get("write_run_log", True))
        self.write_agent_hourly_log = bool(self.params.get("write_agent_hourly_log", False))
        self.write_end_run_person_log = bool(self.params.get("write_end_run_person_log", False))

        self.agent_log_file = self._optional_log_path(
            self.write_agent_tick_log, "agent_log_file"
        )
        self.agent_hourly_log_file = self._optional_log_path(
            self.write_agent_hourly_log, "agent_hourly_log_file"
        )
        self.run_log_file = self._optional_log_path(
            self.write_run_log, "run_log_file"
        )
        self.run_person_log_file = self._optional_log_path(
            self.write_end_run_person_log, "run_person_log_file"
        )

        logger.info(f"HeatRiskModel2 initialized at time={time.time() - self.start_time} seconds")

        # if self.write_agent_tick_log:
        #    if "agent_log_file" not in self.params:
        #        raise MissingRequiredParameterError(["agent_log_file"])
        #    self.agent_log_file = self.data_path / self.params["agent_log_file"]
        #    if not self.agent_log_file.parent.exists():
        #        self.agent_log_file.parent.mkdir(parents=True, exist_ok=True)

        # create the agent log file path
        # if self.write_agent_hourly_log:
        #    if "agent_hourly_log_file" not in self.params:
        #       raise MissingRequiredParameterError(["agent_hourly_log_file"])
        #    self.agent_hourly_log_file = self.data_path / self.params["agent_hourly_log_file"]
        #    if not self.agent_hourly_log_file.parent.exists():
        #        self.agent_hourly_log_file.parent.mkdir(parents=True, exist_ok=True)

        # create the run log file path
        # if self.write_run_log:
        #    if "run_log_file" not in self.params:
        #        raise MissingRequiredParameterError(["run_log_file"])
        #    self.run_log_file = self.data_path / self.params["run_log_file"]
        #    if not self.run_log_file.parent.exists():
        #        self.run_log_file.parent.mkdir(parents=True, exist_ok=True)

        # if self.write_end_run_person_log:
        #    if "run_person_log_file" not in self.params:
        #        raise MissingRequiredParameterError(["run_person_log_file"])
        #    self.run_person_log_file = self.data_path / self.params["run_person_log_file"]
        #    if not self.run_person_log_file.parent.exists():
        #        self.run_person_log_file.parent.mkdir(parents=True, exist_ok=True)

        # show the initialization time
        # logger.info(f"HeatRiskModel2 initialized at time={time.time() - self.start_time} seconds")

    def _optional_log_path(self, enabled: bool, param_name: str):
        if not enabled:
            return None
        if param_name not in self.params:
            raise MissingRequiredParameterError([param_name])

        path = self.data_path / self.params[param_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def heat_threshold_cooling_center(self) -> float:
        return self._heat_threshold_cooling_center

    def heat_threshold_health_effect(self) -> float:
        return self._heat_threshold_health_effect

    def degrees_f_minutes_cooling_center(self) -> float:
        return self._degrees_f_minutes_cooling_center

    def degrees_f_minutes_health_effect(self) -> float:
        return self._degrees_f_minutes_health_effect

    def build_context(self) -> None:
        """Initialize population"""

        # register the environment
        logger.info(
            f"Registering HeatRiskEnvironment at "
            f"time={time.time()-self.start_time} seconds..."
        )
        CasmPop.register_environment(
            HeatRiskEnvironment("HeatRiskEnvironment")
        )
        logger.info(
            f"HeatRiskEnvironment registered at "
            f"time={time.time()-self.start_time} seconds."
        )

        # register the person and place agent types
        logger.info(
            f"Registering person type (TYPE={Person.TYPE})..."
        )
        CasmPop.setPersonClass(Person, PersonDataWithHeatRisk)

        logger.info(
            f"Registering place type (TYPE={Place.TYPE})..."
        )
        CasmPop.setPlaceClass(Place, PlaceDataWithClimate)

        # register the person behavior engine
        Person.registerBehaviorEngine(HeatRiskBehaviorEngine)

        # register the activities
        CasmPop.register_planned_activity_names(
            ["sp_hh_id", "sp_work_id", "sp_school_id"]
        )
        CasmPop.register_activity_names(
            ["home", "work", "school", "cooling_center"]
        )

        logger.debug("Now running initialize population for CasmPop...")

        super().build_context()
        logger.info("Population initialized for HeatRiskModel2")

        # set up the environment
        environment = self.get_environment()
        self._cooling_center_operating_hours = (
            environment.cooling_center_operating_hours
        )
        self._distance_threshold_cooling_center = (
            environment.distance_threshold_cooling_center
        )
        self._heat_threshold_cooling_center = (
            environment.heat_threshold_cooling_center
        )
        self._heat_threshold_health_effect = (
            environment.heat_threshold_health_effect
        )
        self._degrees_f_minutes_threshold_cooling_center = (
            environment.degrees_f_minutes_threshold_cooling_center
        )
        self._degrees_f_minutes_threshold_health_effect = (
            environment.degrees_f_minutes_threshold_cooling_center
        )

        # Need to sum heat index across all ticks; initialize to 0
        self.total_heat_index = 0

        logger.info(
            f"Boolean for using cooling center operating hours is "
            f"{self._cooling_center_operating_hours}C"
        )
        logger.info(
            f"Distance threshold set to go to a cooling center is "
            f"{self._distance_threshold_cooling_center}C"
        )
        logger.info(
            f"Heat threshold set to seek a cooling center is "
            f"{self._heat_threshold_cooling_center}C"
        )
        logger.info(
            f"Heat threshold set to to have a health effect is "
            f"{self._heat_threshold_health_effect}C"
        )
        logger.info(
            f"Degrees F-minutes set to seek a cooling center is "
            f"{self._degrees_f_minutes_threshold_cooling_center}C"
        )
        logger.info(
            f"Degrees F-minutes set to to have a health effect is "
            f"{self._degrees_f_minutes_threshold_health_effect}C"
        )

        # check the first agent
        person = next(self.context.agents())
        logger.info(person)
        candidates = environment.get_closest_cooling_center(
            person.currentPlaceID, 3
        )
        if not candidates:
            logger.error(
                f"No cooling stations found for person {person.id} "
                f"at place {person.currentPlaceID}."
            )
        else:
            logger.info(
                f"Closest cooling stations for person {person.id} at "
                f"place {person.currentPlaceID}:"
            )
            for row in candidates:
                logger.info(
                    f"  Cooling station {row[0]} at {row[1]} with "
                    f"distance {row[1]} meters"
                )

        # log the agents
        logger.info("Logging agents after population initialization")
        self.log_agents()

    def create_input_tables(self):
        logger.info("Creating input tables for HeatRiskModel2...")
        super().create_input_tables()

        self.conn.execute("SHOW TABLES").pl().show()

        # pull the cooling centers experiment table name from the
        # model parameters
        cooling_centers_experiment_table = (
            Model.get_model().params.get(
                "cooling_centers_experiment.table"
            )
        )

        if not check_if_table_exists(
            self.conn, cooling_centers_experiment_table
        ):
            raise MissingRequiredTableError(
                cooling_centers_experiment_table
            )

        # if the table exists, create a view for it so we can easily query it
        # in the model
        logger.info(
            "creating <cooling_centers_experiment> view from "
            f"{cooling_centers_experiment_table}..."
        )
        cooling_centers_experiment_identifier = (
            quote_table_identifier(cooling_centers_experiment_table)
        )

        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW experiments AS
            SELECT * FROM {cooling_centers_experiment_identifier};
            """
        )
        cooling_centers_experiment_df = self.conn.execute(
            "SELECT * FROM experiments"
        ).pl()
        loaded_rows = cooling_centers_experiment_df.shape[0]
        logger.info(
            "Loaded cooling centers experiment data with "
            f"{loaded_rows} rows"
        )

        experiment_id = int(self.params["experiment_id"])
        logger.info(
            "Filtering experiment data for "
            f"experiment_id={experiment_id}..."
        )
        create_closest_cooling_centers(self.conn, experiment_id)
        closest_cooling_center_df2 = self.conn.execute(
            "SELECT * FROM closest_cooling_center"
        ).pl()
        logger.info(
            "Filtered closest cooling centers data has "
            f"{closest_cooling_center_df2.shape[0]} rows"
        )

    def step(self) -> None:
        """Step the model."""
        super().step()
        logger.info("Running step for HeatRiskModel")

    def get_run_log_data(self) -> RunLogData:
        """Aggregate agent data and return run-level summary statistics."""

        # Collect all agents
        agents = list(self.context.agents(agent_type=0))

        imputation = int(self.params.get("Imputation", 1))
        experiment_id = int(self.params.get("experiment_id", 1))

        # For millions of agents, you may want to use generators for memory efficiency
        # Flatten all heat_indices
        all_heat_indices = [
            val
            for person in agents
            for val in person.state.heat_indices
            # if not isinstance(val, float) or not math.isnan(val)
        ]
        filtered_all_heat_indices = [x for x in all_heat_indices if not math.isnan(x)]
        hrs_with_heh_cc = [
            person.state.max_hours_with_heh_cooling_center for person in agents
        ]
        hrs_with_heh_health = [
            person.state.max_hours_with_heh_health_effect for person in agents
        ]
        moved_to_cc = [
            person.state.ever_moved_to_cooling_center for person in agents
        ]
        totalVisits = sum(p.state.cooling_center_visits for p in agents)
        experienced_he = [
            person.state.experienced_heat_event for person in agents
        ]

        # Compute aggregates
        countAgents = len(agents)
        totalHeatIndex = float(sum(list(filtered_all_heat_indices)))
        totalHrsWithHEHCoolingCenter = float(sum(hrs_with_heh_cc))
        totalHrsWithHEHHealthEffect = float(sum(hrs_with_heh_health))
        countMovedToCoolingCenter = float(
            sum(1 for moved in moved_to_cc if moved)
        )
        countVisitsToCoolingCenter = float(totalVisits)
        countExperiencedHealthEffect = float(
            sum(1 for exp in experienced_he if exp)
        )
        totalFMinutesCoolingCenter = float(
            sum(p.state.f_minutes_cumulative_cooling_center for p in agents)
        )
        totalFMinutesHealthEffect = float(
            sum(p.state.f_minutes_cumulative_health_effect for p in agents)
        )
        all_heat_ints = [
            val
            for p in agents
            for val in p.state.heat_intensity
            if not math.isnan(val)
        ]
        totalHeatIntensity = float(sum(all_heat_ints))
        maxHeatIntensity = (
            float(max(all_heat_ints)) if all_heat_ints else float("-inf")
        )
        totalDEHHCoolingCenter = float(
            sum(p.state.dehh_cumulative_cooling_center for p in agents)
        )
        totalDEHHHealthEffect = float(
            sum(p.state.dehh_cumulative_health_effect for p in agents)
        )

        return RunLogData(
            imputation,
            experiment_id,
            countAgents,
            totalHeatIndex,
            totalHrsWithHEHCoolingCenter,
            totalHrsWithHEHHealthEffect,
            countMovedToCoolingCenter,
            countExperiencedHealthEffect,
            countVisitsToCoolingCenter,
            totalFMinutesCoolingCenter,
            totalFMinutesHealthEffect,
            totalHeatIntensity,
            maxHeatIntensity,
            totalDEHHCoolingCenter,
            totalDEHHHealthEffect,
        )

    def get_total_heat_index(self) -> float:
        # Collect all agents
        agents = list(self.context.agents(agent_type=0))

        # For millions of agents, you may want to use generators for
        # memory efficiency
        all_heat_indices = [
            val for person in agents for val in person.state.heat_indices
        ]
        filtered_all_heat_indices = [x for x in all_heat_indices if not math.isnan(x)]
        total_heat_index_for_tick = sum(list(filtered_all_heat_indices))
        self.total_heat_index = total_heat_index_for_tick

    def get_person_log_data(self, person: Person) -> PersonLogData:
        """Get the agent data for logging."""
        # heat_threshold_cooling_center = self.get_environment().heat_threshold_cooling_center
        # heat = filter_heat_indices(person.state.heat_indices, heat_threshold_cooling_center)
        imputation = int(self.params.get("Imputation", 1))
        experiment_id = int(self.params.get("experiment_id", 1))

        return PersonLogData(
            Imputation=imputation,
            experiment_id=experiment_id,
            minute_of_day=self.cal.minute_of_day,
            hour_of_day=self.cal.hour_of_day,
            rank=self.comm.Get_rank(),
            agent_id=person.id,
            x=person.pt.x,
            y=person.pt.y,
            place_id=person.currentPlaceID,
            outdoorHeatIndex=person.state.outdoor_heat_indices[0],
            outdoorWBGT=person.state.outdoor_wbgt_indices[0],
            outdoorDewPoint=person.state.outdoor_dew_point_indices[0],
            outdoorTemperature=person.state.outdoor_temp_indices[0],
            heatIndex=person.state.heat_indices[0],
            hourlyExcessHeatCoolingCenter=(
                person.state.hourly_excess_heat_cooling_center[0]
                if person.state.hourly_excess_heat_cooling_center
                else float("nan")
            ),
            hourlyExcessHeatHealthEffect=(
                person.state.hourly_excess_heat_health_effect[0]
                if person.state.hourly_excess_heat_health_effect
                else float("nan")
            ),
            hrsWithHourlyExcessHeatCoolingCenter=(
                person.state.hours_with_heh_cooling_center
            ),
            hrsWithHourlyExcessHeatHealthEffect=(
                person.state.hours_with_heh_health_effect
            ),
            experiencedHeatEvent=person.state.experienced_heat_event,
            movedToCoolingCenter=person.state.moved_to_cooling_center,
            outdoors=person.state.outdoors,
            hasACAccess=person.state.has_ac_access,
            heatEventPlaceId=person.state.heat_event_place_id,
            coolingCenterId=person.state.cooling_center_id,
            fMinutesCumulativeCoolingCenter=(
                person.state.f_minutes_cumulative_cooling_center
            ),
            fMinutesCumulativeHealthEffect=(
                person.state.f_minutes_cumulative_health_effect
            ),
            fMinutesThisTickCoolingCenter=(
                person.state.f_minutes_history_cooling_center[-1]
                if person.state.f_minutes_history_cooling_center
                else 0.0
            ),
            fMinutesThisTickHealthEffect=(
                person.state.f_minutes_history_health_effect[-1]
                if person.state.f_minutes_history_health_effect
                else 0.0
            ),
            heatIntensity=(
                person.state.heat_intensity[0]
                if person.state.heat_intensity
                else float("nan")
            ),
            dehhCoolingCenter=person.state.dehh_cumulative_cooling_center,
            dehhHealthEffect=person.state.dehh_cumulative_health_effect,
        )

    def log_run(self) -> None:
        """Log the run's data."""
        if not self.write_run_log:
            return

        # create a DataFrame for the agent logs
        logger.info("Logging run data...")
        run_log_df = pl.DataFrame([self.get_run_log_data()])

        # convert the DataFrame to an Arrow Table
        run_log_table = run_log_df.to_arrow()

        # Define partition schema
        partition_schema = pa.schema([pa.field("Imputation", pa.int32()), pa.field("experiment_id", pa.int32())])

        # Set Hive-style partitioning
        partitioning = HivePartitioning(partition_schema)

        # Write dataset
        ds.write_dataset(
            data=run_log_table,
            base_dir=self.run_log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

    def log_agents(self) -> None:
        """Log the agents' data."""
        if not self.write_agent_tick_log:
            return

        # create a DataFrame for the agent logs
        logger.info("Logging agents' data...")

        agent_log_df = pl.DataFrame(
            [
                self.get_person_log_data(person)
                for person in self.context.agents(agent_type=0)
            ]
        )

        # convert the DataFrame to an Arrow Table
        agent_log_table = agent_log_df.to_arrow()
        self.log_run()
        # Define partition schema
        partition_schema = pa.schema(
            [
                pa.field("Imputation", pa.int32()),
                pa.field("experiment_id", pa.int32()),
                pa.field("minute_of_day", pa.int32()),
                pa.field("rank", pa.int32()),
            ]
        )

        # Set Hive-style partitioning
        partitioning = HivePartitioning(partition_schema)

        # Write dataset
        ds.write_dataset(
            data=agent_log_table,
            base_dir=self.agent_log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )

        # Track total heat index after each tick
        self.get_total_heat_index()

    def log_hourly_persons(self) -> None:
        """Collect every agent's hourly rows and write a single Parquet file."""
        if not self.write_agent_hourly_log:
            return

        logger.info("Writing hourly person log …")

        imputation = int(self.params.get("Imputation", 1))
        experiment_id = int(self.params.get("experiment_id", 1))

        rows = []
        for person in self.context.agents(agent_type=0):
            # The deque holds the newest hour first, we can just extend the list
            rows.extend(person.state.hourly_log)

        if not rows:
            logger.warning("No hourly rows to write.")
            return

        df = pl.DataFrame(rows).with_columns(
            [
                pl.lit(imputation).cast(pl.Int32).alias("Imputation"),
                pl.lit(experiment_id).cast(pl.Int32).alias(
                    "experiment_id"
                ),
            ]
        )
        table = df.to_arrow()

        # You can reuse the same `hourly_person_log_file` path you defined in __init__
        # (or create a new one if you prefer).  Here we keep the same location.
        partition_schema = pa.schema(
            [
                pa.field("Imputation", pa.int32()),
                pa.field("experiment_id", pa.int32()),
                pa.field("hour_of_day", pa.int32()),
                pa.field("rank", pa.int32()),
            ]
        )
        partitioning = HivePartitioning(partition_schema)

        ds.write_dataset(
            data=table,
            base_dir=self.agent_hourly_log_file,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info(f"Hourly person log written - {len(rows)} rows total.")

    def log_end_run_persons(self) -> None:
        """
        Create a Parquet file that contains ONE compact row per
        agent with the ten columns you specified.
        """
        if not self.write_end_run_person_log:
            return

        logger.info("Writing end of run person summary …")  # noqa: E501
        imputation = int(self.params.get("Imputation", 1))
        experiment_id = int(self.params.get("experiment_id", 1))

        rows = []
        for person in self.context.agents(agent_type=0):
            # 1. Compute the *average* heat index across the whole run
            heat_vals = [
                v for v in person.state.heat_indices if not math.isnan(v)
            ]
            avg_heat = (
                float(sum(heat_vals) / len(heat_vals))
                if heat_vals
                else float("nan")
            )

            # 2. Compute the *average* and *max* of hourly intensity
            intensity_vals = [
                v for v in person.state.heat_intensity if not math.isnan(v)
            ]
            avg_intensity = (
                float(sum(intensity_vals) / len(intensity_vals))
                if intensity_vals
                else float("nan")
            )
            max_intensity = (
                float(max(intensity_vals))
                if intensity_vals
                else float("-inf")
            )

            # 3. Pull the other fields directly from the agent state
            rows.append(
                {
                    "Imputation": imputation,
                    "experiment_id": experiment_id,
                    "agent_id": person.id,
                    "hrsWithhourlyExcessHeatCoolingCenter": (
                        person.state.max_hours_with_heh_cooling_center
                    ),
                    "hrsWithhourlyExcessHeatHealthEffect": (
                        person.state.max_hours_with_heh_health_effect
                    ),
                    "fMinutesCumulativeCoolingCenter": (
                        person.state.f_minutes_cumulative_cooling_center
                    ),
                    "fMinutesCumulativeHealthEffect": (
                        person.state.f_minutes_cumulative_health_effect
                    ),
                    "averageHeatIndex": avg_heat,
                    "averageHeatIntensity": avg_intensity,
                    "maxHeatIntensity": max_intensity,
                    "dehhCoolingCenter": (
                        person.state.dehh_cumulative_cooling_center
                    ),
                    "dehhHealthEffect": (
                        person.state.dehh_cumulative_health_effect
                    ),
                }
            )
        if not rows:
            logger.warning(
                "No agents found: end of run person file will be empty."
            )
            return

        df = pl.DataFrame(rows)
        table = df.to_arrow()

        # ------------------------------------------------------------------
        # Write the table, we reuse the same base folder you already gave
        # for run level logs, just a sub folder called “person_end”.
        # ------------------------------------------------------------------
        partition_schema = pa.schema([pa.field("Imputation", pa.int32()), pa.field("experiment_id", pa.int32())])
        partitioning = HivePartitioning(partition_schema)
        ds.write_dataset(
            data=table,
            base_dir=self.run_person_log_file,  # <-- defined in __init__ (see below)
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info(
            f"End of run person summary written - {len(rows)} rows."
        )

    def at_end(self) -> None:
        logger.info("Logging final model run data...")
        # 1) Write the hour level file (all agents, all hours)
        if self.write_agent_hourly_log:
            self.log_hourly_persons()
        # 2) End of run per agent file (your new compact table)
        if self.write_end_run_person_log:
            self.log_end_run_persons()
        # 2) Write the regular run level summary (unchanged)
        if self.write_run_log:
            self.log_run()


# Register HeatRiskModel
Models.add_model(HeatRiskModel2.__module__ + "." + HeatRiskModel2.__name__, HeatRiskModel2)
