"""CASMSocial model for DC metro schedules with daycare and discretionary anchors."""

from __future__ import annotations

from loguru import logger
from mpi4py import MPI

from casmsocial.casmpop import AgentLogger, CasmPop
from casmsocial.factory import Models
from casmsocial.person import Person, PersonData
from casmsocial.place import Place, PlaceData

PLANNED_ACTIVITY_NAMES = [
    "sp_hh_id",
    "sp_work_id",
    "sp_school_id",
    "sp_daycare_id",
    "sp_shopping_id",
    "sp_meal_id",
    "sp_personal_care_id",
    "sp_social_id",
    "sp_recreation_id",
    "sp_healthcare_id",
    "sp_errand_id",
    "sp_other_id",
]
ACTIVITY_NAMES = [
    "home",
    "work",
    "school",
    "daycare",
    "shopping",
    "meal",
    "personal_care",
    "social",
    "recreation",
    "healthcare",
    "errand",
    "other",
]


class DCMetroScheduleModel(CasmPop):
    """A schedule-driven model with explicit daycare and discretionary places.

    Each person has at most one local anchor for each activity category. This
    matches the current CASMSocial planned-activity representation; a person
    with multiple shopping, social, or other destinations needs a future
    multi-anchor representation before loading that schedule.
    """

    @classmethod
    def get_default_parameters(cls) -> dict:
        params = super().get_default_parameters()
        params.update({"model.name": cls.__module__ + "." + cls.__name__})
        return params

    @classmethod
    def register_schedule_activities(cls) -> None:
        """Register the activity IDs and person-anchor columns used by this model."""
        CasmPop.register_planned_activity_names(PLANNED_ACTIVITY_NAMES)
        CasmPop.register_activity_names(ACTIVITY_NAMES)

    def __init__(self, comm: MPI.Intracomm, params: dict):
        super().__init__(comm, params)

    def build_context(self) -> None:
        """Register DC schedule activity anchors before constructing the population."""
        self.add_observer(AgentLogger("AgentLogger", self))
        logger.info(f"Registering person type (TYPE={Person.TYPE})...")
        CasmPop.setPersonClass(Person, PersonData)
        logger.info(f"Registering place type (TYPE={Place.TYPE})...")
        CasmPop.setPlaceClass(Place, PlaceData)
        self.register_schedule_activities()
        super().build_context()
        self.log_agents()


Models.add_model(DCMetroScheduleModel.__module__ + "." + DCMetroScheduleModel.__name__, DCMetroScheduleModel)
