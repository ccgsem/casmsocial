"""CASMSocial model for Colorado Front Range event-level schedules."""

from __future__ import annotations

from loguru import logger
from mpi4py import MPI

from casmsocial.casmpop import AgentLogger, CasmPop
from casmsocial.factory import Models
from casmsocial.person import Person, PersonData
from casmsocial.place import Place, PlaceData


PLANNED_ACTIVITY_NAMES = [
    "sp_hh_id", "sp_work_id", "sp_school_id", "sp_daycare_id", "sp_shopping_id",
    "sp_meal_id", "sp_personal_care_id", "sp_social_id", "sp_recreation_id",
    "sp_healthcare_id", "sp_errand_id", "sp_other_id",
]
ACTIVITY_NAMES = [
    "home", "work", "school", "daycare", "shopping", "meal", "personal_care",
    "social", "recreation", "healthcare", "errand", "other",
]


class ColoradoFrontRangeScheduleModel(CasmPop):
    """Schedule-driven model whose activity rows supply authoritative places.

    Person anchor columns are retained for base-model compatibility, but a
    routed activity's ``sp_act_id`` determines its actual destination.
    """

    @classmethod
    def get_default_parameters(cls) -> dict:
        params = super().get_default_parameters()
        params.update({"model.name": cls.__module__ + "." + cls.__name__})
        return params

    @classmethod
    def register_schedule_activities(cls) -> None:
        CasmPop.register_planned_activity_names(PLANNED_ACTIVITY_NAMES)
        CasmPop.register_activity_names(ACTIVITY_NAMES)

    def __init__(self, comm: MPI.Intracomm, params: dict):
        super().__init__(comm, params)

    def build_context(self) -> None:
        if self._agent_log_enabled():
            self.add_observer(AgentLogger("AgentLogger", self))
        logger.info("Registering Colorado Front Range schedule activities")
        CasmPop.setPersonClass(Person, PersonData)
        CasmPop.setPlaceClass(Place, PlaceData)
        self.register_schedule_activities()
        super().build_context()
        self.log_agents()


Models.add_model(
    ColoradoFrontRangeScheduleModel.__module__ + "." + ColoradoFrontRangeScheduleModel.__name__,
    ColoradoFrontRangeScheduleModel,
)
