from __future__ import annotations

from casmsocial.casmpop import CasmPop
from casmsocial.citysim.dcmetro_schedule_model import (
    ACTIVITY_NAMES,
    PLANNED_ACTIVITY_NAMES,
    DCMetroScheduleModel,
)
from casmsocial.factory import Models


def test_dc_metro_schedule_model_registers_daycare_and_discretionary_activities():
    previous_activity_names = CasmPop.get_activity_names()
    previous_planned_activity_names = CasmPop.get_planned_activity_names()
    previous_activities_data_type = CasmPop._CasmPop__activities_data_type
    try:
        DCMetroScheduleModel.register_schedule_activities()

        assert CasmPop.get_activity_names() == ACTIVITY_NAMES
        assert CasmPop.get_planned_activity_names() == PLANNED_ACTIVITY_NAMES
        assert ACTIVITY_NAMES.index("daycare") == 3
        assert "sp_daycare_id" in PLANNED_ACTIVITY_NAMES
        assert "sp_social_id" in PLANNED_ACTIVITY_NAMES
        assert "sp_healthcare_id" in PLANNED_ACTIVITY_NAMES
    finally:
        CasmPop.register_activity_names(previous_activity_names)
        CasmPop.register_planned_activity_names(previous_planned_activity_names)
        CasmPop._CasmPop__activities_data_type = previous_activities_data_type


def test_dc_metro_schedule_model_is_registered_with_factory():
    model_name = "casmsocial.citysim.dcmetro_schedule_model.DCMetroScheduleModel"

    assert Models.create_model(model_name) is DCMetroScheduleModel
