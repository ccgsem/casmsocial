from casmsocial.casmpop import CasmPop
from casmsocial.citysim.colorado_front_range_schedule_model import (
    ACTIVITY_NAMES,
    PLANNED_ACTIVITY_NAMES,
    ColoradoFrontRangeScheduleModel,
)


def test_colorado_model_registers_event_schedule_activity_contract():
    previous_names = CasmPop.get_planned_activity_names()
    try:
        ColoradoFrontRangeScheduleModel.register_schedule_activities()
        assert CasmPop.get_planned_activity_names() == PLANNED_ACTIVITY_NAMES
        assert CasmPop.get_activity_names() == ACTIVITY_NAMES
    finally:
        CasmPop.register_planned_activity_names(previous_names)
