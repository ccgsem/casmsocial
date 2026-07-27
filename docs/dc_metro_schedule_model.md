# DC Metro Schedule Model

`casmsocial.citysim.dcmetro_schedule_model.DCMetroScheduleModel` extends the
base CASMSocial schedule model with anchors and activity IDs for daycare and
discretionary activities.

| Activity ID | Activity | Person anchor |
| ---: | --- | --- |
| 0 | home | `sp_hh_id` |
| 1 | work | `sp_work_id` |
| 2 | school | `sp_school_id` |
| 3 | daycare | `sp_daycare_id` |
| 4–11 | shopping, meal, personal care, social, recreation, healthcare, errand, other | matching `sp_<activity>_id` |

The current planned-activity runtime accepts one anchor of each kind per
person. It is suitable for schedule exports that consolidate a person's daily
activities of a kind to one assigned destination. A schedule containing two or
more destinations of the same kind requires a future multi-anchor runtime
extension; the exporter should reject such a plan rather than silently moving
the person to the wrong place.

Set `model.name` to
`casmsocial.citysim.dcmetro_schedule_model.DCMetroScheduleModel` when loading
the exported local inputs. The model remains compatible with the existing
timeless `social_networks` input and derives in-person interaction opportunities
from scheduled co-location.
