"""agent_log and agent_state_delta x/y are the current place's longitude/latitude."""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from casmsocial.casmpop import AgentLogger, DeltaAgentStateLogger


def make_model(tmp_path, people, places):
    return SimpleNamespace(
        params={
            "simulation.run_id": "test-run",
            "random.seed": 42,
            "observers.output_dir": str(tmp_path),
            "observers.agent_log_file": "agent_log.parquet",
            "observers.delta_agent_state_file": "agent_state_delta.parquet",
            "observers.delta_agent_state_audit_file": "agent_state_delta_audit.parquet",
        },
        context=SimpleNamespace(agents=lambda **_: iter(people)),
        comm=SimpleNamespace(Get_rank=lambda: 0),
        cal=SimpleNamespace(tick=60.0),
        places_proj=SimpleNamespace(lookup_place=places.get),
    )


class FakePerson(SimpleNamespace):
    @property
    def place_id(self):
        return self.state.place_id

    @property
    def rank_place_id(self):
        return self.state.place_id


def person(agent_id, place_id):
    # pt holds integer-truncated input values; logged x/y must not use it.
    return FakePerson(id=agent_id, pt=SimpleNamespace(x=708827, y=3961929), state=SimpleNamespace(place_id=place_id))


def place(latitude, longitude):
    return SimpleNamespace(data=SimpleNamespace(latitude=latitude, longitude=longitude))


def logged(tmp_path, people, places, **params):
    model = make_model(tmp_path, people, places)
    model.params.update(params)
    observer = AgentLogger("test", model)
    observer.on_step(model)
    return observer.get_output_tables(model)["agent_log"]


def test_x_is_longitude_and_y_is_latitude_of_the_current_place(tmp_path):
    walker = person(1, 10)
    places = {10: place(35.779273, -78.689598), 20: place(35.91, -79.05)}
    table = logged(tmp_path, [walker], places)
    assert table.select(["x", "y"]).to_pylist() == [{"x": -78.689598, "y": 35.779273}]
    walker.state.place_id = 20
    table = logged(tmp_path / "moved", [walker], places)
    assert table.select(["x", "y"]).to_pylist() == [{"x": -79.05, "y": 35.91}]
    assert "latitude" not in table.column_names


@pytest.mark.parametrize("missing", ["unknown place", "nan", "no projection"])
def test_unknown_coordinates_are_null_even_before_known_ones(tmp_path, missing):
    # Many leading nulls must not make the column type depend on a sample.
    places = {10: place(35.78, -78.64), 11: place(float("nan"), float("nan"))}
    lost = {"unknown place": 99, "nan": 11, "no projection": 10}[missing]
    people = [person(i, lost) for i in range(150)] + [person(999, 10)]
    model = make_model(tmp_path, people, places)
    if missing == "no projection":
        del model.places_proj
    observer = AgentLogger("test", model)
    observer.on_step(model)
    table = observer.get_output_tables(model)["agent_log"]
    assert table.schema.field("x").type == table.schema.field("y").type == pa.float64()
    assert table["x"].null_count == 150 + (missing == "no projection")
    if missing != "no projection":
        assert table.select(["x", "y"]).to_pylist()[-1] == {"x": -78.64, "y": 35.78}


def test_coordinates_only_when_requested_by_custom_columns(tmp_path):
    table = logged(tmp_path, [person(1, 10)], {10: place(35.78, -78.64)}, **{"observers.agent_log_columns": ["place_id"]})
    assert table.column_names[-1] == "place_id" and "x" not in table.column_names


def test_delta_state_uses_the_same_coordinates_and_tolerates_nulls(tmp_path):
    places = {10: place(35.78, -78.64)}
    people = [person(i, 99) for i in range(150)] + [person(999, 10)]
    model = make_model(tmp_path, people, places)
    model.params["observers.delta_agent_state.enabled"] = True
    observer = DeltaAgentStateLogger("test", model)
    observer.on_step(model)
    delta = observer.get_output_tables(model)["agent_state_delta"]
    assert delta.schema.field("x").type == pa.float64()
    assert delta["x"].null_count == 150
    rows = {row["agent_id"]: row for row in delta.select(["agent_id", "x", "y"]).to_pylist()}
    assert rows[999] == {"agent_id": 999, "x": -78.64, "y": 35.78}
    # A move to a place with coordinates is a state change; staying put is not.
    people[0].state.place_id = 10
    model.cal.tick = 120.0
    observer.on_step(model)
    changed = observer.get_output_tables(model)["agent_state_delta"]
    assert changed.select(["agent_id", "x"]).to_pylist() == [{"agent_id": 0, "x": -78.64}]
