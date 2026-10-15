"""agent_log latitude/longitude come from the person's current place."""

from types import SimpleNamespace

import pytest

from casmsocial.casmpop import AgentLogger


def make_model(tmp_path, people, places):
    return SimpleNamespace(
        params={
            "simulation.run_id": "test-run",
            "random.seed": 42,
            "observers.output_dir": str(tmp_path),
            "observers.agent_log_file": "agent_log.parquet",
        },
        context=SimpleNamespace(agents=lambda **_: iter(people)),
        comm=SimpleNamespace(Get_rank=lambda: 0),
        cal=SimpleNamespace(tick=60.0),
        places_proj=SimpleNamespace(lookup_place=places.get),
    )


def person(agent_id, place_id):
    return SimpleNamespace(id=agent_id, pt=SimpleNamespace(x=1.0, y=2.0), state=SimpleNamespace(place_id=place_id))


def place(latitude, longitude):
    return SimpleNamespace(data=SimpleNamespace(latitude=latitude, longitude=longitude))


def logged(tmp_path, people, places):
    model = make_model(tmp_path, people, places)
    observer = AgentLogger("test", model)
    observer.on_step(model)
    return observer.get_output_tables(model)["agent_log"]


def test_coordinates_follow_the_current_place(tmp_path):
    walker = person(1, 10)
    places = {10: place(35.78, -78.64), 20: place(35.91, -79.05)}
    table = logged(tmp_path, [walker], places)
    assert table.select(["latitude", "longitude"]).to_pylist() == [{"latitude": 35.78, "longitude": -78.64}]
    walker.state.place_id = 20
    table = logged(tmp_path / "moved", [walker], places)
    assert table.select(["latitude", "longitude"]).to_pylist() == [{"latitude": 35.91, "longitude": -79.05}]


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
    assert table.schema.field("latitude").type == table.schema.field("longitude").type
    assert str(table.schema.field("latitude").type) == "double"
    assert table["latitude"].null_count == 150 + (missing == "no projection")
    if missing != "no projection":
        assert table["latitude"].to_pylist()[-1] == 35.78


def test_coordinates_only_when_requested_by_custom_columns(tmp_path):
    model = make_model(tmp_path, [person(1, 10)], {10: place(35.78, -78.64)})
    model.params["observers.agent_log_columns"] = ["place_id"]
    observer = AgentLogger("test", model)
    observer.on_step(model)
    assert "latitude" not in observer.get_output_tables(model)["agent_log"].column_names
