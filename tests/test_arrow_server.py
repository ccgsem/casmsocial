from __future__ import annotations

import pyarrow as pa
import pyarrow.flight as flight
import pytest

from casmsocial.arrow_server import CasmPopFlightServer, start_arrow_server
from casmsocial.casmpop import CasmPop


class _FakeModel:
    """Minimal stand-in for CasmPop, exposing only what the server needs."""

    def __init__(self, tables: dict[str, pa.Table]):
        self._tables = tables

    def get_observer_output_tables(self) -> dict[str, pa.Table]:
        return dict(self._tables)


def _bare_model(*, rank: int, enabled: bool) -> CasmPop:
    model = CasmPop.__new__(CasmPop)
    model.rank = rank
    model.params = {
        "observers.arrow_server.enabled": enabled,
        "observers.arrow_server.host": "127.0.0.1",
    }
    model._arrow_server_handle = None
    return model


def test_arrow_server_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=0, enabled=False)

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is None
    assert not (tmp_path / "arrow_endpoint.txt").exists()


def test_arrow_server_skipped_on_non_zero_rank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=1, enabled=True)

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is None
    assert not (tmp_path / "arrow_endpoint.txt").exists()


def test_arrow_server_start_failure_is_handled_gracefully(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=0, enabled=True)

    def _raise(*args, **kwargs):
        raise OSError("could not bind")

    monkeypatch.setattr("casmsocial.arrow_server.start_arrow_server", _raise)

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is None


def test_stop_arrow_server_without_having_started_is_a_noop():
    model = CasmPop.__new__(CasmPop)
    model._stop_arrow_server()
    assert getattr(model, "_arrow_server_handle", None) is None


def test_start_arrow_server_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=0, enabled=True)
    sentinel = object()
    model._arrow_server_handle = sentinel

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is sentinel


def test_arrow_server_round_trip(tmp_path):
    table = pa.table({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    model = _FakeModel({"agent_log": table})

    handle = start_arrow_server(model, host="127.0.0.1", endpoint_dir=tmp_path)
    try:
        endpoint_file = tmp_path / "arrow_endpoint.txt"
        assert endpoint_file.exists()
        host, port_str = endpoint_file.read_text(encoding="utf-8").strip().split(":")
        assert host == "127.0.0.1"
        port = int(port_str)
        assert port > 0
        assert handle.port == port

        client = flight.connect(f"grpc://{host}:{port}")

        names = {fi.descriptor.path[0].decode("utf-8") for fi in client.list_flights()}
        assert names == {"agent_log"}

        fetched = client.do_get(flight.Ticket(b"agent_log")).read_all()
        assert fetched.equals(table)

        info = client.get_flight_info(flight.FlightDescriptor.for_path("agent_log"))
        assert "x" in str(info.schema)

        with pytest.raises(flight.FlightServerError):
            client.do_get(flight.Ticket(b"missing_channel")).read_all()
    finally:
        handle.shutdown()


def test_arrow_server_connection_refused_after_shutdown(tmp_path):
    model = _FakeModel({})
    handle = start_arrow_server(model, host="127.0.0.1", endpoint_dir=tmp_path)
    host, port = handle.host, handle.port
    handle.shutdown()

    client = flight.connect(f"grpc://{host}:{port}")
    with pytest.raises(flight.FlightUnavailableError):
        client.do_get(flight.Ticket(b"whatever")).read_all()


def test_do_put_is_unsupported():
    server = CasmPopFlightServer(("127.0.0.1", 0), _FakeModel({}))
    try:
        client = flight.connect(f"grpc://127.0.0.1:{server.port}")
        table = pa.table({"x": [1]})
        with pytest.raises(flight.FlightServerError):
            writer, _ = client.do_put(flight.FlightDescriptor.for_path("t"), table.schema)
            writer.write_table(table)
            writer.close()
    finally:
        server.shutdown()
