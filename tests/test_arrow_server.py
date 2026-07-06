from __future__ import annotations

import importlib.util

import pyarrow as pa
import pytest

from casmsocial.arrow_serialization import ArrowServerUnavailableError
from casmsocial.casmpop import CasmPop

ICE_AVAILABLE = importlib.util.find_spec("Ice") is not None

# casmsocial.arrow_server itself raises ArrowServerUnavailableError at import
# time when zeroc-ice isn't installed (by design -- see its module
# docstring), so this module-level import must stay conditional: importing
# it unconditionally would break collection of this entire test file in any
# environment without the optional `service` extra (e.g. default CI).
if ICE_AVAILABLE:
    from casmsocial.arrow_server import CasmPopArrowServerI, start_arrow_server
else:
    CasmPopArrowServerI = None
    start_arrow_server = None


class _FakeModel:
    """Minimal stand-in for CasmPop, exposing only what the servant needs."""

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


@pytest.mark.skipif(
    not ICE_AVAILABLE,
    reason=(
        "monkeypatch.setattr('casmsocial.arrow_server....') must import the "
        "real module to patch it; without zeroc-ice that module fails to "
        "import on its own (which is exactly what this test simulates), so "
        "the patch target can't be resolved at all. The real degrade path "
        "this test models is already exercised for free in a no-Ice "
        "environment by every other test here that calls "
        "_start_arrow_server_if_enabled() without zeroc-ice installed."
    ),
)
def test_arrow_server_unavailable_is_handled_gracefully(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=0, enabled=True)

    def _raise(*args, **kwargs):
        raise ArrowServerUnavailableError("zeroc-ice not installed")

    monkeypatch.setattr("casmsocial.arrow_server.start_arrow_server", _raise)

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is None


@pytest.mark.skipif(
    ICE_AVAILABLE,
    reason="exercises the real (unmocked) degrade path when zeroc-ice genuinely isn't installed",
)
def test_arrow_server_start_degrades_gracefully_when_ice_not_installed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _bare_model(rank=0, enabled=True)

    model._start_arrow_server_if_enabled()

    assert model._arrow_server_handle is None
    assert not (tmp_path / "arrow_endpoint.txt").exists()


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


@pytest.mark.skipif(not ICE_AVAILABLE, reason="zeroc-ice not installed (service extra)")
def test_arrow_server_round_trip(tmp_path):
    import Ice

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

        communicator = Ice.initialize([])
        try:
            from arrowservice import ArrowServerPrx, TableNotFound

            proxy = communicator.stringToProxy(f"ArrowServer:default -h {host} -p {port}")
            server = ArrowServerPrx.checkedCast(proxy)
            assert server is not None

            assert server.listTableNames() == ["agent_log"]

            arrow_table = server.getTable("agent_log")
            from casmsocial.arrow_serialization import serialize_table

            assert bytes(arrow_table.data) == serialize_table(table)

            assert "x" in server.getTableSchema("agent_log")

            with pytest.raises(TableNotFound):
                server.getTable("missing_channel")
        finally:
            communicator.destroy()
    finally:
        handle.shutdown()


@pytest.mark.skipif(not ICE_AVAILABLE, reason="zeroc-ice not installed (service extra)")
def test_arrow_server_connection_refused_after_shutdown(tmp_path):
    import Ice

    model = _FakeModel({})
    handle = start_arrow_server(model, host="127.0.0.1", endpoint_dir=tmp_path)
    host, port = handle.host, handle.port
    handle.shutdown()

    communicator = Ice.initialize([])
    try:
        from arrowservice import ArrowServerPrx

        proxy = communicator.stringToProxy(f"ArrowServer:default -h {host} -p {port}")
        with pytest.raises(Ice.LocalException):
            ArrowServerPrx.checkedCast(proxy)
    finally:
        communicator.destroy()


@pytest.mark.skipif(not ICE_AVAILABLE, reason="zeroc-ice not installed (service extra)")
def test_servant_get_table_not_found_without_ice_running():
    table = pa.table({"x": [1]})
    servant = CasmPopArrowServerI(_FakeModel({"agent_log": table}))

    assert servant.listTableNames() == ["agent_log"]
    with pytest.raises(NotImplementedError):
        servant.uploadTable("agent_log", None)
