"""Optional live Arrow/Ice observation server for an in-progress CasmPop run.

Requires the `service` extra (`uv sync --extra service`). Nothing in this
module is imported at top level from casmpop.py -- only inside the method
that starts the adapter -- so importing casmsocial never requires zeroc-ice
to be installed. See casmsocial.remote_llm_adapter for the same lazy-import
pattern applied to the optional `anthropic` dependency.

Because the Ice servant must subclass the generated ``arrowservice.ArrowServer``
skeleton (a plain duck-typed class is rejected by the Ice runtime), Ice and
the generated bindings are imported here at module scope rather than per
function. That's safe only because this whole module is itself imported
lazily by casmpop.py -- never at casmsocial's own import time.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from casmsocial.arrow_serialization import ArrowServerUnavailableError, serialize_table

if TYPE_CHECKING:
    from casmsocial.model import Model

ENDPOINT_FILENAME = "arrow_endpoint.txt"


try:
    import Ice

    import arrowservice
except ImportError as exc:
    raise ArrowServerUnavailableError(
        "Arrow/Ice live observation server requires the `zeroc-ice` package; "
        "install it with `uv sync --extra service`."
    ) from exc


@dataclass
class ArrowServerHandle:
    """Lifecycle handle for the embedded Ice adapter, kept on CasmPop."""

    communicator: Any
    adapter: Any
    host: str
    port: int
    endpoint_file: pathlib.Path

    def shutdown(self) -> None:
        self.communicator.destroy()


class CasmPopArrowServerI(arrowservice.ArrowServer):
    """Ice servant exposing a live CasmPop's observer output tables.

    Unlike casmservice's ArrowServerI (upload-based, backed by a static
    dict), this servant has no internal cache: every method calls
    ``model.get_observer_output_tables()`` fresh, so callers always see
    whatever the most recently completed step produced.
    """

    def __init__(self, model: Model) -> None:
        self._model = model

    def getTable(self, tableName, current=None):
        table = self._model.get_observer_output_tables().get(tableName)
        if table is None:
            raise arrowservice.TableNotFound(tableName)
        return arrowservice.ArrowTable(data=serialize_table(table))

    def getTableSchema(self, tableName, current=None):
        table = self._model.get_observer_output_tables().get(tableName)
        if table is None:
            raise arrowservice.TableNotFound(tableName)
        return str(table.schema)

    def listTableNames(self, current=None):
        return list(self._model.get_observer_output_tables().keys())

    def uploadTable(self, tableName, table, current=None):
        raise NotImplementedError("CasmPopArrowServerI is read-only; uploadTable is unsupported")


def _resolve_bound_port(adapter: Any) -> int:
    """Read back the OS-assigned port from the adapter's published endpoint."""
    endpoint = adapter.getEndpoints()[0]
    info = endpoint.getInfo()
    port = getattr(info, "port", None)
    if port is not None:
        return int(port)
    match = re.search(r"-p (\d+)", endpoint.toString())
    if match is None:
        raise ArrowServerUnavailableError(f"could not determine bound port from endpoint {endpoint}")
    return int(match.group(1))


def start_arrow_server(model: Model, *, host: str, endpoint_dir: pathlib.Path) -> ArrowServerHandle:
    """Start an embedded Ice ArrowServer adapter bound to an OS-assigned port.

    Writes ``<endpoint_dir>/arrow_endpoint.txt`` containing ``"{host}:{port}"``
    once the adapter is activated and the port is known.
    """
    communicator = Ice.initialize([])
    endpoint_str = f"default -h {host} -p 0"
    adapter = communicator.createObjectAdapterWithEndpoints("ArrowAdapter", endpoint_str)
    servant = CasmPopArrowServerI(model)
    adapter.add(servant, Ice.Identity(name="ArrowServer"))
    adapter.activate()

    port = _resolve_bound_port(adapter)

    endpoint_dir.mkdir(parents=True, exist_ok=True)
    endpoint_file = endpoint_dir / ENDPOINT_FILENAME
    endpoint_file.write_text(f"{host}:{port}\n", encoding="utf-8")

    return ArrowServerHandle(
        communicator=communicator,
        adapter=adapter,
        host=host,
        port=port,
        endpoint_file=endpoint_file,
    )
