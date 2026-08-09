"""Optional live Arrow Flight observation server for an in-progress CasmPop run.

Speaks Apache Arrow Flight, which ships inside the ``pyarrow`` wheel and is
already a hard dependency of casmsocial. Nothing here needs a lazy import or
an optional extra: importing this module never requires anything beyond what
casmsocial already installs.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.flight as flight

if TYPE_CHECKING:
    from casmsocial.model import Model

ENDPOINT_FILENAME = "arrow_endpoint.txt"


@dataclass
class ArrowServerHandle:
    """Lifecycle handle for the embedded Flight server, kept on CasmPop."""

    server: CasmPopFlightServer
    host: str
    port: int
    endpoint_file: pathlib.Path

    def shutdown(self) -> None:
        self.server.shutdown()


class CasmPopFlightServer(flight.FlightServerBase):
    """Arrow Flight server exposing a live CasmPop's observer output tables.

    Unlike casmservice's ArrowFlightServer (upload-based, backed by a static
    dict), this server has no internal cache: every request calls
    ``model.get_observer_output_tables()`` fresh, so callers always see
    whatever the most recently completed step produced. Read-only: DoPut is
    unsupported.
    """

    def __init__(self, location: str | tuple[str, int], model: Model) -> None:
        super().__init__(location)
        self._model = model

    def _table_name(self, descriptor: flight.FlightDescriptor) -> str:
        if descriptor.path is None or len(descriptor.path) != 1:
            raise flight.FlightServerError("descriptor path must be a single table name")
        return descriptor.path[0].decode("utf-8")

    def _table(self, name: str) -> pa.Table:
        table = self._model.get_observer_output_tables().get(name)
        if table is None:
            raise flight.FlightServerError(f"table not found: {name}")
        return table

    def _flight_info(self, name: str, table: pa.Table) -> flight.FlightInfo:
        descriptor = flight.FlightDescriptor.for_path(name)
        endpoint = flight.FlightEndpoint(name.encode("utf-8"), [])
        return flight.FlightInfo(table.schema, descriptor, [endpoint], table.num_rows, table.nbytes)

    def get_flight_info(self, context, descriptor: flight.FlightDescriptor) -> flight.FlightInfo:
        name = self._table_name(descriptor)
        return self._flight_info(name, self._table(name))

    def do_get(self, context, ticket: flight.Ticket) -> flight.RecordBatchStream:
        name = ticket.ticket.decode("utf-8")
        return flight.RecordBatchStream(self._table(name))

    def do_put(self, context, descriptor, reader, writer) -> None:
        raise flight.FlightServerError("CasmPopFlightServer is read-only; DoPut is unsupported")

    def list_flights(self, context, criteria):
        for name, table in self._model.get_observer_output_tables().items():
            yield self._flight_info(name, table)


def start_arrow_server(model: Model, *, host: str, endpoint_dir: pathlib.Path) -> ArrowServerHandle:
    """Start an embedded Arrow Flight server bound to an OS-assigned port.

    The server starts serving as soon as it's constructed -- no separate
    activation step, and no background thread needed, since the underlying
    gRPC server runs its own request-handling threads.

    Writes ``<endpoint_dir>/arrow_endpoint.txt`` containing ``"{host}:{port}"``
    once the port is known.
    """
    server = CasmPopFlightServer((host, 0), model)
    port = server.port

    endpoint_dir.mkdir(parents=True, exist_ok=True)
    endpoint_file = endpoint_dir / ENDPOINT_FILENAME
    endpoint_file.write_text(f"{host}:{port}\n", encoding="utf-8")

    return ArrowServerHandle(server=server, host=host, port=port, endpoint_file=endpoint_file)
