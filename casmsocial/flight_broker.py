"""Read-only Arrow Flight adapter for :mod:`casmsocial.observation_broker`."""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight

from casmsocial.observation_broker import ObservationBroker, ObservationCursorExpiredError

ENDPOINT_FILENAME = "arrow_endpoint.txt"


class BrokerFlightServer(flight.FlightServerBase):
    """Expose retained broker batches through the standard Arrow Flight API."""

    def __init__(self, location: str | tuple[str, int], broker: ObservationBroker) -> None:
        super().__init__(location)
        self._broker = broker

    @staticmethod
    def _channel(descriptor: flight.FlightDescriptor) -> str:
        if descriptor.path is None or len(descriptor.path) != 1:
            raise flight.FlightServerError("Flight descriptor path must contain one channel name")
        return descriptor.path[0].decode("utf-8")

    def _table(self, channel: str) -> pa.Table:
        try:
            batches = self._broker.read(channel).batches
        except ObservationCursorExpiredError as error:
            raise flight.FlightServerError(str(error)) from error
        if not batches:
            raise flight.FlightServerError(f"no retained observation batches for channel {channel!r}")
        return pa.concat_tables([batch.table for batch in batches])

    def _info(self, channel: str) -> flight.FlightInfo:
        table = self._table(channel)
        descriptor = flight.FlightDescriptor.for_path(channel)
        endpoint = flight.FlightEndpoint(flight.Ticket(channel.encode("utf-8")), [])
        return flight.FlightInfo(table.schema, descriptor, [endpoint], table.num_rows, table.nbytes)

    def list_flights(self, context, criteria):
        for channel in self._broker.channels():
            try:
                yield self._info(channel)
            except flight.FlightServerError:
                continue

    def get_flight_info(self, context, descriptor: flight.FlightDescriptor) -> flight.FlightInfo:
        return self._info(self._channel(descriptor))

    def do_get(self, context, ticket: flight.Ticket) -> flight.RecordBatchStream:
        return flight.RecordBatchStream(self._table(ticket.ticket.decode("utf-8")))

    def do_put(self, context, descriptor, reader, writer) -> None:
        raise flight.FlightServerError("BrokerFlightServer is read-only")


def start_broker_flight_server(run_dir: Path, broker: ObservationBroker) -> BrokerFlightServer:
    """Start a loopback Flight server and retain CASMService's endpoint-file convention."""
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    server = BrokerFlightServer(("127.0.0.1", 0), broker)
    endpoint_file = run_dir / ENDPOINT_FILENAME
    endpoint_file.write_text(f"127.0.0.1:{server.port}\n", encoding="utf-8")
    os.chmod(endpoint_file, 0o600)
    return server
