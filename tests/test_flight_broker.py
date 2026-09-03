import pyarrow as pa
import pyarrow.flight as flight

from casmsocial.flight_broker import BrokerFlightServer, start_broker_flight_server
from casmsocial.observation_broker import ObservationBroker


def test_flight_server_discovers_and_reads_broker_channel():
    broker = ObservationBroker()
    broker.publish("agents", pa.table({"agent_id": [1, 2]}))
    server = BrokerFlightServer(("127.0.0.1", 0), broker)
    try:
        client = flight.connect(f"grpc://127.0.0.1:{server.port}")
        info = client.get_flight_info(flight.FlightDescriptor.for_path("agents"))
        table = client.do_get(info.endpoints[0].ticket).read_all()
        assert table.to_pydict() == {"agent_id": [1, 2]}
    finally:
        server.shutdown()


def test_flight_server_writes_casmservice_compatible_endpoint_file(tmp_path):
    server = start_broker_flight_server(tmp_path, ObservationBroker())
    try:
        assert (tmp_path / "arrow_endpoint.txt").read_text().strip() == f"127.0.0.1:{server.port}"
        assert (tmp_path / "arrow_endpoint.txt").stat().st_mode & 0o077 == 0
    finally:
        server.shutdown()
