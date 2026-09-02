import json
import time
from threading import Event

import grpc
import pyarrow as pa
import pyarrow.flight as flight

from casmsocial.flight_broker import BrokerFlightServer
from casmsocial.grpc_control import ENDPOINT_FILENAME, start_control_server
from casmsocial.observation_broker import ObservationBroker
from casmsocial.proto import casm_runner_pb2 as pb2, casm_runner_pb2_grpc as pb2_grpc


def test_grpc_and_flight_return_same_broker_observations(tmp_path):
    broker = ObservationBroker()
    started = Event()
    release = Event()

    def start_run(run_id, config_json):
        assert run_id == "run-1"
        assert config_json == b"{}"
        started.set()
        assert release.wait(timeout=2)
        broker.publish("agents", pa.table({"id": [1]}))
        broker.publish("agents", pa.table({"id": [2]}))
        broker.close()

    control = start_control_server(tmp_path, broker, start_run)
    flights = BrokerFlightServer(("127.0.0.1", 0), broker)
    try:
        endpoint = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())["control"]["address"]
        channel = grpc.insecure_channel(endpoint)
        stub = pb2_grpc.SimulatorControlStub(channel)
        stub.Start(pb2.StartRequest(run_id="run-1", config_json=b"{}"))
        assert started.wait(timeout=2)
        assert stub.GetState(pb2.GetStateRequest(run_id="run-1")).state == pb2.RUN_STATE_RUNNING
        assert stub.Cancel(pb2.CancelRequest(run_id="run-1")).acknowledged is False
        assert stub.GetState(pb2.GetStateRequest(run_id="run-1")).state == pb2.RUN_STATE_RUNNING
        release.set()
        for _ in range(100):
            if stub.GetState(pb2.GetStateRequest(run_id="run-1")).state == pb2.RUN_STATE_COMPLETED:
                break
            time.sleep(0.01)
        streamed = list(stub.StreamObs(pb2.StreamObsRequest(run_id="run-1", channel="agents")))
        streamed_tables = [pa.ipc.open_stream(pa.py_buffer(batch.arrow_ipc)).read_all() for batch in streamed]
        flight_client = flight.connect(f"grpc://127.0.0.1:{flights.port}")
        info = flight_client.get_flight_info(flight.FlightDescriptor.for_path("agents"))
        flight_table = flight_client.do_get(info.endpoints[0].ticket).read_all()
        assert [batch.tick for batch in streamed] == [0, 1]
        assert pa.concat_tables(streamed_tables).equals(flight_table)
        assert stub.GetState(pb2.GetStateRequest(run_id="run-1")).state == pb2.RUN_STATE_COMPLETED
        channel.close()
    finally:
        control.stop(0).wait()
        flights.shutdown()
