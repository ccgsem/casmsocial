import json

from casmsocial.grpc_control import ENDPOINT_FILENAME
from casmsocial.grpc_runner import start_runner


def test_runner_writes_combined_loopback_endpoint_manifest(tmp_path):
    control, flights = start_runner(tmp_path)
    try:
        manifest = json.loads((tmp_path / ENDPOINT_FILENAME).read_text())
        assert manifest["control"]["address"].startswith("127.0.0.1:")
        assert manifest["control"]["protocol"] == "casm.runner.v1"
        assert manifest["flight"] == {
            "address": f"127.0.0.1:{flights.port}",
            "protocol": "arrow.flight",
        }
        assert (tmp_path / "arrow_endpoint.txt").read_text().strip() == f"127.0.0.1:{flights.port}"
    finally:
        control.stop(0).wait()
        flights.shutdown()
