"""Compatibility coverage for the internal runner protobuf namespace."""

from casmsocial.casmsim.proto import casm_runner_pb2 as internal_pb2, casm_runner_pb2_grpc as internal_pb2_grpc
from casmsocial.proto import casm_runner_pb2 as legacy_pb2, casm_runner_pb2_grpc as legacy_pb2_grpc


def test_legacy_proto_imports_reexport_the_internal_contract():
    assert legacy_pb2.StartRequest is internal_pb2.StartRequest
    assert legacy_pb2.RUN_STATE_COMPLETED == internal_pb2.RUN_STATE_COMPLETED
    assert legacy_pb2_grpc.SimulatorControlStub is internal_pb2_grpc.SimulatorControlStub
