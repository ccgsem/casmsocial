"""Compatibility coverage for the standalone runner protobuf namespace."""

from casmsim.proto import casm_runner_pb2, casm_runner_pb2_grpc

from casmsocial.proto import casm_runner_pb2 as legacy_pb2, casm_runner_pb2_grpc as legacy_pb2_grpc


def test_legacy_proto_imports_reexport_the_standalone_contract():
    assert legacy_pb2.StartRequest is casm_runner_pb2.StartRequest
    assert legacy_pb2.RUN_STATE_COMPLETED == casm_runner_pb2.RUN_STATE_COMPLETED
    assert legacy_pb2_grpc.SimulatorControlStub is casm_runner_pb2_grpc.SimulatorControlStub
