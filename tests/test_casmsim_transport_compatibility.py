"""Compatibility coverage for the internal runtime transport namespace."""

from casmsocial.casmsim.flight_server import BrokerFlightServer as InternalBrokerFlightServer
from casmsocial.casmsim.grpc_control import SimulatorControlServicer as InternalControlServicer
from casmsocial.casmsim.grpc_runner import main as internal_runner_main, start_runner as internal_start_runner
from casmsocial.casmsim.observation_broker import ObservationBroker as InternalObservationBroker
from casmsocial.casmsim.repast_observation_broker import RepastObservationBrokerAdapter as InternalRepastAdapter
from casmsocial.flight_broker import BrokerFlightServer as LegacyBrokerFlightServer
from casmsocial.grpc_control import SimulatorControlServicer as LegacyControlServicer
from casmsocial.grpc_runner import main as legacy_runner_main, start_runner as legacy_start_runner
from casmsocial.observation_broker import ObservationBroker as LegacyObservationBroker
from casmsocial.repast_observation_broker import RepastObservationBrokerAdapter as LegacyRepastAdapter


def test_legacy_transport_imports_reexport_internal_runtime_types():
    """Existing callers retain the same classes while the runtime uses casmsim."""
    assert LegacyBrokerFlightServer is InternalBrokerFlightServer
    assert LegacyControlServicer is InternalControlServicer
    assert legacy_runner_main is internal_runner_main
    assert legacy_start_runner is internal_start_runner
    assert LegacyObservationBroker is InternalObservationBroker
    assert LegacyRepastAdapter is InternalRepastAdapter
