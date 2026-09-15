"""Compatibility coverage for the internal runtime transport namespace."""

from casmsocial.casmsim.flight_server import BrokerFlightServer as InternalBrokerFlightServer
from casmsocial.casmsim.grpc_control import SimulatorControlServicer as InternalControlServicer
from casmsocial.casmsim.observation_broker import ObservationBroker as InternalObservationBroker
from casmsocial.casmsim.repast_observation_broker import RepastObservationBrokerAdapter as InternalRepastAdapter
from casmsocial.flight_broker import BrokerFlightServer as LegacyBrokerFlightServer
from casmsocial.grpc_control import SimulatorControlServicer as LegacyControlServicer
from casmsocial.observation_broker import ObservationBroker as LegacyObservationBroker
from casmsocial.repast_observation_broker import RepastObservationBrokerAdapter as LegacyRepastAdapter


def test_legacy_transport_imports_reexport_internal_runtime_types():
    """Existing callers retain the same classes while the runtime uses casmsim."""
    assert LegacyBrokerFlightServer is InternalBrokerFlightServer
    assert LegacyControlServicer is InternalControlServicer
    assert LegacyObservationBroker is InternalObservationBroker
    assert LegacyRepastAdapter is InternalRepastAdapter
