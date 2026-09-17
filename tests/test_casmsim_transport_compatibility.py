"""Compatibility coverage for public CASMSocial transport imports."""

from casmsim.flight_server import BrokerFlightServer
from casmsim.grpc_runner import SimulatorControlServicer
from casmsim.observation_broker import ObservationBroker

from casmsocial.flight_broker import BrokerFlightServer as LegacyBrokerFlightServer
from casmsocial.grpc_control import SimulatorControlServicer as LegacyControlServicer
from casmsocial.observation_broker import ObservationBroker as LegacyObservationBroker


def test_legacy_transport_imports_reexport_standalone_runtime_types():
    assert LegacyBrokerFlightServer is BrokerFlightServer
    assert LegacyControlServicer is SimulatorControlServicer
    assert LegacyObservationBroker is ObservationBroker
