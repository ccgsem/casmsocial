"""Compatibility coverage for the internal runtime transport namespace."""

from casmsocial.casmsim.flight_server import BrokerFlightServer as InternalBrokerFlightServer
from casmsocial.casmsim.observation_broker import ObservationBroker as InternalObservationBroker
from casmsocial.flight_broker import BrokerFlightServer as LegacyBrokerFlightServer
from casmsocial.observation_broker import ObservationBroker as LegacyObservationBroker


def test_legacy_transport_imports_reexport_internal_runtime_types():
    """Existing callers retain the same classes while the runtime uses casmsim."""
    assert LegacyBrokerFlightServer is InternalBrokerFlightServer
    assert LegacyObservationBroker is InternalObservationBroker
