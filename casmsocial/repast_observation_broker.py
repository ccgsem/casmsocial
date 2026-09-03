"""Repast4py observer adapter that publishes model-owned Arrow outputs."""

from __future__ import annotations

from typing import Protocol

import pyarrow as pa

from casmsocial.observation_broker import ObservationBroker
from casmsocial.observer import Observer


class ObserverOutputModel(Protocol):
    def get_observer_output_tables(self) -> dict[str, pa.Table]: ...


class RepastObservationBrokerAdapter(Observer):
    """Publish each repast4py observer-table snapshot to the shared broker."""

    # Run after model-owned observers such as AgentLogger. A launcher may
    # register this adapter before model initialization, while those loggers
    # are registered during build_context.
    step_priority = 100

    def __init__(self, broker: ObservationBroker, channels: set[str] | None = None) -> None:
        super().__init__("RepastObservationBrokerAdapter")
        self._broker = broker
        self._channels = channels

    def on_step(self, model: ObserverOutputModel) -> None:
        for channel, table in model.get_observer_output_tables().items():
            if self._channels is None or channel in self._channels:
                self._broker.publish(channel, table)

    def on_end(self, model: ObserverOutputModel) -> None:
        self._broker.close()
