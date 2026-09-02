import pyarrow as pa

from casmsocial.casmpop import CasmPop
from casmsocial.observation_broker import ObservationBroker
from casmsocial.observer import Observer
from casmsocial.repast_observation_broker import RepastObservationBrokerAdapter


class _Model:
    def get_observer_output_tables(self):
        return {"agents": pa.table({"id": [1]}), "places": pa.table({"id": [2]})}


def test_repast_adapter_publishes_selected_model_owned_channels_and_closes():
    broker = ObservationBroker()
    adapter = RepastObservationBrokerAdapter(broker, channels={"agents"})

    adapter.on_step(_Model())
    adapter.on_end(_Model())

    assert broker.read("agents").batches[0].table.to_pydict() == {"id": [1]}
    assert broker.read("places").batches == ()
    assert broker.read("agents").closed is True


class _CurrentTickLogger(Observer):
    def __init__(self):
        super().__init__("CurrentTickLogger")
        self.table = None

    def on_step(self, model):
        self.table = pa.table({"id": [model.tick]})

    def get_output_tables(self, model):
        return {} if self.table is None else {"agents": self.table}


class _LifecycleModel:
    def __init__(self):
        self._observers = []
        self.tick = 7

    def get_observer_output_tables(self):
        tables = {}
        for observer in self._observers:
            tables.update(observer.get_output_tables(self))
        return tables


def test_repast_adapter_runs_after_model_loggers_registered_later():
    model = _LifecycleModel()
    broker = ObservationBroker()
    adapter = RepastObservationBrokerAdapter(broker, channels={"agents"})

    # The transport adapter is registered before the model creates its logger.
    CasmPop.add_observer(model, adapter)
    CasmPop.add_observer(model, _CurrentTickLogger())
    CasmPop._notify_step_observers(model)

    assert broker.read("agents").batches[0].table.to_pydict() == {"id": [7]}
