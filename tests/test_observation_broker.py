from __future__ import annotations

import pyarrow as pa
import pytest

from casmsocial.observation_broker import (
    ObservationBackpressureError,
    ObservationBroker,
    ObservationBrokerClosedError,
    ObservationBrokerLimits,
    ObservationCursorExpiredError,
    RetentionPolicy,
)


def _table(value: int) -> pa.Table:
    return pa.table({"value": [value]})


def test_broker_assigns_ordered_per_channel_batch_ids():
    broker = ObservationBroker()

    first = broker.publish("agents", _table(1))
    other = broker.publish("places", _table(2))
    second = broker.publish("agents", _table(3))

    assert (first.batch_id, other.batch_id, second.batch_id) == (0, 0, 1)
    read = broker.read("agents")
    assert [batch.table.column("value").to_pylist() for batch in read.batches] == [[1], [3]]
    assert read.next_batch_id == 2


def test_broker_evicts_oldest_and_rejects_expired_resume_cursor():
    broker = ObservationBroker(ObservationBrokerLimits(max_batches_per_channel=2, max_bytes_per_channel=1024))
    for value in range(3):
        broker.publish("agents", _table(value))

    retained = broker.read("agents", start_batch_id=1)
    assert [batch.batch_id for batch in retained.batches] == [1, 2]
    with pytest.raises(ObservationCursorExpiredError, match="retains batches from 1"):
        broker.read("agents", start_batch_id=0)


def test_broker_fails_publish_when_non_evicting_limit_is_reached():
    broker = ObservationBroker(
        ObservationBrokerLimits(
            max_batches_per_channel=1,
            max_bytes_per_channel=1024,
            retention_policy=RetentionPolicy.FAIL_PUBLISH,
        )
    )
    broker.publish("agents", _table(1))

    with pytest.raises(ObservationBackpressureError, match="retention limit"):
        broker.publish("agents", _table(2))


def test_broker_rejects_oversized_batches_and_invalid_future_cursors():
    broker = ObservationBroker(ObservationBrokerLimits(max_batches_per_channel=2, max_bytes_per_channel=1))
    with pytest.raises(ObservationBackpressureError, match="exceeds"):
        broker.publish("agents", _table(1))

    with pytest.raises(ValueError, match="must not be negative"):
        broker.read("agents", start_batch_id=-1)
    with pytest.raises(ValueError, match="after channel"):
        broker.read("agents", start_batch_id=1)


def test_broker_closure_is_visible_to_readers_and_prevents_publish():
    broker = ObservationBroker()
    broker.publish("agents", _table(1))
    broker.close()

    assert broker.read("agents").closed is True
    with pytest.raises(ObservationBrokerClosedError, match="after broker closure"):
        broker.publish("agents", _table(2))
