"""Two-rank cancellation harness, launched by test_runner_mpi_cancellation."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread, main_thread
from unittest.mock import Mock

import pyarrow as pa
from casmsim.grpc_runner import SimulatorControlServicer
from casmsim.observation_broker import ObservationBroker
from casmsim.proto import casm_runner_pb2 as pb
from mpi4py import MPI

from casmsocial import grpc_runner
from casmsocial.casmpop import CasmPop


def main(phase):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert comm.Get_size() == 2
    entered, cancel_done = Event(), Event()
    record = {"rank": rank, "cancel_calls": 0, "constructed": False, "started": False}

    def gate():
        entered.set()
        assert cancel_done.wait(5), "cancellation did not finish"

    class Model:
        """Collective initialization plus the real CasmPop cancellation check."""

        def __init__(self, actual_comm, params):
            assert current_thread() is main_thread()
            assert actual_comm is comm
            self.comm, self.rank, self.size = comm, rank, comm.Get_size()
            self._cancel_event = Event()
            self.observer = None
            self.runner = Mock()
            self.runner.schedule.tick = 1
            self.cal = Mock()
            if rank == 0 and phase == "constructor":
                gate()
            comm.Barrier()
            record["constructed"] = True

        def add_observer(self, observer):
            self.observer = observer
            if phase == "observer_registration":
                gate()
            elif phase == "flush":
                original_flush = observer._observer.flush

                def flush():
                    gate()
                    original_flush()

                observer._observer.flush = flush

        def cancel(self):
            record["cancel_calls"] += 1
            CasmPop.cancel(self)

        def start(self):
            assert current_thread() is main_thread()
            # Workers must reach this collective even when rank 0 was
            # cancelled before construction or before model.start().
            comm.Barrier()
            record["started"] = True
            if phase in {"running", "flush"}:
                # One completed tick before the request arrives.
                self.cal.increment(60)
            if rank == 0 and phase == "running":
                gate()
            if phase != "flush":
                CasmPop.step(self)
            record["stopped"] = self.runner.schedule_stop.called
            record["calendar_steps"] = self.cal.increment.call_count
            self.ranks = comm.allgather(rank)
            if self.observer is not None:
                self.observer.on_end(self)

        def get_observer_output_tables(self):
            return {"ranks": pa.table({"rank": self.ranks})}

    factory = Mock()
    factory.Models.create_model.return_value = Model
    sys.modules["casmsocial.factory"] = factory
    sys.modules["casmsocial.__main__"] = Mock()
    if rank == 0 and phase == "resolver":
        original_resolve = grpc_runner.resolve_adapter

        def resolve(actual_comm, params):
            gate()
            return original_resolve(actual_comm, params)

        grpc_runner.resolve_adapter = resolve

    if rank == 0:
        handoff = grpc_runner._MainThreadRunHandoff()
        broker = ObservationBroker()
        servicer = SimulatorControlServicer(broker, handoff.submit, handoff.cancel)
        context = Mock()
        servicer.Start(pb.StartRequest(run_id="run-1", config_json=b'{"model.name":"test"}'), context)

        def request_cancel():
            try:
                if phase != "queued":
                    assert entered.wait(5), "model did not reach cancellation gate"
                request = pb.CancelRequest(run_id="run-1")
                assert servicer.Cancel(request, context).acknowledged
                assert servicer.Cancel(request, context).acknowledged
                assert servicer.GetState(pb.GetStateRequest(run_id="run-1"), context).state == pb.RUN_STATE_RUNNING
            finally:
                cancel_done.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            if phase == "queued":
                request_cancel()
            future = pool.submit(request_cancel) if phase != "queued" else None
            grpc_runner._rank0_main(comm, handoff, broker)
            if future is not None:
                future.result(timeout=5)
        servicer._worker.join(timeout=5)
        record["terminal_state"] = servicer.GetState(pb.GetStateRequest(run_id="run-1"), context).state
        record["terminal_cancel_acknowledged"] = servicer.Cancel(pb.CancelRequest(run_id="run-1"), context).acknowledged
        record["broker_closed"] = broker.closed
        record["output_ranks"] = broker.read("ranks").batches[0].table["rank"].to_pylist()
    else:
        grpc_runner._worker_main(comm)

    records = comm.allgather(record)
    assert all(item["constructed"] and item["started"] for item in records), records
    assert [item["cancel_calls"] for item in records] == [1, 0], records
    assert all(item["stopped"] == (phase != "flush") for item in records), records
    assert all(item["calendar_steps"] == (1 if phase in {"running", "flush"} else 0) for item in records), records
    assert records[0]["terminal_state"] == pb.RUN_STATE_CANCELLED, records
    assert not records[0]["terminal_cancel_acknowledged"], records
    assert records[0]["broker_closed"] and records[0]["output_ranks"] == [0, 1], records
    if rank == 0:
        print(json.dumps({"phase": phase, "ranks": records}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
