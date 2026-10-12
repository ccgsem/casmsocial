"""Inject recoverable startup and fatal rank-local errors into two MPI ranks."""

import json
import sys
from unittest.mock import Mock

from casmsim.grpc_runner import SimulatorControlServicer
from casmsim.observation_broker import ObservationBroker
from casmsim.proto import casm_runner_pb2 as pb
from mpi4py import MPI

from casmsocial import grpc_runner


def main(phase, failing_rank):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    assert world.Get_size() == 2
    message = f"injected {phase} failure on rank {rank}"
    recoverable = phase in {"invalid_config", "startup_interrupt", "server_startup"}

    class Comm:
        """Delegate to real MPI, with injection at the final barrier boundary."""

        Get_rank = world.Get_rank
        Get_size = world.Get_size
        bcast = world.bcast
        Abort = world.Abort

        def Barrier(self):
            if phase == "completion" and rank == failing_rank:
                raise RuntimeError(message)
            world.Barrier()

    comm = Comm()

    class Adapter:
        def __init__(self, actual_comm, params):
            self.observer = None
            if phase == "constructor" and rank == failing_rank:
                raise RuntimeError(message)
            world.Barrier()

        def add_observer(self, observer):
            self.observer = observer

        def cancel(self):
            pass

        def start(self):
            if phase == "start" and rank == failing_rank:
                raise RuntimeError(message)
            world.Barrier()
            if phase == "flush" and rank == failing_rank:
                raise RuntimeError(message)
            if self.observer is not None:
                self.observer.flush()

    grpc_runner.resolve_adapter = lambda actual_comm, params: Adapter(actual_comm, params)
    record = {"rank": rank}
    if phase == "server_startup":
        grpc_runner.get_comm = lambda: comm

        def fail_control(*args, **kwargs):
            raise RuntimeError(message)

        grpc_runner.start_control_server = fail_control
        sys.argv = ["casmsocial-runner", "--run-dir", "unused-test-run-directory"]
        try:
            grpc_runner.main()
        except RuntimeError as error:
            assert rank == 0 and str(error) == message
            record["startup_failed"] = True
        else:
            assert rank == 1
            record["worker_released"] = True
    elif rank == 0:
        handoff = grpc_runner._MainThreadRunHandoff()
        broker = ObservationBroker()
        if phase == "startup_interrupt":

            def interrupt_wait():
                raise KeyboardInterrupt()

            handoff.wait_for_request = interrupt_wait
            try:
                grpc_runner._rank0_main(comm, handoff, broker)
            except KeyboardInterrupt:
                record["startup_failed"] = True
            try:
                handoff.submit("late-run", b"{}")
            except RuntimeError:
                record["late_request_rejected"] = True
        else:
            servicer = SimulatorControlServicer(broker, handoff.submit, handoff.cancel)
            context = Mock()
            config = b"[]" if phase == "invalid_config" else b'{"runner.entry_point":"test:Adapter"}'
            servicer.Start(pb.StartRequest(run_id="run-1", config_json=config), context)
            grpc_runner._rank0_main(comm, handoff, broker)
            servicer._worker.join(timeout=5)
            assert not servicer._worker.is_alive()
            state = servicer.GetState(pb.GetStateRequest(run_id="run-1"), context)
            assert state.state == pb.RUN_STATE_FAILED
            record["startup_failed"] = True
        assert broker.closed
    else:
        grpc_runner._worker_main(comm)
        record["worker_released"] = True

    if not recoverable:
        raise AssertionError("fatal MPI failure returned instead of aborting the job")
    records = world.allgather(record)
    assert records[0]["startup_failed"] and records[1]["worker_released"], records
    if phase == "startup_interrupt":
        assert records[0]["late_request_rejected"], records
    if rank == 0:
        print(json.dumps({"phase": phase, "ranks": records}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
